import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, TextDataset, \
    TrainingArguments, Trainer, pipeline

from utility import dataset_access


def distill_trainer(text_path, epochs, model_name, batch_size, output_dir):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataset = TextDataset(tokenizer=tokenizer, file_path=text_path, block_size=256)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        warmup_steps=500,
        save_steps=2000,
        logging_steps=10,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        devices=[2],
        accelerator="gpu"
    )
    trainer.train()
    # trainer.save_model(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def model_polish(generator, text):
    generated = generator('<s>'+text+'</s>-><p>')
    trimmed = generated[0]['generated_text'].split('</s>-><p>')[1].split('</p>')[0]
    return trimmed


if __name__ == '__main__':
    model_params = {
        "TRAIN_BATCH_SIZE": 16,  # batch size within each alternative training loop
        "TRAIN_EPOCHS": 10,  # number of training epochs
        "LEARNING_RATE_KG": 1e-5,  # learning rate
        "LEARNING_RATE_INF": 1e-5,  # learning rate
        "MAX_INPUT_KG_LENGTH": 150,  # max length of all input text
        "MAX_SOURCE_KG_LENGTH": 80,  # max length of input question
        "MAX_TARGET_KG_LENGTH": 50,  # max length of target knowledge
        "MAX_SOURCE_INF_LENGTH": 150,  # max length of all input text
        "MAX_TARGET_INF_LENGTH": 10,  # max length of output answer text
        "SEED": 42,  # set seed for reproducibility
    }

    data_path = r'./augmentation/GSM8K/training_data.jsonl'
    raw_data = dataset_access.load_jsonl(data_path)
    train_data = []
    for item in raw_data:
        train_data.append((item['base'], item['polished']))
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    np.random.shuffle(train_data)
    text_path = r'./augmentation/GSM8K/training_data.txt'
    with open(text_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            new_line = '<s>' + item[0] + '</s>-><p>' + item[1] + '</p>\n'
            f.write(new_line)

    output_dir = r'./distilledGPT/'
    model_dir = output_dir + 'Trainer'
    distill_trainer(text_path, model_params['TRAIN_EPOCHS'], 'gpt2-xl', model_params['TRAIN_BATCH_SIZE'], model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=256)
    test_data_path = r'./raw_datasets/GSM8K/test.jsonl'
    test_data = dataset_access.load_jsonl(test_data_path)
