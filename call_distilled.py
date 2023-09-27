from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

from utility import dataset_access


def model_polish(generator, text):
    generated = generator('<s>'+text+'</s>-><p>')
    trimmed = generated[0]['generated_text'].split('</s>-><p>')[1].split('</p>')[0]
    return trimmed


if __name__ == '__main__':
    output_dir = r'./distilledGPT/'
    model_dir = output_dir + 'Trainer'
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=256)
    test_data_path = r'./raw_datasets/GSM8K/test.jsonl'
    test_data = dataset_access.load_jsonl(test_data_path)
    for item in test_data[0: 9]:
        print(model_polish(generator, item['question']))
