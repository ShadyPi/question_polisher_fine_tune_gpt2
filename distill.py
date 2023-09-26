import torch
import transformers
import yaml

from utility import dataset_access, query_assemble, llm, metrics
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_encode(params, batch_data, tokenizer):
    inputs = [base + ' ' + polished for base, polished in batch_data]
    inputs = tokenizer.batch_encode_plus(
        inputs,
        max_length=model_params['MAX_INPUT_KG_LENGTH'],
        pad_to_max_length=True,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
    )
    bases = [base for base, polished in batch_data]
    bases = tokenizer.batch_encode_plus(
        bases,
        max_length=model_params['MAX_INPUT_KG_LENGTH'],
        pad_to_max_length=True,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
    )
    return {
        'inputs_ids': inputs['input_ids'].to(dtype=torch.long),
        'inputs_mask': inputs['attention_mask'].to(dtype=torch.long),
        'bases_ids': bases['input_ids'].to(dtype=torch.long),
        'bases_mask': bases['attention_mask'].to(dtype=torch.long),
    }


def distill(tokenizer, model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    ids = data['inputs_ids'].to(device, dtype=torch.long)
    ids[data['inputs_ids'] == tokenizer.pad_token_id] = 0
    mask = data['inputs_mask'].to(device, dtype=torch.long)
    bases_mask = data['bases_mask'].to(device, dtype=torch.long)
    labels = data['inputs_ids'].clone().detach()
    labels[data['inputs_ids'] == tokenizer.pad_token_id] = -100
    for index, base_mask in zip(range(labels.size(0)), bases_mask):
        labels[index, :base_mask.sum()] = torch.tensor([-100 for i in range(base_mask.sum())]).to(device)
    labels = labels.to(device, dtype=torch.long)
    outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
    loss = outputs.loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def model_polish(tokenizer, model, text):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
        print(input_ids.size)
        polished_ids = model.generate(
            input_ids=input_ids,
            do_sample=True,
            max_length=50 + input_ids.size(-1),
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
        )
        polished_text = tokenizer.decode(
            polished_ids[input_ids.size(-1):],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return polished_text


def evaluate(epoch, tokenizer, model):
    raw_datasets_path = r'./raw_datasets/'
    results_path = r'./polished_test_results/'
    config_path = results_path + 'config.yaml'
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)
    raw_datasets = config['raw_datasets']
    data_formats = config['data_formats']
    test_config = config['test_config']
    start_point, end_point = config['start_point'], config['end_point']
    part = 'test'
    if start_point < 0:
        start_point = None
    if end_point < 0:
        end_point = None

    for dataset in raw_datasets:
        assert dataset in data_formats.keys(), 'Data format not defined!'
        assert data_formats[dataset] in ['jsonl', 'json'], 'Invalid data format'
        raw_dataset_path = raw_datasets_path + dataset + '/' + part + '.' + data_formats[dataset]
        save_path = results_path+dataset+'/results_'+part+'_'+str(start_point)+'_'+str(end_point)+'_'+str(epoch)+'.jsonl'
        if data_formats[dataset] == 'jsonl':
            data = dataset_access.load_jsonl(raw_dataset_path, start_point, end_point)
        else:
            data = dataset_access.load_json(raw_dataset_path, start_point, end_point)
        if dataset == 'GSM8K':
            with open(r'./augmentation/demo.txt', 'r') as f:
                demo = f.read()
            queries = [query_assemble.score_GSM8K(demo, model_polish(tokenizer, model, item['question'])) for item in data]
        results = llm.async_query(test_config, queries)
        correct = 0
        for index in range(len(data)):
            answer = int(data[index]['answer'].split('####')[1].replace(',', ''))
            output = metrics.clean_response(dataset, results[index][0])
            data[index]['polished'] = queries[index]
            data[index]['answer'] = answer
            data[index]['output'] = output
            data[index]['full_response'] = results[index][0]
            correct += (answer == output)
            # print(results[index][0])
            # print(answer, output)
        dataset_access.save_jsonl(save_path, data)
        return correct / len(data)


if __name__ == '__main__':
    model_name = 'gpt2'
    model_params = {
        "TRAIN_BATCH_SIZE": 64,  # batch size within each alternative training loop
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
    output_dir = r'./distilledGPT/'

    # Prepare the data
    data_path = r'./augmentation/GSM8K/training_data.jsonl'
    raw_data = dataset_access.load_jsonl(data_path)
    train_data = []
    for item in raw_data:
        train_data.append((item['base'], item['polished']))
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    np.random.shuffle(train_data)
    batched_data = [train_data[i: i + model_params['TRAIN_BATCH_SIZE']] for i in
                    range(0, len(train_data), model_params['TRAIN_BATCH_SIZE'])]

    # Load the pre-trained GPT-2 model
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = transformers.GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
    model = model.to(device)

    # Fine-tune the model
    optimizer = torch.optim.Adam(params=model.parameters(), lr=model_params["LEARNING_RATE_KG"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    model.to(device)

    best_acc = 0
    for epoch in range(model_params['TRAIN_EPOCHS']):
        epoch_loss = 0
        for step, batch in enumerate(batched_data):
            encoded_data = batch_encode(model_params, batch, tokenizer)
            loss = distill(tokenizer, model, encoded_data, optimizer)
            epoch_loss += loss
            print(f"Step {step}/{len(batched_data)} - Loss: {loss:.4f}")
        print(f"Epoch {epoch + 1}/{model_params['TRAIN_EPOCHS']} - Loss: {epoch_loss / len(batched_data)}")
        scheduler.step()
        eval_acc = evaluate(epoch, tokenizer, model)
        print(f"Epoch {epoch + 1}/{model_params['TRAIN_EPOCHS']} - Eval_Acc: {eval_acc}")
        if eval_acc > best_acc:
            best_acc = eval_acc
            model_dir = output_dir+'epoch'+str(epoch)
            dataset_access.save_model(model_dir, tokenizer, model)
