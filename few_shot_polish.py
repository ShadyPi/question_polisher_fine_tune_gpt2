import random

import transformers
import yaml
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from string import Template
from utility import dataset_access, query_assemble, llm, metrics
import numpy as np


def get_polish_query(base_text, base_ids, polish_map, tokenizer, text):
    few_shot = retrieve_demo(base_ids, tokenizer, text, 4)
    demo_base = [base_text[i] for i in few_shot]
    demo_polished = [polish_map[i] for i in demo_base]
    polish_query = query_assemble.polish_query(demo_base, demo_polished, text)
    return polish_query


def retrieve_demo(demo_ids, tokenizer, text, demo_num):
    text_ids = tokenizer.encode_plus(
        text,
        max_length=150,
        pad_to_max_length=True,
        truncation=True,
        padding='max_length',
        return_tensors='np'
    )
    text_ids = text_ids['input_ids'].astype(dtype='int64')
    demo_len = np.sqrt(np.sum(demo_ids*demo_ids, axis=1))
    text_len = np.sqrt(np.sum(text_ids*text_ids))
    similarity = (text_ids * demo_ids.T)/demo_len/text_len
    idx = similarity.argsort().tolist()[0][-demo_num:]
    # return random.sample(range(len(demo_ids)), demo_num)
    return []
    # return idx


if __name__ == '__main__':
    demo_path = r'./augmentation/GSM8K/demos.jsonl'
    demos = dataset_access.load_jsonl(demo_path)
    polish_map = {}
    base_texts = []
    for item in demos:
        polish_map[item['base']] = item['polished']
        base_texts.append(item['base'])
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    base_ids = tokenizer.batch_encode_plus(
        base_texts,
        max_length=150,
        pad_to_max_length=True,
        truncation=True,
        padding='max_length',
        return_tensors='np'
    )
    base_ids = base_ids['input_ids'].astype(dtype='int64')
    base_ids[base_ids == tokenizer.pad_token_id] = 0

    test_data_path = r'./raw_datasets/GSM8K/test.jsonl'
    results_path = r'./polished_test_results/'
    config_path = results_path + 'config.yaml'
    dataset = 'GSM8K'
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)
    raw_datasets = config['raw_datasets']
    data_formats = config['data_formats']
    test_config = config['test_config']
    start_point, end_point = config['start_point'], config['end_point']
    if start_point < 0:
        start_point = None
    if end_point < 0:
        end_point = None
    test_data = dataset_access.load_jsonl(test_data_path, start_point, end_point)
    polish_queries = [get_polish_query(base_texts, base_ids, polish_map, tokenizer, item['question']) for item in test_data]
    # system_prompt = 'You are a helpful AI question rewriter. Please read the given exemplars carefully and rewrite the question.'
    system_prompt = 'You are a helpful AI question rewriter. Please rewrite the given question.'
    polished_questions = llm.async_query(test_config, polish_queries)
    queries = [query_assemble.score_GSM8K('', item[0]) for item in polished_questions]
    results = llm.async_query(test_config, queries)
    print(len(results))
    correct = 0
    for index in range(len(test_data)):
        answer = int(test_data[index]['answer'].split('####')[1].replace(',', ''))
        output = metrics.clean_response(dataset, results[index][0])
        test_data[index]['polished'] = queries[index]
        test_data[index]['answer'] = answer
        test_data[index]['output'] = output
        test_data[index]['full_response'] = results[index][0]
        correct += (answer == output)
        # print(results[index][0])
        # print(answer, output)
    save_path = results_path + dataset + '/results_test_' + str(start_point) + '_' + str(end_point) + '_few_shot.jsonl'
    dataset_access.save_jsonl(save_path, test_data)
    print(correct / len(test_data))
