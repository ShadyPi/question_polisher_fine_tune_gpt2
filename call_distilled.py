import yaml
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

from utility import dataset_access, query_assemble, llm, metrics


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
    queries = [query_assemble.score_GSM8K('', item['question']) for item in test_data]
    results = llm.async_query(test_config, queries)
    print(len(results))
    correct = 0
    for index in range(len(test_data)):
        answer = int(test_data[index]['answer'].split('####')[1].replace(',', ''))
        output = metrics.clean_response(dataset, results[index][0])
        test_data[index]['answer'] = answer
        test_data[index]['output'] = output
        test_data[index]['full_response'] = results[index][0]
        correct += (answer == output)
        # print(results[index][0])
        # print(answer, output)
    save_path = results_path + dataset + '/results_test_' + str(start_point) + '_' + str(end_point) + '_polished.jsonl'
    dataset_access.save_jsonl(save_path, test_data)
    print(correct / len(test_data))
