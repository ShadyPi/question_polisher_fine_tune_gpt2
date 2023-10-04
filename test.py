import yaml
from utility import dataset_access, llm, query_assemble, metrics


raw_datasets_path = r'./raw_datasets/'
results_path = r'./test_results/'
config_path = results_path+'config.yaml'

if __name__ == '__main__':
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
        raw_dataset_path = raw_datasets_path+dataset+'/'+part+'.'+data_formats[dataset]
        save_path = results_path + dataset + '/results_' + part + '_' + str(start_point) + '_' + str(end_point) + '.jsonl'
        if data_formats[dataset] == 'jsonl':
            data = dataset_access.load_jsonl(raw_dataset_path, start_point, end_point)
        else:
            data = dataset_access.load_json(raw_dataset_path, start_point, end_point)
        if dataset == 'GSM8K':
            with open(r'./test_results/empty_demo.txt', 'r') as f:
                demo = f.read()
            queries = [query_assemble.score_GSM8K(demo, item['question']+'Let\'s think step by step.\n\n') for item in data]
        results = llm.async_query(test_config, queries)
        print(len(results))
        correct = 0
        for index in range(len(data)):
            answer = int(data[index]['answer'].split('####')[1].replace(',', ''))
            output = metrics.clean_response(dataset, results[index][0])
            data[index]['answer'] = answer
            data[index]['output'] = output
            data[index]['full_response'] = results[index][0]
            correct += (answer == output)
            # print(results[index][0])
            # print(answer, output)
        dataset_access.save_jsonl(save_path, data)
        print(correct/len(data))
