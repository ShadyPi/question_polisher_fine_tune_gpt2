import csv
from collections import Counter
import yaml
from utility import dataset_access, llm, query_assemble

raw_datasets_path = r'./raw_datasets/'
augmentation_path = r'./augmentation/'
config_path = augmentation_path + 'config.yaml'


def augment(dataset, item, examiner_config):
    if dataset == 'GSM8K':
        query = query_assemble.augment_GSM8K(item)
    response = llm.get_response(examiner_config, query)
    augmentations = []
    for choice in response['choices']:
        if choice['finish_reason'] == 'stop':
            augmentations.append(choice['message']['content'])
    return augmentations


def exam(dataset, item, ground_truth, examinee_config):
    with open(r'./augmentation/demo.txt', 'r') as f:
        demo = f.read()
    if dataset == 'GSM8K':
        query = query_assemble.score_GSM8K(demo, item)
    response = llm.get_response(examinee_config, query)
    answers = []
    total = examinee_config['n']
    for choice in response['choices']:
        try:
            answer = int(choice['message']['content'])
            answers.append(answer)
        except ValueError as e:
            continue
    if len(answers) == 0:
        print(response['choices'])
        return -99999, 0
    count = Counter(answers)
    predict, times = count.most_common(1)[0]
    print(answers)
    return predict, times/total, count[ground_truth]/total


if __name__ == '__main__':
    # load config
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)
    raw_datasets = config['raw_datasets']
    data_formats = config['data_formats']
    examiner_config = config['examiner_config']
    examinee_config = config['examinee_config']
    start_point, end_point = config['start_point'], config['end_point']
    # augment question
    for dataset in raw_datasets:
        assert dataset in data_formats.keys(), 'Data format not defined!'
        assert data_formats[dataset] in ['jsonl', 'json'], 'Invalid data format'
        raw_dataset_path = raw_datasets_path+dataset+'/train.'+data_formats[dataset]
        if data_formats[dataset] == 'jsonl':
            data = dataset_access.load_jsonl(raw_dataset_path, start_point, end_point)
        else:
            data = dataset_access.load_json(raw_dataset_path, start_point, end_point)
        score_list = []
        for item in data:
            answer = int(item['answer'].split('####')[1])
            predict, most, score = exam(dataset, item['question'], answer, examinee_config)
            score_list.append([predict == answer, most, score])
            if len(score_list) % 10 == 0:
                print(len(score_list))
        save_path = raw_datasets_path+dataset+'/check_'+str(start_point)+'_'+str(end_point)+'.csv'
        with open(save_path, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(score_list)
