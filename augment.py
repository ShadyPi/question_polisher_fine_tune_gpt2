import csv
from collections import Counter
import yaml
from utility import dataset_access, llm, query_assemble, metrics

raw_datasets_path = r'./raw_datasets/'
augmentation_path = r'./augmentation/'
config_path = augmentation_path + 'config.yaml'


def augment(dataset, question, examiner_config):
    if dataset == 'GSM8K':
        query = query_assemble.augment_GSM8K(question)
    response = llm.get_response(examiner_config, query)
    augmentations = []
    for choice in response['choices']:
        if choice['finish_reason'] == 'stop':
            augmentations.append(choice['message']['content'])
    return augmentations


def exam(dataset, question, answer, examinee_config):
    with open(r'./augmentation/demo.txt', 'r') as f:
        demo = f.read()
    if dataset == 'GSM8K':
        query = query_assemble.score_GSM8K(demo, question)
    response = llm.get_response(examinee_config, query)
    predicts = []
    total = examinee_config['n']
    for choice in response['choices']:
        predict = metrics.clean_response(dataset, choice['message']['content'])
        predicts.append(predict)
    count = Counter(predicts)
    correct = count[answer]
    print(predicts)
    return correct/total


def select(dataset, base_score, candidates, answer, beam_width, examinee_config, last_step):
    question_score = []
    for candidate in candidates:
        score = exam(dataset, candidate, answer, examinee_config)
        if score < base_score or score == 0.0:
            continue
        question_score.append((score, len(candidate), candidate))
    question_score += last_step
    question_score.sort(key=lambda x: (-x[0], x[1]))
    return question_score[:beam_width]


def polish(dataset, base_question, answer, examiner_config, examinee_config, step_num, beam_width):
    base_score = exam(dataset, base_question, answer, examinee_config)
    polish_step = [[(base_score, len(base_question), base_question)]]
    for index in range(1, step_num):
        last_step = polish_step[index-1]
        this_step = []
        min_score = 1.0
        max_score = 0.0
        for score, length, question in last_step:
            min_score = min(min_score, score)
            max_score = max(max_score, score)
        if min_score == 1.0:
            break
        for score, length, question in last_step:
            this_step += augment(dataset, question, examiner_config)
        this_step = select(dataset, min_score, this_step, answer, beam_width, examinee_config, last_step)
        if len(this_step) == 0:
            this_step = last_step
        polish_step.append(this_step)
        print('Polish step#'+str(index))
    return polish_step


if __name__ == '__main__':
    # load config
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)
    raw_datasets = config['raw_datasets']
    data_formats = config['data_formats']
    examiner_config = config['examiner_config']
    examinee_config = config['examinee_config']
    start_point, end_point = config['start_point'], config['end_point']
    step_num = config['step_num']
    beam_width = config['beam_width']
    # augment question
    for dataset in raw_datasets:
        assert dataset in data_formats.keys(), 'Data format not defined!'
        assert data_formats[dataset] in ['jsonl', 'json'], 'Invalid data format'
        raw_dataset_path = raw_datasets_path+dataset+'/train.'+data_formats[dataset]
        if data_formats[dataset] == 'jsonl':
            data = dataset_access.load_jsonl(raw_dataset_path, start_point, end_point)
        else:
            data = dataset_access.load_json(raw_dataset_path, start_point, end_point)
        trajectory_list = []
        for item in data:
            # print('Current item:', len(trajectory_list))
            answer = int(item['answer'].split('####')[1].replace(',', ''))
            trajectory = polish(dataset, item['question'], answer, examiner_config, examinee_config, step_num, beam_width)
            package = dataset_access.pack_trajectory(trajectory)
            id_num = str(start_point+len(trajectory_list))
            save_path = augmentation_path+dataset+'/trajectory_'+id_num+'.json'
            dataset_access.save_jsonl(save_path, [package])
            print('Current item:', id_num, package['base_question']['score'], package['best_question']['score'])
            trajectory_list.append(package)
        save_path = augmentation_path+dataset+'/trajectory_'+str(start_point)+'_'+str(end_point)+'.jsonl'
        dataset_access.save_jsonl(save_path, trajectory_list)
