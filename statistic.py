import csv
import os
import re

from utility import dataset_access

augmentation_path = r'./augmentation/'


def eligible(base, best, now):
    if now <= base:
        return False
    if now <= 0.5 and now - base < 0.4:
        return False
    if now - base <= 0.05:
        return False
    if best - now >= 0.3:
        return False
    return True


if __name__ == '__main__':
    start_point = 200
    end_point = 400
    dataset = 'GSM8K'
    file_list = os.listdir(augmentation_path + dataset)
    json_list = []
    for file_name in file_list:
        if file_name[-4:] == 'json':
            json_list.append(file_name)
    print(json_list)
    cnt = 0
    gap = []
    item_list = []
    demos_set = []
    for file_name in json_list:
        item = dataset_access.load_json(augmentation_path + dataset + '/' + file_name)
        item_list.append(item)
        id_num = int(re.findall(r'\d+', file_name)[0])
        base_score = item['base_question']['score']
        best_score = item['best_question']['score']
        gap.append([id_num, base_score, best_score, best_score - base_score])
        if eligible(base_score, best_score, best_score):
        # if True:
            base_question = item['base_question']['question']
            best_question = item['best_question']['question']
            demos_set.append({'base': base_question, 'polished': best_question})
            cnt += 1
    print(cnt)
    gap.sort()
    save_path = augmentation_path + dataset + '/together.csv'
    dataset_access.save_csv(save_path, gap, ['id_num', 'base', 'best', 'gap'])
    save_path = augmentation_path + dataset + '/together.jsonl'
    dataset_access.save_jsonl(save_path, item_list)
    save_path = augmentation_path + dataset + '/demos.jsonl'
    dataset_access.save_jsonl(save_path, demos_set)

    max_len = 0
    training_set = []
    for item in item_list:
        base_score = item['base_question']['score']
        best_score = item['best_question']['score']
        if not eligible(base_score, best_score, best_score):
            continue
        base_question = item['base_question']['question']
        max_len = max(max_len, len(base_question))
        for step in item['log']:
            for polished in step:
                polished_score = polished['score']
                if eligible(base_score, best_score, polished_score):
                    # print(base_score, polished_score)
                    max_len = max(max_len, len(polished['question']))
                    new_data = {'base': base_question, 'polished': polished['question']}
                    if new_data not in training_set:
                        training_set.append(new_data)

    print('dataset_size:', len(training_set))
    print('longest_question:', max_len)
    save_path = augmentation_path + dataset + '/training_data.jsonl'
    dataset_access.save_jsonl(save_path, training_set)
