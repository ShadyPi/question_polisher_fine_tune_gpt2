import json
import os


def load_json(file_path, start=None, end=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if start is not None and end is not None:
        data = data[start: end+1]
    return data


def save_json(file_path, data):
    data = json.dumps(data)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(data)


def load_jsonl(file_path, start=None, end=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        json_list = list(f)
    if start is not None and end is not None:
        json_list = json_list[start: end+1]
    for index in range(len(json_list)):
        json_list[index] = json.loads(json_list[index])
    return json_list


def save_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            item = json.dumps(item)
            f.write(item+'\n')


def pack_trajectory(trajectory):
    cnt = 0
    log = {}
    for step in trajectory:
        candidates = []
        for candidate in step:
            temp = {'question': candidate[2], 'score': candidate[0]}
            candidates.append(temp)
        log[str(cnt)] = candidates
        cnt += 1
    return log
