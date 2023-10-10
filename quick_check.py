import yaml

from utility import dataset_access, query_assemble, llm, metrics

results_path = r'./polished_test_results/'
config_path = results_path + 'config.yaml'
dataset = 'GSM8K'
with open(config_path, 'r') as config_file:
    config = yaml.load(config_file, Loader=yaml.SafeLoader)
raw_datasets = config['raw_datasets']
data_formats = config['data_formats']
test_config = config['test_config']
model_name = test_config['model']

file_path = r'./augmentation/GSM8K/demos.jsonl'
data = dataset_access.load_jsonl(file_path)
polish_map = {}
for item in data:
    polish_map[item['base']] = item['polished']
file_path = r'./raw_datasets/GSM8K/train.jsonl'
data = dataset_access.load_jsonl(file_path)
bases = []
polishes = []
answers = []
for item in data:
    base = item['question']
    if base not in polish_map.keys():
        continue
    polished = polish_map[base]
    answer = int(item['answer'].split('####')[1].replace(',', ''))
    bases.append(base)
    polishes.append(polished)
    answers.append(answer)
print(len(bases))
base_queries = [query_assemble.score_GSM8K('', item) for item in bases]
base_results = llm.async_query(test_config, base_queries)
polished_queries = [query_assemble.score_GSM8K('', item) for item in polishes]
polish_results = llm.async_query(test_config, polished_queries)

correct_base = 0
correct_polished = 0
log = []
for index in range(len(answers)):
    if len(base_results[index]) == 0:
        base_results[index].append(None)
    base_answer = metrics.clean_response(dataset, base_results[index][0])
    if len(polish_results[index]) == 0:
        polish_results[index].append(None)
    polished_answer = metrics.clean_response(dataset, polish_results[index][0])
    correct_base += (base_answer == answers[index])
    correct_polished += (polished_answer == answers[index])
    base_item = {'output': base_answer, 'question': bases[index], 'response': base_results[index]}
    polished_item = {'output': polished_answer, 'question': polishes[index], 'response': polish_results[index]}
    new_item = {'answer': answers[index], 'base': base_item, 'polished': polished_item}
    log.append(new_item)

dict = {'correct_base': correct_base, 'correct_polished': correct_polished, 'log': log}
dataset_access.save_json(r'check.json', dict)
