from utility import dataset_access

result_file_path = r'./polished_test_results/GSM8K/results_test_0_99_few_shot.jsonl'
result = dataset_access.load_jsonl(result_file_path)
correct = 0
for item in result:
    if item['answer'] == item['output']:
        correct += 1
print(correct/len(result))
