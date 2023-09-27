from utility import dataset_access

plain_test_file = r'./test_results/GSM8K/results_test_0_99.jsonl'
polished_test_file = r'./polished_test_results/GSM8K/results_test_0_99_few_shot.jsonl'
plain = dataset_access.load_jsonl(plain_test_file)
polished = dataset_access.load_jsonl(polished_test_file)
total_win = 0
total_lose = 0
for index in range(len(plain)):
    plain_output = plain[index]['output']
    polished_output = polished[index]['output']
    answer = plain[index]['answer']
    win = (plain_output != answer) and (polished_output == answer)
    lose = (plain_output == answer) and (polished_output != answer)
    total_win += win
    total_lose += lose
    if win:
        print('=====WIN=====')
    if lose:
        print('====LOSE====')
    if win or lose:
        print('base:', plain[index]['question'])
        print('polished:', polished[index]['polished'])
        print('--------------------')
print(total_win, total_lose)
