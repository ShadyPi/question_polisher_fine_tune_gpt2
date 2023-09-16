import csv
from utility import dataset_access

augmentation_path = r'./augmentation/'

if __name__ == '__main__':
    start_point = 0
    end_point = 99
    dataset = 'GSM8K'
    file_path = augmentation_path + dataset + '/trajectory_' + str(start_point) + '_' + str(end_point) + '.jsonl'
    data = dataset_access.load_jsonl(file_path)
    gap = []
    cnt = 0
    for item in data:
        base_score = item['base_question']['score']
        best_score = item['best_question']['score']
        gap.append([base_score, best_score])
        if best_score > base_score and best_score > 0.5:
            cnt += 1
    print(cnt)
    # save_path = augmentation_path+dataset+'/statistics_'+str(start_point)+'_'+str(end_point)+'.csv'
    # dataset_access.save_csv(save_path, gap)
