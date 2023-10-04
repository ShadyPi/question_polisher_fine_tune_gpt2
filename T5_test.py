from utility import dataset_access, llm, query_assemble, metrics
from transformers import T5Tokenizer, T5ForConditionalGeneration


if __name__ == '__main__':
    dataset = 'GSM8K'
    raw_data_path = r'./raw_datasets/GSM8K/test.jsonl'
    raw_data = dataset_access.load_jsonl(raw_data_path, 0, 99)

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto")

    correct = 0
    for item in raw_data:
        input_text = item['question']
        answer = int(item['answer'].split('####')[1].replace(',', ''))
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(input_ids)
        output_text = tokenizer.decode(outputs[0])
        output_answer = metrics.clean_response(dataset, output_text)
        raw_data['response'] = output_text
        raw_data['output'] = output_answer
        correct += (answer == output_answer)
    print(correct)

    save_path = r'./test_results/t5.jsonl'
    dataset_access.save_jsonl(save_path, raw_data)


