import openai
import os
import asyncio
import time

openai.api_key = os.getenv('OPENAI_API_KEY')
system_prompt = 'You are a helpful assistant.'


def get_response(LLM_config, text):
    response = None
    while response is None:
        try:
            if LLM_config['model'] == 'gpt-3.5-turbo':
                messages = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': text}
                ]
                response = openai.ChatCompletion.create(
                    model=LLM_config['model'],
                    messages=messages,
                    temperature=LLM_config['temperature'],
                    max_tokens=LLM_config['max_tokens'],
                    frequency_penalty=LLM_config['frequency_penalty'],
                    presence_penalty=LLM_config['presence_penalty'],
                    n=LLM_config['n'],
                    request_timeout=60
                )
                return response
        except Exception as e:
            print(e)
            print("Retrying....")
            time.sleep(30)


async def async_get_response(LLM_config, texts):
    if LLM_config['model'] == 'gpt-3.5-turbo':
        messages = [
            [{'role': 'system', 'content': system_prompt},
             {'role': 'user', 'content': text}]
            for text in texts]
        batch_size = LLM_config['batch_size']
        batched_messages = [messages[i: i + batch_size] for i in range(0, len(messages), batch_size)]
        responses = []
        for batch_num, message_batch in enumerate(batched_messages):
            async_responses = None
            while async_responses is None:
                try:
                    async_request = [openai.ChatCompletion.acreate(
                        model=LLM_config['model'],
                        messages=message,
                        temperature=LLM_config['temperature'],
                        max_tokens=LLM_config['max_tokens'],
                        frequency_penalty=LLM_config['frequency_penalty'],
                        presence_penalty=LLM_config['presence_penalty'],
                        n=LLM_config['n']
                    ) for message in message_batch]
                    async_responses = await asyncio.gather(*async_request)
                except Exception as e:
                    print(e)
                    print("Retrying....")
                    time.sleep(60)
            responses += async_responses
            print('Batch #{}: question {}-{}'.format(batch_num, batch_size*batch_num, min(batch_size*(batch_num+1)-1, len(messages))))
        return responses
    elif LLM_config['model'] == 'davinci-002':
        batch_size = LLM_config['batch_size']
        batched_messages = [texts[i: i + batch_size] for i in range(0, len(texts), batch_size)]
        responses = []
        for batch_num, message_batch in enumerate(batched_messages):
            async_responses = None
            while async_responses is None:
                try:
                    async_request = [openai.Completion.acreate(
                        engine=LLM_config['model'],
                        prompt=message,
                        temperature=LLM_config['temperature'],
                        max_tokens=LLM_config['max_tokens'],
                        frequency_penalty=LLM_config['frequency_penalty'],
                        presence_penalty=LLM_config['presence_penalty'],
                        stop=LLM_config['stop']
                    ) for message in message_batch]
                    async_responses = await asyncio.gather(*async_request)
                except Exception as e:
                    print(e)
                    print("Retrying....")
                    time.sleep(30)
            responses += async_responses
            print('Batch #{}: question {}-{}'.format(batch_num, batch_size * batch_num,
                                                     min(batch_size * (batch_num + 1) - 1, len(texts))))
        return responses


def async_query(LLM_config, data):
    loop = asyncio.get_event_loop()
    responses = loop.run_until_complete(async_get_response(LLM_config, data))
    assert LLM_config['model'] in ['gpt-3.5-turbo', 'davinci-002'], 'Undefined model'
    if LLM_config['model'] == 'gpt-3.5-turbo':
        # answer = [responses[i]['choices'][0]['message']['content'] for i in range(len(responses))]
        answer = []
        for i in range(len(responses)):
            response = []
            for choice in responses[i]['choices']:
                if choice['finish_reason'] == 'stop':
                    response.append(choice['message']['content'])
            answer.append(response)
    else:
        answer = [responses[i]['choices'][0]['text'] for i in range(len(responses))]
    return answer
