from string import Template


rewrite_instruction = 'Please rewrite the following question to make it easier to solve.\n\n'


def augment_GSM8K(question):
    query_template = 'Question: ${question}'
    query_template = Template(query_template)
    fill = {'question': question}
    query = query_template.substitute(fill)

    return rewrite_instruction+query


def score_GSM8K(demo, question):
    # demo = 'Question: James decides to bulk up.  He weighs 120 kg and gains 20% of his body weight in muscle and 1 ' \
    #        'quarter that much in fat.  How much does he weigh now?\nAnswer: 150\n\n'
    query_template = 'Q: ${question}\nA: '
    query_template = Template(query_template)
    fill = {'question': question}
    query = query_template.substitute(fill)

    return demo+query


def polish_query(demo_base, demo_polished, text):
    # instruction = 'The following exemplars show how to rewrite questions to make them easier:\n\n${demos}' \
    #               'Now please rewrite the following question. Don\'t omit any useful information, especially the numbers.\n\n' \
    #               '[Original]\n${base_text}\n' \
    #               '[New]\n'
    instruction = 'Now please rewrite the following question. Don\'t omit any useful information, especially the numbers.\n\n' \
                  '[Original]\n${base_text}\n' \
                  '[New]\n'
    demo_temp = '[Original]\n${base_text}\n' \
                '[New]\n${polished_text}\n\n'
    instruction = Template(instruction)
    demo_temp = Template(demo_temp)
    demo_text = ''
    for i in range(len(demo_base)):
        demo_item = demo_temp.substitute({'base_text': demo_base[i], 'polished_text': demo_polished[i]})
        demo_text += demo_item
    # fill = {'demos': demo_text, 'base_text': text}
    fill = {'base_text': text}
    query = instruction.substitute(fill)
    return query
