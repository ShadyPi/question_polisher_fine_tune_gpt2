from string import Template


instruction = 'Please simplify the following question to make it easier to solve.\n\n'


def augment_GSM8K(question):
    query_template = 'Question: ${question}'
    query_template = Template(query_template)
    fill = {'question': question}
    query = query_template.substitute(fill)

    return instruction+query


def score_GSM8K(demo, question):
    # demo = 'Question: James decides to bulk up.  He weighs 120 kg and gains 20% of his body weight in muscle and 1 ' \
    #        'quarter that much in fat.  How much does he weigh now?\nAnswer: 150\n\n'
    query_template = 'Question: ${question}\nAnswer: '
    query_template = Template(query_template)
    fill = {'question': question}
    query = query_template.substitute(fill)

    return demo+query
