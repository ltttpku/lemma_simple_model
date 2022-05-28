import json

from numpy import real_if_close

with open('data/tagged_qas.json', 'r') as f:
    qas = json.load(f)
    reasoning_type_lst = []
    for qa in qas:
        for reasoning_type in qa['reasoning_type'].split('$'):
            if reasoning_type in reasoning_type_lst:
                continue
            else:
                reasoning_type_lst.append(reasoning_type)

    with open('data/all_reasoning_types.txt', 'w') as outf:
        for reasontype in reasoning_type_lst:
            outf.write(reasontype)
            outf.write('\n')

