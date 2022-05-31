import random
import json

base_dir = 'data/'
with open(base_dir + 'tagged_qas.json', 'r') as f:
    qas = json.load(f)
    train_qa_lst = []
    test_qa_lst = []
    val_qa_lst = []

    type_qa_dct = {}
    for qa in qas:
        if qa['reasoning_type'] not in type_qa_dct:
            type_qa_dct[qa['reasoning_type']] = [qa]
        else:
            type_qa_dct[qa['reasoning_type']].append(qa)
    
    for key, value in type_qa_dct.items():
        random.shuffle(value)
        train_qa_lst += (value[:int(len(value) * (4 / 6))])
        test_qa_lst += (value[int(len(value) * (4 / 6)):int(len(value) * (5 / 6))])
        val_qa_lst += (value[int(len(value) * (5 / 6)):])
    
    with open(base_dir + 'train_qas.json', 'w') as f:
        json.dump(train_qa_lst, f, indent=4)
    with open(base_dir + 'test_qas.json', 'w') as f:
        json.dump(test_qa_lst, f, indent=4)
    with open(base_dir + 'val_qas.json', 'w') as f:
        json.dump(val_qa_lst, f, indent=4)
