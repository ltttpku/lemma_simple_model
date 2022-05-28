import json

with open('data/tagged_qas.json', 'r') as f:
    tagged_qas = json.load(f)
    train_qas = []
    test_qas = []
    val_qas = []
    for i, qa in  enumerate(tagged_qas):
        if i % 10 == 1:
            test_qas.append(qa)
        elif i % 10 == 2:
            val_qas.append(qa)
        else:
            train_qas.append(qa)
    with open('data/train_qas.json', 'w') as train_f:
        json.dump(train_qas, train_f)
    with open('data/test_qas.json', 'w') as test_f:
        json.dump(test_qas, test_f)
    with open('data/val_qas.json', 'w') as val_f:
        json.dump(val_qas, val_f)