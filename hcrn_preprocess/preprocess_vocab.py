from distutils.util import strtobool
from enum import unique
from glob import glob
import json
# from datautils import utils
import nltk
from collections import Counter
from datautils import utils

import pickle
import numpy as np
import glob, os, argparse

def preprocess_vocab(args):
    print('Loading data')
    with open(args.annotation_file, 'r') as dataset_file:
        instances = json.load(dataset_file)
    print('Building vocab')
    answer_cnt = {}
    for instance in instances:
        answer = instance['answer']
        answer_cnt[answer] = answer_cnt.get(answer, 0) + 1

    answer_token_to_idx = {'<UNK0>': 0, '<UNK1>': 1}
    answer_counter = Counter(answer_cnt)
    frequent_answers = answer_counter.most_common(args.answer_top)
    total_ans = sum(item[1] for item in answer_counter.items())
    total_freq_ans = sum(item[1] for item in frequent_answers)
    print("Number of unique answers:", len(answer_counter))
    print("Total number of answers:", total_ans)
    print("Top %i answers account for %f%%" % (len(frequent_answers), total_freq_ans * 100.0 / total_ans))

    for token, cnt in Counter(answer_cnt).most_common(args.answer_top):
        answer_token_to_idx[token] = len(answer_token_to_idx)
    print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

    question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
    for i, instance in enumerate(instances):
        question = instance['question'].lower()[:-1]
        for token in nltk.word_tokenize(question):
            if token not in question_token_to_idx:
                question_token_to_idx[token] = len(question_token_to_idx)
    print('Get question_token_to_idx')
    print(len(question_token_to_idx))

    vocab = {
        'question_token_to_idx': question_token_to_idx,
        'answer_token_to_idx': answer_token_to_idx,
        'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
    }

    print('Write into %s' % args.vocab_json.format(args.dataset, args.dataset))
    with open(args.vocab_json.format(args.dataset, args.dataset), 'w') as f:
        json.dump(vocab, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='lemma-qa', choices=['tgif-qa', 'msrvtt-qa', 'lemma-qa'], type=str)
    parser.add_argument('--answer_top', default=4000, type=int, help='answer set size')
    # parser.add_argument('--glove_pt',
    #                     help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')
    # parser.add_argument('--output_pt', type=str, default='data/{}/{}_{}_questions.pt')
    parser.add_argument('--vocab_json', type=str, default='data/hcrn_data/{}_vocab.json')
    parser.add_argument('--mode', choices=['train', 'val', 'test'])
    # parser.add_argument('--question_type', choices=['frameqa', 'action', 'transition', 'count', 'none'], default='none')
    parser.add_argument('--seed', type=int, default=666)

    args = parser.parse_args()
    np.random.seed(args.seed)

    args.annotation_file = 'data/hcrn_data/tagged_qas.json'

    preprocess_vocab(args=args)