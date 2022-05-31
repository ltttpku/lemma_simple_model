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
import glob, os


base_dir = '/home/leiting/scratch/hcrn-videoqa/data/lemma-qa/'

def load_video_paths(args):
    with open(base_dir + 'qas.json') as f:
        qas = json.load(f)
        interval2video_id = {}
        video_id = 0
        question_id = 0
        for qa in qas:
            if qa['interval'] not in interval2video_id.keys():
                interval2video_id[qa['interval']] = video_id
                qa['video_id'] = video_id
                video_id += 1
            else:
                qa['video_id'] = interval2video_id[qa['interval']]

            qa['question_id'] = question_id
            question_id += 1
        
        with open(base_dir + 'tagged_qas.json', 'w') as f:
            json.dump(qas, f)
    print('total num of qas:', len(qas))
    unique_qas = []
    existing_video_ids = []
    for qa in qas:
        if qa['video_id'] in existing_video_ids:
            continue
        else:
            existing_video_ids.append(qa['video_id'])
            unique_qas.append(qa)
    return unique_qas


def load_frames_from_interval(interval):
    video_name, fpv, start, end = interval.split('|') # # [start, end)
    start, end = int(start) + 1 , int(end) + 1 # # annotation下标从0开始，但frame img下标从1开始
    start, end = str(start), str(end)
    start, end = start.rjust(5, '0'), end.rjust(5, '0')
    video_path = os.path.join('/scratch/generalvision/LEMMA/videos', video_name, f'fpv{fpv[-1]}', 'img_*.jpg')
    all_video_frames = sorted(glob.glob(video_path))
    start_frame = os.path.join('/scratch/generalvision/LEMMA/videos', video_name, f'fpv{fpv[-1]}', f'img_{start}.jpg')
    end_frame = os.path.join('/scratch/generalvision/LEMMA/videos', video_name, f'fpv{fpv[-1]}', f'img_{end}.jpg')
    start_idx = all_video_frames.index(start_frame)
    end_idx = all_video_frames.index(end_frame)
    return all_video_frames[start_idx:end_idx]


def process_questions(args):
    ''' Encode question tokens'''
    print('Loading data')
    with open(args.annotation_file, 'r') as dataset_file:
        instances = json.load(dataset_file)

    # Either create the vocab or load it from disk
    if args.mode in ['train']:
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
    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.dataset, args.dataset), 'r') as f:
            vocab = json.load(f)

    # Encode all questions
    print('Encoding data')
    questions_encoded = []
    questions_len = []
    question_ids = []
    video_ids_tbw = []
    video_names_tbw = []
    all_answers = []
    for idx, instance in enumerate(instances):
        question = instance['question'].lower()[:-1]
        question_tokens = nltk.word_tokenize(question)
        question_encoded = utils.encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        question_ids.append(idx)
        im_name = instance['video_id']
        video_ids_tbw.append(im_name)
        video_names_tbw.append(im_name)

        if instance['answer'] in vocab['answer_token_to_idx']:
            answer = vocab['answer_token_to_idx'][instance['answer']]
        elif args.mode in ['train']:
            answer = 0
        elif args.mode in ['val', 'test']:
            answer = 1

        all_answers.append(answer)
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])
    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    glove_matrix = None
    if args.mode == 'train':
        token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}
        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print(glove_matrix.shape)

    print('Writing', args.output_pt.format(args.dataset, args.dataset, args.mode))
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'question_id': question_ids,
        'video_ids': np.asarray(video_ids_tbw),
        'video_names': np.array(video_names_tbw),
        'answers': all_answers,
        'glove': glove_matrix,
    }
    with open(args.output_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
        pickle.dump(obj, f)

 

if __name__ == '__main__':
    # video_paths = load_video_paths(None)
    # check_video()
    print('in main')