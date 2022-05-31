import json, os, sys
import nltk

assert os.path.exists('data/char_vocab.txt'), 'run generate_char_vocab.py first'

max_word_len = 17 # # train_psac.py word_len: args.char_max_len
max_sentence_len = 50 # # train_psac.py sentence_len: args.max_len
                    # # train_visual_bert.py sentence_len args.max_len
                    # # train_linguistic_bert.py 
# mode = 'test'
mode = sys.argv[1]

# # input: {mode}_qas_encode.json
# # output: formatted_{mode}_qas_encode.json

with open('data/char_vocab.txt', 'r') as charf:
    char_lst = charf.readlines()
    char_lst = [char.strip() for char in char_lst]

    with open('/home/leiting/scratch/lemma_simple_model/data/answer_set.txt', 'r') as ans_f:
        answer_set = ans_f.readlines()
        answer_set = [ans.strip() for ans in answer_set]

        with open(f'/home/leiting/scratch/lemma_simple_model/data/{mode}_qas_encode.json', 'r') as f:
            tagged_qas = json.load(f)
            for qa in tagged_qas:
                question_encode = []
                for word in qa['question_encode'].split():
                    question_encode.append(int(word))
                qa['question_encode'] = question_encode

                if mode == 'train':
                    answer_encode = int(qa['answer_encode'])
                    qa['answer_encode'] = answer_encode
                else:
                    if qa['answer'] in answer_set:
                        qa['answer_encode'] = answer_set.index(qa['answer'])
                    else:
                        qa['answer_encode'] = answer_set.index('<UNK>')
                        
                char_tokens = []
                # sentence = qa['question'].lower().replace(',', '').replace('?', '').replace('\'s', ' \'s')
                # words = sentence.split()
                question = qa['question'].lower()[:-1]
                words = nltk.word_tokenize(question)
                for w in words:
                    c_t = []
                    for c in list(w):
                        c_t.append(char_lst.index(c))
                    for i in range(max_word_len - len(list(w))):
                        c_t.append(char_lst.index('|'))
                        
                    char_tokens.append(c_t)
                for z in range(max_sentence_len - len(words)):
                    char_tokens.append([char_lst.index('|')] * max_word_len)
                
                qa['question_char_encode'] = char_tokens
        
            with open(f'/home/leiting/scratch/lemma_simple_model/data/formatted_{mode}_qas_encode.json', 'w') as outf:
                json.dump(tagged_qas, outf)
                