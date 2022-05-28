import nltk
import json

modes = ['train' ,'test', 'val']
for mode in modes:
    input_file = f'data/{mode}_qas.json'
    # # output: {mode}_qas_encode.json, answer_set.txt, vocab.txt
    output_file = f'data/{mode}_qas_encode.json'

    with open('data/lemma-qa_vocab.json', 'r') as lemma_vocab_f:
        input_vocab = json.load(lemma_vocab_f)
        answer_set_lst = ['<UNK>']
        # # generate answer_set
        for ans in input_vocab['answer_token_to_idx']:
            if ans in ['<UNK>' , '<UNK0>', '<UNK1>', '<NULL>']:
                continue
            if ans not in answer_set_lst:
                answer_set_lst.append(ans)
        with open('data/answer_set.txt', 'w') as answerset_f:
            for ans in answer_set_lst:
                answerset_f.write(ans)
                answerset_f.write('\n')
        
        # # generate vocab_set
        vocab_lst = ['<UNK>' ,'<CLS>', '<SEP>']
        for word in input_vocab['question_token_to_idx']:
            if word in ['<UNK>' , '<UNK0>', '<UNK1>', '<NULL>']:
                continue
            if word not in vocab_lst:
                vocab_lst.append(word)
        with open('data/vocab.txt', 'w') as vocab_f:
            for word in vocab_lst:
                vocab_f.write(word)
                vocab_f.write('\n')

        with open(input_file, 'r') as f:
            qas = json.load(f)
            for qa in qas:
                # question_word_lst = qa['question'][:-1].split(' ') # #去掉标点符号
                encoded_q = []
                question = qa['question'].lower()[:-1]
                question_word_lst = nltk.word_tokenize(question)
                for word in question_word_lst:
                    word = word.lower()
                    if word not in vocab_lst:
                        import pdb; pdb.set_trace()
                    assert word in vocab_lst
                    encoded_q.append(str(vocab_lst.index(word)))
                encoded_q = ' '.join(encoded_q)
                qa['question_encode'] = encoded_q
                if mode == 'train':
                    qa['answer_encode'] = str(answer_set_lst.index(qa['answer'].lower()))
            with open(output_file, 'w') as outf:
                json.dump(qas, outf)
            

