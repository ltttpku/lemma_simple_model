import numpy as np
import pickle

glove_pt_path = '/home/leiting/scratch/hcrn-videoqa/data/glove/glove.840.300d.pkl'

with open('data/vocab.txt',  'r') as vocabf:
    token_itow = vocabf.readlines()

    glove_matrix = None
        
    # token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}
    print("Load glove from %s" % glove_pt_path)
    glove = pickle.load(open(glove_pt_path, 'rb'))
    dim_word = glove['the'].shape[0]
    print('dim_word:', dim_word)
     
    glove_matrix = []
    for i in range(len(token_itow)):
        vector = glove.get(token_itow[i].strip(), np.zeros((dim_word,)))
        glove_matrix.append(vector)
    glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
    print(glove_matrix.shape)

    obj= {
           'glove': glove_matrix,
    }

    output_file = 'data/glove.pt'
    with open(output_file, 'wb') as outf:
        pickle.dump(obj, outf)