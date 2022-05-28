# lemma_simple_model

>cd CODE 
   cp hcrn-videoqa/data/lemma-qa/lemma-qa_vocab.json  lemma_simple_model/data
   cp hcrn-videoqa/data/lemma-qa/tagged_qas.json  lemma_simple_model/data


mode in ['train', 'test', 'val']
1. change dir
> cd lemma_simple_model

1. tagged_qas.json --> **{mode}_qas.json**, naive train test split
> python hme_preprocess/split.py 

2. {mode}_qas.json， lemma-qa_vocab.json --> **{mode}_qas_encode.json**, **answer_set.txt, vocab.txt**
> python hme_preprocess/mode_qas2mode_qas_encode.py

3. LEMMA videos --> **video_feature_20.h5** ( default: vgg+c3d), used by HME and HGA
> python hme_preprocess/generate_feature20.py

4.  vocab.txt --> **glove.pt**, used for word embedding by all models
need to set 'glove_pt_path' in generate_glove_matrix.py
NOTE: CLS and SEP token are initialized as np.zeros((dim_word,))
> python preprocess/generate_glove_matrix.py

5. tagged_qas.json --> **char_vocab.txt**, used by PSAC
> python preprocess/generate_char_vocab.py

6. {mode}_qas_encode.json --> **formatted_{mode}_qas_encode.json**, 
''' python preprocess/format_mode_qas_encode.py {mode} '''





