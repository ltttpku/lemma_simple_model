# lemma_simple_model

### Preprocess

```bash
$ cd .. 

$ cp hcrn-videoqa/data/lemma-qa/lemma-qa_vocab.json  lemma_simple_model/data

$ cp hcrn-videoqa/data/lemma-qa/tagged_qas.json  lemma_simple_model/data
```

--------------------------------------
mode in ['train', 'test', 'val']
1. change dir

```bash
$ cd lemma_simple_model
$ mkdir data
```

2. tagged_qas.json --> **{mode}_qas.json**, naive train test split

```bash
$ python hme_preprocess/split.py 
```

3. {mode}_qas.json， lemma-qa_vocab.json --> **{mode}_qas_encode.json**, **answer_set.txt, vocab.txt**

```bash
$ python hme_preprocess/mode_qas2mode_qas_encode.py
```

4. LEMMA videos --> **video_feature_20.h5** ( default: vgg+c3d), used by HME and HGA

```bash
$ python hme_preprocess/generate_feature20.py
```

5.  vocab.txt --> **glove.pt**, used for word embedding by all models


need to set 'glove_pt_path' in generate_glove_matrix.py

NOTE: CLS and SEP token are initialized as np.zeros((dim_word,)) 

```bash
$ python preprocess/generate_glove_matrix.py
```

6. tagged_qas.json --> **char_vocab.txt**, used by PSAC

```bash
$ python preprocess/generate_char_vocab.py
```

7. {mode}_qas_encode.json --> **formatted_{mode}_qas_encode.json**, 

```bash
$ python preprocess/format_mode_qas_encode.py {mode}
```

8. tagged_qas.json -->all_reasoning_types.txt
```bash
$ python preprocess reasoning_types.py
```

### Train

```bash
$ python train_xxx.py
```


### TODO
+ dataset/dataset.py: visual feature preload!
+ QA distribution statistics?


|  MODEL | reasoning_type_acc calculator | ALL_ACC |
|  ----  |         :----:                | :----: |  
|  cnn_lstm   |       1       |          -          |
| visual_bert |       0       |          -          |
|  pure_lstm  |       0       |          -          |
| hme         |       0       |          -          |
|  hga        |       0       |          -          |
| psac        |       0       |          -          |