# lemma_simple_model

## TODO
+ dataset/dataset.py: visual feature preload! DONE: reduce training time from 6h to 30min :p
+ QA distribution statistics?


|        MODEL         | add reasoning_type_acc calculator | ALL_ACC |
|        ----           |         :----:                 | :----: |  
|  cnn_lstm                        |       1      |       >0.37       |
| visual_bert(from_pretrained)     |       1      |       0.757       |
|  pure_lstm                       |       0      |       0.37        |
| linguistic_bert(from_pretrained) |       1      |       0.68    |
|  hme                             |       0      |        -        |
|  hga                             |       0      |        -        |
|  psac                            |       1      |        -        |
| hcrn                             |     1       |       >0.58      |
| hcrn(w/o visual)                 |     1        |      >0.50     |


## Preprocess

### HCRN-Preprocess
+  原始qas.json放到 data/hcrn_data/ 下

1. assign each qa an id: data/hcrn_data/qas.json --> data/hcrn_data/**tagged_qas.json**

```bash
$ python hcrn_preprocess/qas2tagged_qas.py  # tagged_qa中，video_id 和 <video_name, interval>一一对一, 
# tag qas using /scratch/generalvision/LEMMA/vid_intervals.json
```


2. preprocess visual features (only need to run ONCE even if there's a new set of qas) :  --> data/hcrn_data/lemma-qa_{}_feat.h5

```bash
$ python preprocess/preprocess_features.py --gpu_id 0 --dataset lemma-qa --model resnet101

$ python preprocess/preprocess_features.py --dataset lemma-qa --model resnext101 --image_height 112 --image_width 112
```

3. preprocess vocab: --> data/hcrn_data/lemma-qa_vocab.json


```bash
$ python hcrn_preprocess/preprocess_vocab.py
```


------------------------------------------


### Other Preprocess
```bash

$ cp data/hcrn_data/lemma-qa_vocab.json  data/

$ cp data/hcrn_data/tagged_qas.json  data/
```

--------------------------------------
mode in ['train', 'test', 'val']


1. tagged_qas.json --> **{mode}_qas.json**, naive train test split

```bash
$ python hme_preprocess/split.py 
```

3. {mode}_qas.json， lemma-qa_vocab.json --> **{mode}_qas_encode.json**, **answer_set.txt, vocab.txt**

```bash
$ python hme_preprocess/mode_qas2mode_qas_encode.py
```

4. LEMMA videos --> **video_feature_20.h5** (ONLY need to run once, default: vgg+c3d), used by HME and HGA

```bash
$ python hme_preprocess/generate_feature20.py
```

5.  vocab.txt --> **glove.pt**, used for word embedding by all models


+ need to set 'glove_pt_path' in generate_glove_matrix.py


```bash
$ python preprocess/generate_glove_matrix.py
```

6. tagged_qas.json --> **char_vocab.txt**, used by PSAC

```bash
$ python preprocess/generate_char_vocab.py
```

7. {mode}_qas_encode.json --> **formatted_{mode}_qas_encode.json**, 

+ need to define max_word_len, max_sentence_len for psac;

+ need to define max_sentence_len for visual_bert and linguistic_bert(args.max_len);


```bash
$ python preprocess/format_mode_qas_encode.py {mode}
```


8. tagged_qas.json -->all_reasoning_types.txt
```bash
$ python preprocess/reasoning_types.py
```

## Train

```bash
$ python train_xxx.py
```

> tensorboard --logdir cnn_lstm_logs/events
> 
> logs: cnn_lstm_logs/events/{TIME}/log.txt


