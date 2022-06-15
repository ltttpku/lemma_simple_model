# lemma_simple_model


## PREPROCESS
```bash
$ cd lemma_simple_model
$ mkdir data
```
+ download & put video_feature_20.h5 to data/ , 
+ download & put lemma-qa_appearance_feat.h5  lemma-qa_motion_feat.h5 to data/hcrn_data/
+ download & put video_features to $FEATURE_BASE_PATH

+ put train_qas.json, test_qas.json, val_qas.json, tagged_qa.json, vid_intervals.json to data/


+ run PREPROCESS.sh (it'll run the following scripts for you)
  + python preprocess/preprocess_vocab.py
    + train_qas.json --> lemma-qa_vocab.json
    + may need to add "has", "that" to lemma-qa_vocab.json

  + python preprocess/mode_qas2mode_qas_encode.py
    + {mode}_qas.json， lemma-qa_vocab.json --> {mode}_qas_encode.json, answer_set.txt, vocab.txt
    
  + python preprocess/generate_glove_matrix.py
    + need to set glove path
    + vocab.txt --> glove.pt
    
  + python preprocess/generate_char_vocab.py
    + tagged_qas.json --> char_vocab.txt
    
  + python preprocess/format_mode_qas_encode.py {mode}
    + need to define max_word_len for psac (args.char_max_len)
    + need to define max_sentence_len for psac, visual_bert and linguistic_bert(args.max_len)
    
  + python preprocess/reasoning_types.py
    + tagged_qas.json -->all_reasoning_types.txt

## TRAIN

```bash
$ python $TRAIN_MODEL_PY
```
$TRAIN_MODEL_PY in train_hcrn.py, train_hme.py, train_hga.py

<br /> 

```bash
$ python $TRAIN_MODEL_PY --feature_base_path $FEATURE_BASE_PATH
```

$TRAIN\_MODEL\_PY in {train_psac.py, train_pure_lstm.py, train_linguistic_bert.py, train_visual_bert.py, }

$FEATURE_BASE_PATH is the path to preprocessed resnet features.

For bert-based model, you need to set BertTokenizer_CKPT and BertModel_CKPT for the model.