# lemma_simple_model

## Environment Setup
We provide all environment configurations in ``environment.txt``. To install all packages, you can create a conda environment and install the packages as follows: 
```bash
conda create -n lemma python=3.8
conda activate lemma
pip install -r environment.txt
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113 
```
In our experiments, we used NVIDIA CUDA 11.3 on Ubuntu 20.04. Similar CUDA version should also be acceptable with corresponding version control for ``torch`` and ``torchvision``.

## Data Download
For data, features and data download, please refer to our [website](https://sites.google.com/view/egotaskqa). Within the google drive, you can find features and model checkpoints under ``features/`` and ``checkpoints/`` respectively.

After download, create ``data/`` under the current directory by
```bash
$ cd lemma_simple_model
$ mkdir data
$ mkdir data/hcrn_data
```
Next, put the features, data and checkpoitns to subdirectories as follows:
+ download and put ``features/video_feature_20.h5`` to ``data/``.
+ download and put ``features/lemma-qa_appearance_feat.h5``, ``features/lemma-qa_motion_feat.h5`` to ``data/hcrn_data/``.
+ download ``features/video_features.zip`` and unzip it to ``$FEATURE_BASE_PATH``.
+ download ``features/glove.840.300d.pkl`` to ``$GLOVE_PT_PATH`` and set ``glove_pt_path`` to ``$GLOVE_PT_PATH`` in ``preprocess/generate_glove_matrix.py``.
+ download and put ``data/train_qas.json``, ``data/test_qas.json``, ``data/val_qas.json``, ``data/tagged_qa.json``, ``data/vid_intervals.json`` to ``$BASE_DATA_DIR``

## Preprocessing
After downloading all data to their correct locations, run the following for preprocessing:
```bash
$ chmod a+x PREPROCESS.sh
$ ./PREPROCESS.sh $BASE_DATA_DIR
```
This script will run the following preprocess for features and texts:
  - ```bash
    $ python preprocess/preprocess_vocab.py
    ```
    This will generate ``lemma-qa_vocab.json``.
  - ```bash
    $ python preprocess/mode_qas2mode_qas_encode.py
    ```
    This will convert {mode}_qas.json，lemma-qa_vocab.json to {mode}_qas_encode.json, answer_set.txt, vocab.txt.
  - ```bash
    $ python preprocess/generate_glove_matrix.py
    ```
    Before running ``PREPROCESS.sh``, please make sure that the ``glove_pt_path`` is correctly set. This script will generate ``glove.pt``.
  - ```bash
    $ python preprocess/generate_char_vocab.py
    ```
    This script will generate ``char_vocab.txt``.
    
  - ```bash
    $ python preprocess/format_mode_qas_encode.py {mode}
    ```
    Before running the experiments, please make sure that ``max_word_len`` in ``preprocess/format_mode_qas_encode.py`` is equal to ``args.char_max_len`` defined in ``train_psac.py``. Similary, make sure that ``max_sentence_len`` in ``preprocess/format_mode_qas_encode.py`` is equal to ``args.max_len`` in ``train_psac.py``, ``train_linguistic_bert.py`` and ``train_visual_bert.py``.
    
  - ```bash
    $ python preprocess/reasoning_types.py
    ```
    THis will generate ``all_reasoning_types.txt``.


## Training

To train the model from scratch we provide the following model files:
 - ``train_hcrn.py``: HCRN experiment
 - ``train_hga.py``: HGA experiment
 - ``train_hme.py``: HME experiment
 - ``train_linguistic_bert.py``: BERT experiment
 - ``train_psac.py``: PSAC experiment
 - ``train_pure_lstm.py``: LSTM experiment (addtional LSTM and CNN-LSTM experiment)
 - ``train_visual_bert.py``: VisualBERT experiment

Use the following command and substitute ``$TRAIN_MODEL_PY`` to the model you want to experiment with:
```bash
$ python $TRAIN_MODEL_PY --base_data_dir $BASE_DATA_DIR
```
for models ``$TRAIN_MODEL_PY`` in ``train_hcrn.py``, ``train_hme.py``, ``train_hga.py`` (you might also want to change the ``app_feat_path``, ``motion_feat_path`` and ``video_feat_path`` in these files for adjusting the feature path) and 

```bash
$ python $TRAIN_MODEL_PY --feature_base_path $FEATURE_BASE_PATH --base_data_dir $BASE_DATA_DIR
```
for models ``$TRAIN_MODEL_PY`` in ``train_psac.py``, ``train_pure_lstm.py``, ``train_linguistic_bert.py``, ``train_visual_bert.py``.


For bert-based model, you need to set BertTokenizer_CKPT and BertModel_CKPT for the model to load pretrained model from huggingface.
+ For linguistic_bert, set BertTokenizer_CKPT="bert-base-uncased", BertModel_CKPT="bert-base-uncased".

+ For visual_bert, set BertTokenizer_CKPT="bert-base-uncased", VisualBertModel_CKPT="uclanlp/visualbert-vqa-coco-pre".

## Reload ckpts & test_only
To reload checkpoints and only run inference on test_qas, run the following command:

```bash
$ python $TRAIN_MODEL_PY --base_data_dir $BASE_DATA_DIR --reload_model_path $RELOAD_MODEL_PATH --test_only 1
```
for models ``$TRAIN_MODEL_PY`` in ``train_hcrn.py``, ``train_hme.py``, ``train_hga.py`` and 

```bash
$ python $TRAIN_MODEL_PY --feature_base_path $FEATURE_BASE_PATH --base_data_dir $BASE_DATA_DIR --reload_model_path $RELOAD_MODEL_PATH --test_only 1
```
for models ``$TRAIN_MODEL_PY`` in ``train_psac.py``, ``train_pure_lstm.py``, ``train_linguistic_bert.py``, ``train_visual_bert.py``.


## Acknowledgement
This code heavily used resources from [VisualBERT](https://huggingface.co/docs/transformers/v4.19.2/en/model_doc/visual_bert#visualbert), [HCRN](https://github.com/thaolmk54/hcrn-videoqa), [HGA](https://github.com/Jumpin2/HGA), [HME](https://github.com/fanchenyou/HME-VideoQA), [PSAC](https://github.com/lixiangpengcs/PSAC). We thank the authors for open-sourcing their awesome projects.
