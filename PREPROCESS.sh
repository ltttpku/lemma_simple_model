
echo '>>> running preprocess/preprocess_vocab.py'
python preprocess/preprocess_vocab.py

echo '>>> running preprocess/mode_qas2mode_qas_encode.py'
python preprocess/mode_qas2mode_qas_encode.py

echo '>>> running preprocess/generate_glove_matrix.py'
python preprocess/generate_glove_matrix.py

echo '>>> running preprocess/generate_char_vocab.py'
python preprocess/generate_char_vocab.py

echo '>>> running preprocess/format_mode_qas_encode.py $mode'
echo '>>> it may take 1~2 min'
for mode in train test val
do
    python preprocess/format_mode_qas_encode.py $mode
done

python preprocess/reasoning_types.py
echo '>>> DONE!'