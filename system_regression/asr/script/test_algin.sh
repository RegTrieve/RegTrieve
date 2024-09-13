#!/bin/bash

vocab_path='/data/c/mingfeicheng/system_regression/system_regression/asr/align/vocab'
dataset='heysquad_human'
debug='False'
batch_size=16

# Define a function to run the model testing command
run_model_test() {
    old_model_name=$1
    old_model_path=$2
    new_model_name=$3
    new_model_path=$4
    align=$5

    CUDA_VISIBLE_DEVICES=3 python test_model_align.py \
    --old_model "$old_model_name" \
    --new_model "$new_model_name" \
    --oldmodel_path "$old_model_path" \
    --newmodel_path "$new_model_path" \
    --dataset "$dataset" \
    --debug "$debug" \
    --batch_size "$batch_size" \
    --align "$align" \
    --vocab_path "$vocab_path"
}

# Array of test configurations
declare -a tests=(
    "s2t s2t-small-librispeech-asr whisper whisper-tiny old2new" # 1. s2t small -> whisper tiny
    "s2t s2t-medium-librispeech-asr whisper whisper-tiny old2new" # 2. s2t medium -> whisper tiny
    "s2t s2t-large-librispeech-asr whisper whisper-tiny old2new" # 3. s2t large -> whisper tiny
    "s2t s2t-large-librispeech-asr whisper whisper-tiny.en old2new" # 4. s2t large -> whisper tiny en
    "whisper whisper-tiny whisper whisper-tiny.en old2new" # 5. whisper tiny -> whisper tiny en
    "whisper whisper-tiny.en whisper whisper-tiny old2new" # 6. whisper tiny en -> whisper tiny
    "s2t s2t-small-librispeech-asr whisper whisper-tiny.en old2new" # 7. s2t small -> whisper tiny en
)

# Iterate over the test configurations and run tests
for test in "${tests[@]}"; do
    IFS=' ' read -r old_model_name old_model_path new_model_name new_model_path align <<< "$test"
    run_model_test "$old_model_name" "$old_model_path" "$new_model_name" "$new_model_path" "$align"
done