#!/bin/bash
vocab_path='/data/c/mingfeicheng/system_regression/system_regression/asr/align/vocab'
dataset='heysquad_human'
debug='False'
batch_size=16
ensemble_type='logit'
weight_adj='False'

declare -a ensembles=("dropout" "pertub")
declare -a tests=(
    "s2t-small-librispeech-asr whisper-tiny old2new" # 1. s2t small -> whisper tiny
    "s2t-medium-librispeech-asr whisper-tiny old2new" # 2. s2t medium -> whisper tiny
    "s2t-large-librispeech-asr whisper-tiny old2new" # 3. s2t large -> whisper tiny
    "s2t-large-librispeech-asr whisper-tiny.en old2new" # 4. s2t large -> whisper tiny en
    "whisper-tiny whisper-tiny.en old2new" # 5. whisper tiny -> whisper tiny en
    "whisper-tiny.en whisper-tiny old2new" # 6. whisper tiny en -> whisper tiny
    "s2t-small-librispeech-asr whisper-tiny.en old2new" # 7. s2t small -> whisper tiny en
)

# Function to run the Python script
run_test() {
  old_model_path=$1
  new_model_path=$2
  align=$3
  ensemble=$4

  # Extract the model name from the path, assumes model name is the prefix before the first '-'
  old_model_name="${old_model_path%%-*}"
  new_model_name="${new_model_path%%-*}"

  CUDA_VISIBLE_DEVICES=3 python /data/c/mingfeicheng/system_regression/system_regression/asr/reg_reduce/regbug_reducing_align.py \
    --old_model "$old_model_name" \
    --new_model "$new_model_name" \
    --oldmodel_path "$old_model_path" \
    --newmodel_path "$new_model_path" \
    --dataset "$dataset" \
    --debug "$debug" \
    --batch_size "$batch_size" \
    --align "$align" \
    --vocab_path "$vocab_path" \
    --ensemble_type "$ensemble_type" \
    --ensemble "$ensemble" \
    --weight_adj "$weight_adj"
}

# Iterate over ensemble types and test configurations
for ensemble in "${ensembles[@]}"; do
  for test in "${tests[@]}"; do
    IFS=' ' read -r old_model_path new_model_path align <<< "$test"
    run_test "$old_model_path" "$new_model_path" "$align" "$ensemble"
  done
done
