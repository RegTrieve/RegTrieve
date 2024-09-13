#!/bin/sh

base_model_lst="whisper-tiny.en"
blending_model_lst="whisper-tiny"

for base_model_name in $base_model_lst; do
  for blending_model_name in $blending_model_lst; do
    CUDA_VISIBLE_DEVICES=1 python /data/c/mingfeicheng/system_regression/system_regression/asr/align/align_vocab.py \
      --base_model_name_or_path "/data/c/mingfeicheng/system_regression/pretrained_models/${base_model_name}" \
      --blending_model_name_or_path "/data/c/mingfeicheng/system_regression/pretrained_models/${blending_model_name}" \
      --dataset_dir "None" \
      --vocab_mapping_save_dir "/data/c/mingfeicheng/system_regression/system_regression/asr/align/vocab/${blending_model_name}_to_${base_model_name}" \
      --cache_dir "/data/c/mingfeicheng/system_regression/outputs/cache_dir" \
      --model_max_length 2048 \
      --vocab_mapping_type "default" \
      --num_process 10
  done
done

for base_model_name in $blending_model_lst; do
  for blending_model_name in $base_model_lst; do
    CUDA_VISIBLE_DEVICES=1 python /data/c/mingfeicheng/system_regression/system_regression/asr/align/align_vocab.py \
      --base_model_name_or_path "/data/c/mingfeicheng/system_regression/pretrained_models/${base_model_name}" \
      --blending_model_name_or_path "/data/c/mingfeicheng/system_regression/pretrained_models/${blending_model_name}" \
      --dataset_dir "None" \
      --vocab_mapping_save_dir "/data/c/mingfeicheng/system_regression/system_regression/asr/align/vocab/${blending_model_name}_to_${base_model_name}" \
      --cache_dir "/data/c/mingfeicheng/system_regression/outputs/cache_dir" \
      --model_max_length 2048 \
      --vocab_mapping_type "default" \
      --num_process 10
  done
done