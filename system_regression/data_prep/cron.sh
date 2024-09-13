#!/bin/bash
# * * * * * /path_to/system_regression/data_prep/cron.sh
if pgrep -f create_knn_qa.py > /dev/null
then
    echo "Process create_knn_qa.py is running."
else
    echo "No process create_knn_qa.py found. Running the command."
    export CUDA_VISIBLE_DEVICES='3' 
    cd /path_to/system_regression/data_prep
    nohup /home/ubuntu/.conda/envs/fuse_attack/bin/python create_knn_qa.py --model_type roberta --model_name_or_path /path_to/system_regression/qa/save/output-roberta-large-train_origin-dev_origin --do_train --do_eval --do_lower_case --predict_file human_transcribed/s2t-large-librispeech-asr.json --train_file /path_to/system_regression/data/HeySQuAD_json/train-common-original-48849.json --per_gpu_eval_batch_size=8 --max_seq_length 512 --doc_stride 128 --eval_single True --overwrite_cache --metric_type ig > ig.log 2>&1 &
fi