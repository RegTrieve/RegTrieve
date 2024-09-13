export CUDA_VISIBLE_DEVICES='2';nohup python create_knn_asr.py --dataset heysquad_human > knn_asr.log 2>&1 &


export CUDA_VISIBLE_DEVICES='3';nohup python create_knn_qa.py --model_type roberta --model_name_or_path /path_to/system_regression/qa/save/output-roberta-large-train_origin-dev_origin  --do_train --do_lower_case  --train_file s2t-large-librispeech-asr.json --per_gpu_train_batch_size=64 --max_seq_length 512 --doc_stride 128 --eval_single True --metric_type loss --mode token --threads 4 --overwrite_cache > s2t_large_loss_2.log 2>&1 &

 export CUDA_VISIBLE_DEVICES='0';nohup python create_knn_qa.py --model_type roberta --model_name_or_path /path_to/system_regression/qa/save/output-roberta-large-train_origin-dev_origin  --do_train --do_lower_case  --train_file s2t-medium-librispeech-asr.json --per_gpu_train_batch_size=64 --max_seq_length 512 --doc_stride 128 --eval_single True --metric_type loss --mode token --threads 4 --overwrite_cache > s2t_medium_loss_2.log 2>&1 &
 
 export CUDA_VISIBLE_DEVICES='1';nohup python create_knn_qa.py --model_type roberta --model_name_or_path /path_to/system_regression/qa/save/output-roberta-large-train_origin-dev_origin  --do_train --do_lower_case  --train_file s2t-small-librispeech-asr.json --per_gpu_train_batch_size=64 --max_seq_length 512 --doc_stride 128 --eval_single True --metric_type loss --mode token --threads 4 --overwrite_cache > s2t_small_loss_2.log 2>&1 &

 export CUDA_VISIBLE_DEVICES='0';nohup python create_knn_qa.py --model_type roberta --model_name_or_path /path_to/system_regression/qa/save/output-roberta-large-train_origin-dev_origin --do_train --do_lower_case  --train_file /path_to/system_regression/predictions/asr/human_transcribed/train_question/${MODEL}.json --per_gpu_eval_batch_size=8 --max_seq_length 512 --doc_stride 128 --eval_single True --overwrite_cache --metric_type loss --mode question --threads 16 > knn_${MODEL}_question.log &




export CUDA_VISIBLE_DEVICES='1'; python create_knn_qa.py --model_type roberta --model_name_or_path /path_to/system_regression/qa/save/output-roberta-large-train_origin-dev_origin --do_train --do_eval --do_lower_case --predict_file human_transcribed/s2t-large-librispeech-asr.json --train_file /path_to/system_regression/data/HeySQuAD_json/train-common-original-48849.json --per_gpu_eval_batch_size=8 --max_seq_length 512 --doc_stride 128 --eval_single True --overwrite_cache --metric_type ig > ig.log 2>&1 &


