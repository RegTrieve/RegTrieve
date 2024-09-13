nohup python run_squad.py \
  --model_type roberta \
  --model_name_or_path 'roberta-large' \
  --do_eval \
  --do_train \
  --do_lower_case \
  --train_file 'train-common-original-48849.json' \
  --predict_file 'dev-common-original-1002.json' \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --save_steps 10000 \
  --output_dir './save/roberta-large-train_origin-dev_origin' \
  --overwrite_output_dir --overwrite_cache > train_origin-dev-origin.log &


  # python /transformers/examples/legacy/question-answering/run_squad.py \
  # --model_type roberta \
  # --model_name_or_path roberta-large \
  # --do_eval \
  # --do_train \
  # --do_lower_case \
  # --train_file /HeySQuAD_train/train-common-human-transcribed-48849.json \
  # --predict_file /HeySQuAD_test/dev-common-human-transcribed-1002.json \
  # --per_gpu_train_batch_size=4 \
  # --per_gpu_eval_batch_size=4 \
  # --learning_rate 3e-5 \
  # --num_train_epochs 2.0 \
  # --max_seq_length 512 \
  # --doc_stride 128 \
  # --data_dir /HeySQuAD_json/ \
  # --output_dir roberta-large-human