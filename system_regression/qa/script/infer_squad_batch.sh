OLD_FILE="s2t-small-librispeech-asr"
NEW_FILE="s2t-medium-librispeech-asr"

python ../run_squad.py --model_type roberta --model_name_or_path ../save/output-roberta-large-train_origin-dev_origin --do_eval --do_lower_case --predict_file human_transcribed/eval/$OLD_FILE.json --per_gpu_eval_batch_size=8 --max_seq_length=512 --doc_stride=128 --output_dir roberta-large-train-origin-dev-origin --eval_single True --overwrite_cache
python ../run_squad.py --model_type roberta --model_name_or_path ../save/output-roberta-large-train_origin-dev_origin --do_eval --do_lower_case --predict_file human_transcribed/eval/$NEW_FILE.json --per_gpu_eval_batch_size=8 --max_seq_length=512 --doc_stride=128 --output_dir roberta-large-train-origin-dev-origin --eval_single True --overwrite_cache

python ../run_squad.py --model_type roberta --model_name_or_path ../save/output-roberta-large-train_origin-dev_origin --do_eval --do_lower_case --predict_file human_transcribed/eval/ensemble/$OLD_FILE-$NEW_FILE-max.json --per_gpu_eval_batch_size=8 --max_seq_length=512 --doc_stride=128 --output_dir roberta-large-train-origin-dev-origin --eval_single True --overwrite_cache
python ../run_squad.py --model_type roberta --model_name_or_path ../save/output-roberta-large-train_origin-dev_origin --do_eval --do_lower_case --predict_file human_transcribed/eval/ensemble/$OLD_FILE-$NEW_FILE-avg.json --per_gpu_eval_batch_size=8 --max_seq_length=512 --doc_stride=128 --output_dir roberta-large-train-origin-dev-origin --eval_single True --overwrite_cache
python ../run_squad.py --model_type roberta --model_name_or_path ../save/output-roberta-large-train_origin-dev_origin --do_eval --do_lower_case --predict_file human_transcribed/eval/ensemble/$OLD_FILE-$NEW_FILE-pertub.json --per_gpu_eval_batch_size=8 --max_seq_length=512 --doc_stride=128 --output_dir roberta-large-train-origin-dev-origin --eval_single True --overwrite_cache
python ../run_squad.py --model_type roberta --model_name_or_path ../save/output-roberta-large-train_origin-dev_origin --do_eval --do_lower_case --predict_file human_transcribed/eval/ensemble/$OLD_FILE-$NEW_FILE-dropout.json --per_gpu_eval_batch_size=8 --max_seq_length=512 --doc_stride=128 --output_dir roberta-large-train-origin-dev-origin --eval_single True --overwrite_cache


lambdas=("0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0" "1.1" "1.2" "1.3" "1.4" "1.5" "1.6" "1.7" "1.8" "1.9" "2.0")


# asr loss
for ((i = 0; i < 21; i++)); do
    export CUDA_VISIBLE_DEVICES='2';python ../run_squad.py --model_type roberta --model_name_or_path ../save/output-roberta-large-train_origin-dev_origin --do_eval --do_lower_case --predict_file human_transcribed/eval/ensemble/$OLD_FILE-$NEW_FILE-asr-asrlambda${lambdas[$i]}.json  --per_gpu_eval_batch_size=8 --max_seq_length 512 --doc_stride 128 --output_dir roberta-large-train-origin-dev-origin --eval_single True --overwrite_cache

done

# qa question loss
for ((i = 0; i < 21; i++)); do
    export CUDA_VISIBLE_DEVICES='2';python ../run_squad.py --model_type roberta --model_name_or_path ../save/output-roberta-large-train_origin-dev_origin --do_eval --do_lower_case --predict_file human_transcribed/eval/ensemble/$OLD_FILE-$NEW_FILE-qa-question-qalambda${lambdas[$i]}.json  --per_gpu_eval_batch_size=8 --max_seq_length 512 --doc_stride 128 --output_dir roberta-large-train-origin-dev-origin --eval_single True --overwrite_cache
done

# qa question merge
for ((i = 0; i < 21; i++)); do
    export CUDA_VISIBLE_DEVICES='2';python ../run_squad.py --model_type roberta --model_name_or_path ../save/output-roberta-large-train_origin-dev_origin --do_eval --do_lower_case --predict_file human_transcribed/eval/ensemble/$OLD_FILE-$NEW_FILE-qa-question-merge-qalambda0.3-asrlambda0.3-qaweight${lambdas[$i]}.json  --per_gpu_eval_batch_size=8 --max_seq_length 512 --doc_stride 128 --output_dir roberta-large-train-origin-dev-origin --eval_single True --overwrite_cache

done
