OLD_FILE="s2t-small-librispeech-asr"
# NEW_FILE="s2t-medium-librispeech-asr"
NEW_FILE="s2t-large-librispeech-asr"

## Ensemble asr loss
asr_lambdas=("0.0" "0.1" "0.3" "0.5" "0.7" "0.9" "1.1" "1.3")
for ((i = 0; i < 8; i++)); do
    export CUDA_VISIBLE_DEVICES='2';python ../run_squad.py --model_type roberta --model_name_or_path /path_to/system_regression/qa/save/output-roberta-large-train_origin-dev_origin --do_eval --do_lower_case --predict_file human_transcribed/eval/ensemble/$OLD_FILE-$NEW_FILE-asr-asrlambda${asr_lambdas[$i]}.json  --per_gpu_eval_batch_size=8 --max_seq_length 512 --doc_stride 128 --output_dir roberta-large-train-origin-dev-origin --eval_single True --overwrite_cache
done

## Ensemble qa question loss
# qa_lambdas=("0.0" "0.1" "0.3" "0.5" "0.7" "0.9" "1.1" "1.3")
# for ((i = 0; i < 8; i++)); do
#     export CUDA_VISIBLE_DEVICES='3';python ../run_squad.py --model_type roberta --model_name_or_path /path_to/system_regression/qa/save/output-roberta-large-train_origin-dev_origin --do_eval --do_lower_case --predict_file human_transcribed/eval/ensemble/$OLD_FILE-$NEW_FILE-qa-question-qalambda${qa_lambdas[$i]}.json  --per_gpu_eval_batch_size=8 --max_seq_length 512 --doc_stride 128 --output_dir roberta-large-train-origin-dev-origin --eval_single True --overwrite_cache
# done

# Ensemble qa question merge loss
# asr_lambda=0.3
# qa_lambda=0.3
# qa_asr_weights=("0.0" "0.2" "0.4" "0.6" "0.8" "1.0" "1.2" "1.4")
# # qa_asr_weights=("0.0" "0.3" "0.6" "0.9" "1.1" "1.3" "1.5" "1.7")
# for ((i = 0; i < 8; i++)); do
#     export CUDA_VISIBLE_DEVICES='1';python ../run_squad.py --model_type roberta --model_name_or_path /path_to/system_regression/qa/save/output-roberta-large-train_origin-dev_origin --do_eval --do_lower_case --predict_file human_transcribed/eval/ensemble/$OLD_FILE-$NEW_FILE-qa-question-merge-qalambda$qa_lambda-asrlambda$asr_lambda-qaweight${qa_asr_weights[$i]}.json  --per_gpu_eval_batch_size=8 --max_seq_length 512 --doc_stride 128 --output_dir roberta-large-train-origin-dev-origin --eval_single True --overwrite_cache
# done

## Ensemble qa token loss
# qa_lambdas=("0.0" "0.1" "0.3" "0.5" "0.7" "0.9" "1.1" "1.3")
# for ((i = 0; i < 8; i++)); do
#     export CUDA_VISIBLE_DEVICES='3';python ../run_squad.py --model_type roberta --model_name_or_path /path_to/system_regression/qa/save/output-roberta-large-train_origin-dev_origin --do_eval --do_lower_case --predict_file human_transcribed/eval/ensemble/$OLD_FILE-$NEW_FILE-qa-token-qalambda${qa_lambdas[$i]}.json  --per_gpu_eval_batch_size=8 --max_seq_length 512 --doc_stride 128 --output_dir roberta-large-train-origin-dev-origin --eval_single True --overwrite_cache
# done


# Ensemble qa token merge loss
# asr_lambda=0.3
# qa_lambda=0.3
# qa_asr_weights=("0.0" "0.2" "0.4" "0.6" "0.8" "1.0" "1.2" "1.4")
# # qa_asr_weights=("0.0" "0.3" "0.6" "0.9" "1.1" "1.3" "1.5" "1.7")
# for ((i = 0; i < 8; i++)); do
#     export CUDA_VISIBLE_DEVICES='1';python ../run_squad.py --model_type roberta --model_name_or_path /path_to/system_regression/qa/save/output-roberta-large-train_origin-dev_origin --do_eval --do_lower_case --predict_file human_transcribed/eval/ensemble/$OLD_FILE-$NEW_FILE-qa-token-merge-qalambda$qa_lambda-asrlambda$asr_lambda-qaweight${qa_asr_weights[$i]}.json  --per_gpu_eval_batch_size=8 --max_seq_length 512 --doc_stride 128 --output_dir roberta-large-train-origin-dev-origin --eval_single True --overwrite_cache
# done
