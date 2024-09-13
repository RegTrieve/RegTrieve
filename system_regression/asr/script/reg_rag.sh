#For asr loss knn ensemble
export CUDA_VISIBLE_DEVICES='2';nohup python reg_rag.py --old_model s2t --new_model s2t --oldmodel_path s2t-small-librispeech-asr --newmodel_path s2t-medium-librispeech-asr --dataset heysquad_human --debug False --rag_type knn --index_metric L2 --old_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_question.json --new_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_question.json --mode asr > s2tsmall-medium-rag-asr-loss-ensemble.log 2>&1 &

export CUDA_VISIBLE_DEVICES='3';nohup python reg_rag.py --old_model s2t --new_model s2t --oldmodel_path s2t-small-librispeech-asr --newmodel_path s2t-large-librispeech-asr --dataset heysquad_human --debug False --rag_type knn --index_metric L2 --old_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_question.json --new_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_question.json --mode asr > s2tsmall-large-rag-asr-loss-ensemble.log 2>&1 &

#For qa question knn ensemble
export CUDA_VISIBLE_DEVICES='0';nohup python reg_rag.py --old_model s2t --new_model s2t --oldmodel_path s2t-small-librispeech-asr --newmodel_path s2t-medium-librispeech-asr --dataset heysquad_human --debug False --rag_type knn --index_metric L2 --old_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_question.json --new_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_question.json --mode question --model_logging True > s2tsmall-medium-rag-qa-question-loss.log 2>&1 &

export CUDA_VISIBLE_DEVICES='1';nohup python reg_rag.py --old_model s2t --new_model s2t --oldmodel_path s2t-small-librispeech-asr --newmodel_path s2t-large-librispeech-asr --dataset heysquad_human --debug False --rag_type knn --index_metric L2 --old_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_question.json --new_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_question.json --mode question  --model_logging True > s2tsmall-large-rag-qa-question-loss.log 2>&1 &

#For qa question merge knn ensemble
export CUDA_VISIBLE_DEVICES='2';nohup python reg_rag.py --old_model s2t --new_model s2t --oldmodel_path s2t-small-librispeech-asr --newmodel_path s2t-medium-librispeech-asr --dataset heysquad_human --debug False --rag_type knn --index_metric L2 --old_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_question.json --new_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_question.json --mode question-merge --model_logging True > s2tsmall-medium-rag-qa-question-merge-loss.log 2>&1 &

export CUDA_VISIBLE_DEVICES='3';nohup python reg_rag.py --old_model s2t --new_model s2t --oldmodel_path s2t-small-librispeech-asr --newmodel_path s2t-large-librispeech-asr --dataset heysquad_human --debug False --rag_type knn --index_metric L2 --old_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss.json --new_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss.json --mode question-merge  --model_logging True > s2tsmall-large-rag-qa-question-merge-loss.log 2>&1 &


#For qa token loss knn ensemble
export CUDA_VISIBLE_DEVICES='1';nohup python reg_rag.py --old_model s2t --new_model s2t --oldmodel_path s2t-small-librispeech-asr --newmodel_path s2t-medium-librispeech-asr --dataset heysquad_human --debug False --rag_type knn --index_metric L2 --old_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_token.json --new_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_token.json --mode token --model_logging True > s2tsmall-medium-rag-qa-token-loss.log 2>&1 &

export CUDA_VISIBLE_DEVICES='0';nohup python reg_rag.py --old_model s2t --new_model s2t --oldmodel_path s2t-small-librispeech-asr --newmodel_path s2t-large-librispeech-asr --dataset heysquad_human --debug False --rag_type knn --index_metric L2 --old_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_token.json --new_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_token.json --mode token  --model_logging True > s2tsmall-large-rag-qa-token-loss.log 2>&1 &



#For qa token-merge loss knn ensemble
export CUDA_VISIBLE_DEVICES='2';nohup python reg_rag.py --old_model s2t --new_model s2t --oldmodel_path s2t-small-librispeech-asr --newmodel_path s2t-medium-librispeech-asr --dataset heysquad_human --debug False --rag_type knn --index_metric L2 --old_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_token.json --new_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_token.json --mode token-merge --model_logging True > s2tsmall-medium-rag-qa-token-merge-loss-qa.log 2>&1 &


export CUDA_VISIBLE_DEVICES='1';nohup python reg_rag.py --old_model s2t --new_model s2t --oldmodel_path s2t-small-librispeech-asr --newmodel_path s2t-large-librispeech-asr --dataset heysquad_human --debug False --rag_type knn --index_metric L2 --old_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_token.json --new_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_token.json --mode token-merge  --model_logging True > s2tsmall-large-rag-qa-token-merge-loss-qa.log 2>&1 &





