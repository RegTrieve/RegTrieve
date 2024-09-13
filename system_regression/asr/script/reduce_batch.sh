OLD_MODEL="s2t-small-librispeech-asr"
NEW_MODEL="s2t-medium-librispeech-asr"

python ../test_model.py --old_model s2t --new_model s2t --oldmodel_path $OLD_MODEL --newmodel_path $NEW_MODEL --dataset heysquad_human  --debug False --batch_size 8

# baseline ensemble
python ../reg_reduce/regbug_reducing.py --old_model s2t --new_model s2t --oldmodel_path $OLD_MODEL --newmodel_path $NEW_MODEL --dataset heysquad_human --ensemble_type logit --ensemble max --debug False --weight_adj False --batch_size 8
python ../reg_reduce/regbug_reducing.py --old_model s2t --new_model s2t --oldmodel_path $OLD_MODEL --newmodel_path $NEW_MODEL --dataset heysquad_human --ensemble_type logit --ensemble avg --debug False --weight_adj False --batch_size 8
python ../reg_reduce/regbug_reducing.py --old_model s2t --new_model s2t --oldmodel_path $OLD_MODEL --newmodel_path $NEW_MODEL --dataset heysquad_human --ensemble_type logit --ensemble pertub --debug False --weight_adj False --batch_size 8
python ../reg_reduce/regbug_reducing.py --old_model s2t --new_model s2t --oldmodel_path $OLD_MODEL --newmodel_path $NEW_MODEL --dataset heysquad_human --ensemble_type logit --ensemble dropout --debug False --weight_adj False --batch_size 8

#For asr loss knn ensemble
python ../reg_reduce/reg_rag.py --old_model s2t --new_model s2t --oldmodel_path $OLD_MODEL --newmodel_path $NEW_MODEL --dataset heysquad_human --debug False --rag_type knn --index_metric L2 --old_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_question.json --new_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_question.json --mode asr 

#For qa question knn ensemble
python ../reg_reduce/reg_rag.py --old_model s2t --new_model s2t --oldmodel_path $OLD_MODEL --newmodel_path $NEW_MODEL --dataset heysquad_human --debug False --rag_type knn --index_metric L2 --old_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_question.json --new_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_question.json --mode question --model_logging True 

#For qa question merge knn ensemble
python ../reg_reduce/reg_rag.py --old_model s2t --new_model s2t --oldmodel_path $OLD_MODEL --newmodel_path $NEW_MODEL --dataset heysquad_human --debug False --rag_type knn --index_metric L2 --old_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_question.json --new_qa_loss_file output-roberta-large-train_origin-dev_origin_qa_loss_question.json --mode question-merge --model_logging True 