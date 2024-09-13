python pre_voc2json.py --voc_text_file /path_to/pretrain-libspeech/fairseq/data-test/spm_unigram10000.txt --voc_json_file /path_to/pretrain-libspeech/fairseq/data-test/spm_unigram10000.json

OUTPUT_DIR=/path_to/pretrained_models/pretrain/
PT_PATH=/path_to/pretrain-libspeech/fairseq/save_dev_other/avg_last_10_checkpoint_to_5w.pt 

mkdir $OUTPUT_DIR
python convert_tsfms.py \
 --pytorch_dump_folder_path $OUTPUT_DIR \
 --model_size small \
 --fairseq_path $PT_PATH \
 --vocab_file /path_to/pretrain-libspeech/fairseq/data/vo-full/spm_unigram10000.json \
 --sp_model /path_to/pretrain-libspeech/fairseq/data/vo-full/spm_unigram10000.model