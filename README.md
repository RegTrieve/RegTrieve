# Install
- Requirements: torch 1.13.1, cuda 11.7, python 3.10
- Modify environment.yaml prefix to your conda install path. Modify the project path in the spoken
- Create Conda Env:  conda create -n system_regression python=3.10
- Run pip install -e .
- Run conda install -c pytorch faiss-cpu=1.8.0


# Prepare Resource Files
## For Dataset
- cd data/, fetch https://huggingface.co/datasets/yijingwu/HeySQuAD_machine and https://huggingface.co/datasets/yijingwu/HeySQuAD_human follows cloning repo instructions.
## For Pretrained Models
- mkdir pretrained_models/
- Download models in pretrained_models/
    - ASR models:
    - QA models: Roberta-large
### For fairseq s2t models pretraining
- To pretrain Fairseq models using the Librispeech example and integrate Heysquad data, follow these steps:
  - refer to official Fairseq repository and documentation (https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/librispeech_example.md) for guidelines on pretraining a fairseq model 
  - if you want to integrate heysquad data in fairseq pretrain or validation
    - refer to fairseq_pretrain_scripts/parquet_convert.py, to generate train-heysquad,dev-heysuqad in fairseq style
    - you may have to modify fairseq s2t data preprocess file (e.g. fairseq/examples/speech_to_text/prep_librispeech_data.py) to add "train-heysquad" and "dev-heysquad" and set download=False to use local data, while you may need to temporarily modify your torchaudio/datasets/librispeech.py file to enable the use of "train-heysquad" and "dev-heysquad" (Alternatively, you can simply replace the necessary files using those in the fairseq_pretrain_scripts/substitute directory. )
  -  if you want to convert fairseq s2t torch file (end with .pt) and vocabulary files (spm_unigram10000.model and spm_unigram10000.txt) to transformer style
     -  refer to the script fairseq_pretrain_scripts/convert2tsfms.sh

# Testing for original Regression
## Run ASR results
- Refer to asr/script/test.sh
- Outputs should be saved in predictions/asr/, including old mdoel predictions, new model predictions, and both model wer comparison results.
## Run QA results
- Refer to qa/script/infer_squad.sh
- Its inputs are in predictions/asr/, and outputs are saved in predictions/qa/
## Print Regression Results
- Refer to report/reg_cal.sh, to get the pipeline regression results with the previoused two steps.

# Train QA model
- train origin dev origin



# Run RAG Ensemble Model
- Run data_prep/asr_process_train.py to create the processed training data of given model both in question and token mode. Refer to data_prep/asr_process_train.sh. 
  
- Run data_prep/create_knn_qa.py both in question and token mode to create qa loss file. Note that we create loss_token in seperate files to speed up, thus may should be run twice. Refer to data_prep/create_knn.sh.
  
- Run data_prep/create_knn_asr.py to create asr loss file. Refer to data_prep/create_knn.sh. 

- Run asr/reg_reduce/reg_rag.py to finish ensemble of asr, qa question and qa token, refer to system_regression/asr/script/reduce_batch.sh