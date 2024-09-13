export CUDA_VISIBLE_DEVICES='0'; nohup python asr_process_train.py --model s2t --model_path s2t-small-librispeech-asr --dataset heysquad_human --debug False --batch_size 16 --mode question > s2t_small.log 2>&1 &

export CUDA_VISIBLE_DEVICES='1'; nohup python asr_process_train.py --model s2t --model_path s2t-medium-librispeech-asr --dataset heysquad_human --debug False --batch_size 16 --mode question > s2t_medium.log 2>&1 &

export CUDA_VISIBLE_DEVICES='2'; nohup python asr_process_train.py --model s2t --model_path s2t-large-librispeech-asr --dataset heysquad_human --debug False --batch_size 16 --mode question > s2t_large.log 2>&1 &

export CUDA_VISIBLE_DEVICES='3'; nohup python asr_process_train.py --model whisper --model_path whisper-small.en --dataset heysquad_human --debug False --batch_size 16 --mode question > whisper-small.en.log 2>&1 &

export CUDA_VISIBLE_DEVICES='2'; nohup python asr_process_train.py --model whisper --model_path distil-small.en --dataset heysquad_human --debug False --batch_size 16 --mode question > distil-small.en.log 2>&1 &


#token level

export CUDA_VISIBLE_DEVICES='3'; nohup python asr_process_train.py --model s2t --model_path s2t-small-librispeech-asr --dataset heysquad_human --debug False --batch_size 16 --mode token > small-asr-train-token.log 2>&1 &

export CUDA_VISIBLE_DEVICES='3'; nohup python asr_process_train.py --model s2t --model_path s2t-large-librispeech-asr --dataset heysquad_human --debug False --batch_size 16 --mode token > large-asr-train-token.log 2>&1 &

export CUDA_VISIBLE_DEVICES='2'; nohup python asr_process_train.py --model s2t --model_path s2t-medium-librispeech-asr --dataset heysquad_human --debug False --batch_size 16 --mode token > medium-asr-train-token.log 2>&1  &