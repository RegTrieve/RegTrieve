
nohup python train_model.py --model whisper --pretrained_model_path whisper-tiny.en --dataset heysquad_human --epochs 1 --dataset_size 1.0 --debug False --output_dir whisper_tiny_en_steps > train_whisper_tiny.log 2>&1  &

nohup python train_model.py --model whisper --pretrained_model_path whisper-tiny --dataset heysquad_human --epochs 1 --dataset_size 1.0 --debug False --output_dir whisper_tiny_steps > train_whisper_tiny.log 2>&1  &


nohup python train_model.py --model whisper --pretrained_model_path whisper-tiny --dataset heysquad_human --epochs 2 --debug False --output_dir whisper_tiny_steps > train_whisper_tiny.log &