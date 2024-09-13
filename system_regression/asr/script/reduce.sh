# s2t small medium - avg
nohup python regbug_reducing.py --old_model s2t --new_model s2t --oldmodel_path s2t-small-librispeech-asr --newmodel_path s2t-medium-librispeech-asr --dataset heysquad_human --ensemble_type logit --ensemble avg --debug False --weight_adj False --batch_size 8 > avg.log 2>&1 &
nohup python regbug_reducing.py --old_model s2t --new_model s2t --oldmodel_path s2t-small-librispeech-asr --newmodel_path s2t-medium-librispeech-asr --dataset heysquad_human --ensemble_type logit --ensemble max --debug False --weight_adj False --batch_size 8 > avg.log 2>&1 &

# whisper tiny en - base en - avg
nohup python regbug_reducing.py --old_model whisper --new_model whisper --oldmodel_path whisper-tiny.en --newmodel_path whisper-base.en --dataset heysquad_human --ensemble_type logit --ensemble avg --debug False --weight_adj False --batch_size 8 > avg.log 2>&1 &

# whisper small en - distil small - avg
nohup python regbug_reducing.py --old_model whisper --new_model whisper --oldmodel_path whisper-small.en --newmodel_path distil-small.en --dataset heysquad_human --ensemble_type logit --ensemble avg --debug False --weight_adj False --batch_size 16 > avg.log 2>&1 &