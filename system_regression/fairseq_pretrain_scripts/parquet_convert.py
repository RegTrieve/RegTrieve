from datasets import load_dataset
import os
from pydub import AudioSegment
import numpy as np
import soundfile as sf
import collections
import re

def find_files_matching_pattern(directory, pattern):
    matched_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if re.match(pattern, file):
                file_path = os.path.join(root, file)
                matched_files.append(file_path)
    return sorted(matched_files)


heysquad_data_path="/path_to/system_regression/data/HeySQuAD_human/data" # your path to data of heysquad dataset
human_train_list=find_files_matching_pattern(heysquad_data_path,"train.*parquet")
human_dev_list=find_files_matching_pattern(heysquad_data_path,"validation.*parquet")

'''Modify here to decide whether to convert training data or test data'''
files=human_train_list
# files=human_dev_list
partition="train"
# partition="test"
output_dir="/path_to/pretrain-libspeech/fairseq/convert_data/train-heysquad"
# output_dir="/path_to/pretrain-libspeech/fairseq/convert_data/dev-heysquad"

data=load_dataset("parquet", data_files={partition:files},split=partition)
# convert audio file to flac format
def convert_audio(audio_array, sampling_rate, flac_output_path,wav_output_path):
    sf.write(wav_output_path, audio_array, sampling_rate)
    # Save as FLAC file
    song = AudioSegment.from_wav(wav_output_path)
    song.export(flac_output_path,format = "flac")
    os.remove(wav_output_path)



speaker_id="12138"

context_hash_table=dict()
context_id_hash_table=collections.defaultdict(int)
trans_hash_table=collections.defaultdict(str)
current_parag_value=0
if not os.path.exists(os.path.join(output_dir,speaker_id)):
    os.mkdir(os.path.join(output_dir,speaker_id))

# generate audio file in flac format (Audio files of the same paragraph in the same folder)
for record in data:
    audio_array = record['audio']['array']  
    sampling_rate = record['audio']['sampling_rate']  
    # audio_id = record['id']  
    context=record["context"]
    audio_id = context_id_hash_table[context]
    context_id_hash_table[context]+=1
    ground_truth=record["question"]
    if context not in context_hash_table:
        context_hash_table[context]=current_parag_value
        if not os.path.exists(os.path.join(output_dir,speaker_id,str(context_hash_table[context]))):
            os.mkdir(os.path.join(output_dir,speaker_id,str(context_hash_table[context])))
        current_parag_value+=1
    parag_id=str(context_hash_table[context])
    flac_prefix=f"{speaker_id}-{parag_id}-{audio_id:04d}"
    flac_dir=os.path.join(output_dir,speaker_id,parag_id)
    trans_path=os.path.join(flac_dir,f"{speaker_id}-{parag_id}.trans.txt")
    flac_output_path = os.path.join(flac_dir, f"{flac_prefix}.flac")
    wav_output_path = os.path.join(flac_dir, f"{audio_id:04d}.wav")
    convert_audio(audio_array, sampling_rate, flac_output_path,wav_output_path)
    # save trans data
    trans_hash_table[trans_path]+=f"{flac_prefix} {ground_truth.upper()}\n"
    print(f"Saved {flac_output_path} {wav_output_path}" )
# generate translation file of every folder
for trans_file in trans_hash_table.keys():
    with open(trans_file,'w',encoding='utf-8') as f:
        f.write(trans_hash_table[trans_file])
        f.close() 

print("All audio files have been saved.")