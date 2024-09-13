from __future__ import print_function

import os
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, Subset
import re
from datasets import load_dataset
from system_regression.asr.utils import wer_normalize_text, wav2vec_normalize_text
import torch

from transformers import AutoProcessor
from transformers import AutoModelForCTC, TrainingArguments, Trainer
from torch.utils.data import IterableDataset
from system_regression.asr.utils import cal_md5
from system_regression.common.const import Constants



def find_files_matching_pattern(directory, pattern):
    matched_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if re.match(pattern, file):
                file_path = os.path.join(root, file)
                matched_files.append(file_path)
    return sorted(matched_files)


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset
        self.column_names=['id', 'input_features', 'length', 'question', 'input_ids']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming your dataset contains features 'input' and 'label'
        item = {key: val for key, val in self.data[idx].items()}
        return item
    
    def map(self, func):
        self.data = [func(x) for x in self.data]
    
    def filter(self, func):
        self.data = [x for x in self.data if func(x)]

class TorchIterDataset(IterableDataset):
    def __init__(self, hf_dataset, map_func):
        self.data = hf_dataset
        self.column_names=['id', 'input_features', 'length', 'question', 'input_ids']
        self.map_func = map_func

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            # Process the line (e.g., tokenize)
            # Here we'll just return the line as a dummy example
            yield self.map_func(x)    
    
    def filter(self, func):
        self.data = [x for x in self.data if func(x)]


def get_heysquad_datasets(processor, model, size=1.0, debugging=False, test_only=True, human_machine="human", ref_dev_ids=None, ref_train_ids=None, ref_dev_hashmap=None, ref_train_hashmap=None, split_ratio=1.0,train_shuffle=True):
    human_train_list=find_files_matching_pattern(Constants.heysquad_human_dir,"train.*parquet")
    human_dev_list=find_files_matching_pattern(Constants.heysquad_human_dir,"validation.*parquet")

    machine_train_list=find_files_matching_pattern(Constants.heysquad_machine_dir,"train.*parquet")
    machine_dev_list=find_files_matching_pattern(Constants.heysquad_machine_dir,"validation.*parquet")
    # Do not know how  to serielize iterable dataset

    def process_files(files, partition='test', if_shuffle=True, filter_ids = None, ref_hashmap= None):
         # num_proc=8,batched=True
        def process_audio_wav(data):
            idx = data["id"] if "id" in data else data['audio']['path']
            data["question"] = wav2vec_normalize_text([data["question"]])[0]
            batch = processor(data["audio"]["array"],
                                    sampling_rate=16_000,
                                    return_tensors="pt", text=data['question'])
            
            batch["input_length"] = len(batch['input_values'][0])
            batch['id'] = idx
            batch['question'] = data["question"]
            return batch
        
        def process_audio_whisper(data):
            idx = data["id"] if "id" in data else ref_hashmap[cal_md5(data['question'], data['context'])]
            # data["question"] = whisper_normalize_text([data["question"]])[0]
            batch = {}
            batch["input_features"] = processor(data["audio"]["array"],
                                    sampling_rate=16_000,
                                    return_tensors="pt").input_features[0]
    
            
            batch['id'] = idx
            #whitespace fix:
            data["question"] = ' '.join(data["question"].split())
            batch["labels"] = processor.tokenizer(data["question"], max_length=200, truncation=True).input_ids
            batch['question']  = data['question']

            return batch
        
        map_func_dict = {'s2t': lambda x:x, 'whisper': process_audio_whisper, 'wav2vec': process_audio_wav}

        if debugging:
            dataset=TorchIterDataset(load_dataset("parquet", data_files={partition:files},split=partition).select(range(300)), map_func=map_func_dict[model])
        else:
            dataset=TorchIterDataset(load_dataset("parquet", data_files={partition:files},split=partition), map_func=map_func_dict[model])
            #  len(dataset.data)
        
        #TODO: 一次map多个data, 利用processor的padding
        if filter_ids:
            # print("Filtering dataset...")
            if partition=="test":
                dataset.filter(lambda x: x['id'] in filter_ids)
            else:
                if ref_hashmap:
                    dataset.filter(lambda x: cal_md5(x['question'],x['context']) in ref_hashmap)
                else:
                    dataset.filter(lambda x: x['id'] in filter_ids)

        if model == 'wav2vec':
            dataset.map(process_audio_wav)
            n_data = len(dataset)
            idx = np.arange(n_data)
            np.random.shuffle(idx)
            split = int(np.floor(size * n_data))
            val_split = int(np.floor(split * split_ratio))
            train_idx = idx[:val_split]
            train_val_idx = idx[val_split:split]
            train_set = Subset(dataset, train_idx)
            train_val_set = Subset(dataset, train_val_idx)
            return train_set, train_val_set
        elif model == 'whisper':
            # dataset.map(process_audio_whisper)
            return dataset, dataset
        elif  model == 's2t':
            return dataset, dataset

    
    if debugging:
        dev_list = [human_dev_list[0]]
        train_list = [human_train_list[0]]
    else:
        dev_list = human_dev_list
        train_list = human_train_list

    if human_machine =="machine":
        dev_list = machine_dev_list
        train_list = machine_train_list

    test_set, test_val_set = process_files(dev_list, "test", False, ref_dev_ids, ref_dev_hashmap)

    if test_only:
        return None, None, test_set, test_val_set,
    else:
        train_set, train_val_set = process_files(train_list, "train", train_shuffle, ref_train_ids, ref_train_hashmap)
        return train_set, train_val_set, test_set, test_val_set

    
