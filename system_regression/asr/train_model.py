#TODO: 1. add finetuning step for qa model(finetune with heysquad_human train)
#TODO: 2. add finetuning step for asr model

from __future__ import print_function

import os
import argparse
import socket
import time

# import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys

from tqdm import tqdm
import json
from utils import get_ref_ids, whisper_wer_normailize_text, DataCollatorCTCWithPadding, DataCollatorSpeechSeq2SeqWithPadding 
from system_regression.data_prep.heysquad import get_heysquad_datasets
from transformers import AutoProcessor
from transformers import AutoModelForCTC, TrainingArguments, Trainer, Seq2SeqTrainer
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments
from system_regression.common.str2bool import str2bool
def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    #TODO: add batch size, seem should add padding
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=300, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_algorithm', type=str, default='adam', help='learning algorithm')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,225,300', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='s2t',
                        choices=['s2t', 'whisper', 'wav2vec'])

    parser.add_argument('--pretrained_model_path', type=str)
    parser.add_argument('--finetune', type=int, default=0, choices=[0, 1])                                
    parser.add_argument('--pretrained', type=int, default=0, choices=[0, 1])
    parser.add_argument('--dataset', type=str, default='heysquad', choices=['heysquad_human'], help='dataset')
    parser.add_argument('--category', type=str, default='c', choices=['b', 'c', 'i', 'p', 'b+c']) # category for imageclef
    parser.add_argument('--version', type=str, default='101', choices=['101', '256'])

    # block depth
    parser.add_argument('--block_depth', type=str, default='3,3,3', help='the depth of each block')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    # train method
    parser.add_argument('--train_method', type=str, default='CE', choices=['CE', 'FL'])
    parser.add_argument('--miscls_path', type=str, default=None, help='misclassified samples\' filenames' )
    parser.add_argument('--debug', type=str2bool, default=False, help='misclassified samples\' filenames' )
    parser.add_argument('--output_dir', type=str)

    opt = parser.parse_args()

    # set the path according to the environment
    opt.model_path = './save/models'
    opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    iterations = opt.block_depth.split(',')
    opt.block_depth = list([])
    for it in iterations:
        opt.block_depth.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)


    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

import numpy as np

def main():
    best_acc = 0
    last_acc = 0
    device = torch.device("cuda")

    opt = parse_option()


    if opt.model =='wav2vec':
    # model
        processor = AutoProcessor.from_pretrained(opt.pretrained_model_path)
        tokenizer = processor.tokenizer
        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        model = AutoModelForCTC.from_pretrained(
        opt.pretrained_model_path,
        ctc_loss_reduction="mean",
        pad_token_id=tokenizer.pad_token_id,
    )
        # model = model.from_pretrained(opt.pretrained_model_path)
        model = model.to(device)
        model.freeze_feature_encoder()

    elif opt.model =='whisper':
        processor = AutoProcessor.from_pretrained(opt.pretrained_model_path, language="en", task="transcribe")

        model = WhisperForConditionalGeneration.from_pretrained(opt.pretrained_model_path)
        model.generation_config.language = "en"
        model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids = None
        model.freeze_encoder()

        # model = model.to(device)
        model.train()
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
        
        

    # dataloader
    if opt.dataset == 'heysquad_human':
        ref_dev_file = '/path_to/dev-common-original-1002.json'#TODO
        with open(ref_dev_file, 'r') as f:
            dev_data = json.load(f)
        dev_ref_ids = get_ref_ids(dev_data)
        ref_train_file = '/path_to/train-common-original-48849.json'#TODO
        with open(ref_train_file, 'r') as f:
            train_data = json.load(f)
        train_ref_ids = get_ref_ids(train_data)

        #TODO: 只取一定比例的训练数据
        train_set, val_set, test_set, test_val_set = get_heysquad_datasets(processor, model=opt.model, ref_dev_ids=dev_ref_ids, ref_train_ids=train_ref_ids, debugging=False, test_only=False) 

    # 拼接两个 DataLoader 对应的 Dataset
        # merge_test_set = ConcatDataset([test_val_set, test_set])
        # if train_set is not None:
        #     merge_train_set =  ConcatDataset([train_set, val_set])
    else:
        raise NotImplementedError(opt.dataset)
    

    training_args = Seq2SeqTrainingArguments(
    output_dir=opt.output_dir,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=200,
    gradient_checkpointing=True,
    predict_with_generate=True,
    generation_max_length=200,
    fp16=True,
    save_steps=1000,
    eval_steps=1000,
    max_steps=30000,
    # group_by_length=True,
    save_strategy='steps',
    logging_steps=500,
    per_device_eval_batch_size=16,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    evaluation_strategy='steps',
    eval_accumulation_steps=4
)
    

    import evaluate
    wer = evaluate.load("/path_to/evaluate_huggingface/evaluate/evaluate-main/metrics/wer")#TODO

    #for whisper
    def compute_metrics(pred):
        # pred_logits = pred.predictions[0]
        # pred_ids = np.argmax(pred_logits, axis=-1)
        pred_ids = pred.predictions
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = whisper_wer_normailize_text(processor.batch_decode(pred_ids, skip_special_tokens=True))
        label_str = whisper_wer_normailize_text(processor.batch_decode(pred.label_ids, skip_special_tokens=True))

        cur_wer = wer.compute(predictions=pred_str, references=label_str)
        return {"wer": cur_wer}


    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=test_set,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == '__main__':
    main()
