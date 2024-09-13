#处理training data里每条question, 获取其中的loss, 按照question_id保存

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import glob
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from system_regression.common.const import Constants
import math

import transformers
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers.trainer_utils import is_main_process
from transformers_interpret import QuestionAnsweringExplainer
from system_regression.asr.utils import load_qa_gt, wer_normalize_text

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
import json
import ujson
import re

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()

#TODO: 添加train dataset的cache，不然每次做数据预处理太慢了

class IndexedDataset(Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        return data, index

def qa_loss(args, train_dataset, model, tokenizer, all_qas_id):
    """Train the model"""

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataset = IndexedDataset(train_dataset)

    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    args.model_name_or_path = os.path.join(Constants.pretrained_model_dir, args.model_name_or_path)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    steps_trained_in_current_epoch = 0

    model.zero_grad()
    # Added here for reproducibility
    set_seed(args)

    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    loss_fct = CrossEntropyLoss(reduction='none')
    loss_dict = {} # {qas_id: {'start_loss': ,'end_loss':, 'avg_loss':}}
    # TODO: add qa hidden state embed, and qa index
    
    for step, batch in enumerate(epoch_iterator):
        # Skip past any already trained steps if resuming training
        if steps_trained_in_current_epoch > 0:
            steps_trained_in_current_epoch -= 1
            continue

        model.eval()
        batch, batch_indices = batch
        batch = tuple(t.to(args.device) for t in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }

        if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
            del inputs["token_type_ids"]

        if args.model_type in ["xlnet", "xlm"]:
            inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
            if args.version_2_with_negative:
                inputs.update({"is_impossible": batch[7]})
            if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                inputs.update(
                    {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                )
        with torch.no_grad():
            outputs = model(**inputs)
            # outputs = model(**inputs, output_hidden_states=True)
            ignored_index = outputs['start_logits'].size(1)
            start_positions = inputs['start_positions'].clamp(0, ignored_index)
            end_positions = inputs['end_positions'].clamp(0, ignored_index)
            start_loss = loss_fct(outputs['start_logits'], start_positions)
            end_loss = loss_fct(outputs['end_logits'], end_positions)
            total_loss = (start_loss + end_loss) / 2
            qas_id = [all_qas_id[i] for i in batch_indices]
        for i, q_id in enumerate(qas_id):
            loss_dict[q_id] = {
            'start_loss': start_loss[i].cpu().item(),
            'end_loss': end_loss[i].cpu().item(),
            'total_loss': total_loss[i].cpu().item()
            }
        # model outputs are always tuple in transformers (see doc)
        # loss = outputs[0]

    return loss_dict

#load train file and call QA Explainer
#qa model如果对该样本预测的答案错误应该怎么办？
#TODO: 1. 区分出qa model预测的start position与 end position正确与错误的情况(还有其对应的logits以及loss的情况)。
    #目前只考虑正确的情况，将正确的token的ig从高到低排序(负的也要在里面)
# 2. 获得了每个qa split token的权重，都要保存下来，后续model predict的时候，如果某个asr token覆盖多个qa token，将其avg/max/sum?  覆盖其中一个qa token的一部分，又该怎么办？这个似乎要看下integrated gradient怎么计算的。
#TODO:如何处理标点符号问题？s2t model将这些标点去除了，但是whisper model似乎在推理时还没有去除这些标点符号？
def qa_ig(args, train_file, model, tokenizer, save_file, cache_file,fast_tokenizer=None, asr_tokenizers=None, asr_tokenizer_names=None, asr_special_chars=None):
    data = load_qa_gt(train_file)
    qa_special_char = 'Ġ'
    end_str = '</s>'
    def get_question_attrs(attributions):
        res = []
        for item in attributions[1:]:
            res.append(item)
            #question的结束符对qa answer的影响也要考虑
            if item[0] == end_str:
                break
        return res
    
    def normalize_token(token, special_char):
        if token[0]==special_char:
            token = token[1:]
        return token.lower()
    
    def check_tokenize(tokens, question, special_char):
        new_tokens = []
        for i, token in enumerate(tokens):
            if token == end_str:
                continue
            if token[0]== special_char:
                new_tokens.append(token[1:])
            else:
                if i==0:
                    new_tokens.append(token)
                else:
                    new_tokens[-1] += token
        return ' '.join(new_tokens) == question
    
    def align_token_attrs(qa_attrs, qa_tokens, asr_tokens,asr_special_char):
        import string
        #TODO: 把后续会在通过normalize去掉的标点符号的token的attribute设为0
        res =  [] # res里放的是asr_tokens list对应的权重，直接按照index顺序索引
        matched_tokens= [] #  for debugging
        qa_i= 0
        asr_i =0
        while asr_i <len(asr_tokens):
            asr_token = normalize_token(asr_tokens[asr_i], asr_special_char)
            qa_token =  normalize_token(qa_tokens[qa_i], qa_special_char)
            cur_asr_count = 1 
            cur_qa_attr_amount =  0
            #匹配成功
            if qa_token==asr_token:
                res.append(qa_attrs[qa_i][1])
                matched_tokens.append(qa_token)
                asr_i += 1
                qa_i += 1
            #匹配失败，需要将asr token or qa token向后延伸
            else:
                cur_qa_attr_amount = qa_attrs[qa_i][1]
                while qa_i < len(qa_tokens) and asr_i <len(asr_tokens):
                    if len(qa_token) < len(asr_token):
                        #因为下一个qa token也是完整的单词，无法extend qa token
                        if qa_tokens[qa_i+1][0] == qa_special_char:
                            print('Error match: not enough qa_token.  %s %s : %s %s' %(qa_token, qa_tokens[qa_i+1], asr_token, asr_tokens[asr_i+1]))
                            break
                        qa_i += 1
                        qa_token += normalize_token(qa_tokens[qa_i], qa_special_char)
                        cur_qa_attr_amount += qa_attrs[qa_i][1]
                    elif len(qa_token) > len(asr_token):
                        #因为下一个asr token也是完整的单词，无法extend qa token
                        if asr_tokens[asr_i+1][0] == asr_special_char:
                            print('Error match: not enough asr_token.  %s %s : %s %s' %(qa_token, qa_tokens[qa_i+1], asr_token, asr_tokens[asr_i+1]))
                            break
                        asr_i += 1
                        asr_token += normalize_token(asr_tokens[asr_i], asr_special_char)
                        cur_asr_count +=1

                    elif qa_token == asr_token:
                        for _ in range(cur_asr_count):
                            res.append(cur_qa_attr_amount/cur_asr_count)
                        matched_tokens.append(qa_token)
                        asr_i += 1
                        qa_i +=1
                        break
                if qa_token!=asr_token:
                    print('Error match token:  %s: %s' %(qa_token, asr_token))
        if  len(asr_tokens) !=len(res):
            print('Error match sentence')
            return None
        return  res
    
    
    norm_map = {
        'whisper': False,
        's2t':True
    }  # {tokenizer_name: if_need_norm}
    ascii_map = {
        'Ã©': 'é',
        'Ãī': 'É',
        'ÅĤ': 'ł',
        'Ã¼': 'ü'
    }

    def map_ascii_tokens(tokens):
        new_tokens = []
        for token in tokens:
            for key in ascii_map:
                token =  token.replace(key, ascii_map[key])
            new_tokens.append(token)
        new_tokens.append(end_str) 
        return new_tokens


    ig_dict = {} #{q_id: {token_str: ig_sore}}, 这里的token_str后续会被asr tokenizer再进行tokenize，然后进行匹配
    skip_count = 0
    if cache_file and os.path.exists(cache_file):
        with open(cache_file) as f:
            ig_dict = json.load(f)
    max_length= 450 # for robera large
    qa_max_length = 200
    for count, i in tqdm(enumerate(list(data.keys()))):
        if i in ig_dict:
            print('skip cache count %d' % count)
            continue
        question = data[i]['question']
        question = re.sub(r'\s{2,}', ' ', question)
        qa_ids = tokenizer(question, max_length=qa_max_length)['input_ids']
        
        context_ids = tokenizer(data[i]['context'], max_length=max_length-len(qa_ids))['input_ids']
        context = tokenizer.decode(context_ids)

        qa_tokens = map_ascii_tokens(tokenizer.tokenize(question))

        norm_question = wer_normalize_text([question])[0]
        norm_qa_tokens=map_ascii_tokens(tokenizer.tokenize(norm_question))
        #TODO: 这种非ascci码的想办法解决
        if not check_tokenize(qa_tokens, question, qa_special_char):
            skip_count += 1
            print('skip %d/%d for question %s token: %s, %s'%(skip_count, count, i, question, qa_tokens))
            continue

        qa_explainer = QuestionAnsweringExplainer(
        model,
        tokenizer
        )
        model.zero_grad()
        word_attributions = qa_explainer(
        question,
        context,
        internal_batch_size=8
    )
        model.zero_grad()
        norm_word_attributions = qa_explainer(
        norm_question,
        context,
        internal_batch_size=8
    )
        data[i]['start_attrs'] = get_question_attrs(word_attributions['start'])
        data[i]['end_attrs'] = get_question_attrs(word_attributions['end'])
        data[i]['norm_start_attrs'] = get_question_attrs(norm_word_attributions['start'])
        data[i]['norm_end_attrs'] = get_question_attrs(norm_word_attributions['end'])

        # print(tokenizer.tokenize(question))
        #TODO: modify this
        for j in range(len(asr_tokenizers)): 
            asr_tokenizer = asr_tokenizers[j]
            if  norm_map[asr_tokenizer_names[j]]:
                asr_tokens =  map_ascii_tokens(asr_tokenizer.tokenize(norm_question))
                #TODO:  解决input_ids与asr_token的长度不匹配问题
                input_ids = asr_tokenizer(norm_question)['input_ids']
                if not check_tokenize(asr_tokens, norm_question, asr_special_chars[j]):
                    print('skip for asr token: %s, %s'%(norm_question, asr_tokens))
                    continue
                # if len(input_ids)  

                start_attrs =  align_token_attrs(data[i]['norm_start_attrs'], norm_qa_tokens,asr_tokens, asr_special_chars[j])
                end_attrs =  align_token_attrs(data[i]['norm_end_attrs'], norm_qa_tokens,asr_tokens, asr_special_chars[j])
            else:
                asr_tokens =  map_ascii_tokens(asr_tokenizer.tokenize(question))
                input_ids = asr_tokenizer(question)['input_ids']
                if len(input_ids) != len(asr_tokens) +2:
                    print('error len') 

                if not check_tokenize(asr_tokens, question, asr_special_chars[j]):
                    print('skip for asr token: %s, %s'%(question, asr_tokens))
                    continue
                start_attrs =  align_token_attrs(data[i]['start_attrs'], qa_tokens, asr_tokens, asr_special_chars[j])
                end_attrs =  align_token_attrs(data[i]['end_attrs'], qa_tokens,asr_tokens, asr_special_chars[j])
            
            if start_attrs is None  or end_attrs is None:
                continue
            if i not in ig_dict:
                ig_dict[i]  = {}
            ig_dict[i][asr_tokenizer_names[j]] = {'start_attrs':start_attrs, 'end_attrs': end_attrs}
        
        if count % 50 ==0:
            with open(save_file, 'w') as f:
                json.dump(ig_dict ,f)
    return ig_dict
    


def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    padding_strategy="max_length",
    return_dataset=False,
    threads=1,
    tqdm_enabled=True,
):
    from functools import partial
    from multiprocessing import Pool, cpu_count
    from transformers.data.processors.squad import squad_convert_example_to_features_init, squad_convert_example_to_features
    # Defining helper methods
    features = []

    threads = min(threads, cpu_count())
    with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
                disable=not tqdm_enabled,
            )
        )

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)
        all_qas_id = [f.qas_id for f in features]

        if not is_training:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask
            )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_is_impossible
            )

        return features, dataset,  all_qas_id
    else:
        return features, all_qas_id

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    if evaluate:
        cached_features_file = os.path.join(
            input_dir,
            "cached_dev_{}_{}".format(
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.predict_file
            ),
        )
    else:
        cached_features_file = os.path.join(
            input_dir,
            "cached_train_{}_{}".format(
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.train_file # 可能是绝对路径，要处理 
            ),
        )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warning("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
            else:
                examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

        features, dataset, all_qas_id = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

        if args.local_rank in [-1, 0]:
            pass
            #not support cache data file currently
            # logger.info("Saving features into cached file %s", cached_features_file)
            # torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features, all_qas_id
    return dataset, all_qas_id


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from huggingface.co",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help=(
            "The maximum total input sequence length after WordPiece tokenization. Sequences "
            "longer than this will be truncated, and sequences shorter than this will be padded."
        ),
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help=(
            "The maximum number of tokens for the question. Questions longer than this will "
            "be truncated to this length."
        ),
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        ),
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help=(
            "If true, all of the warnings related to data processing will be printed. "
            "A number of warnings are expected for a normal SQuAD evaluation."
        ),
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help=(
            "language id of input for language-specific xlm models (see"
            " tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)"
        ),
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help=(
            "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
            "See details at https://nvidia.github.io/apex/amp.html"
        ),
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    
    parser.add_argument("--eval_single", type=bool, default=False)
    parser.add_argument("--save_eval_file", type=str, default="")
    parser.add_argument("--metric_type", type=str, default="loss")
    
    #与asr_process_train的三种mode对应
    parser.add_argument("--mode", type=str, default="single",choices=['question', 'token', 'cascade'])

    args = parser.parse_args()
    save_loss_dir = "%s/%s/" % (Constants.qa_knn_cache_dir, args.train_file.split('/')[-1].rsplit('.', 1)[0])
    if not os.path.exists(save_loss_dir):
        os.makedirs(save_loss_dir)
    
    args.data_dir = os.path.join(Constants.asr_prediction_dir, 'human_transcribed','train_%s' % args.mode)


    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=False,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    )
    # a= 'i want to'
    # data = tokenizer(a)
    # data['input_id'] 

    fast_tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=True,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    )
    
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters: %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    if args.mode == 'token':
        # split train file and check check
        split_file_num = 10
        original_train_file = os.path.join(args.data_dir, args.train_file)
        split_train_file_last = os.path.join(args.data_dir, '%s_%d'% (args.train_file, split_file_num-1))
        model_name = args.model_name_or_path.split('/')[-1]

        if not os.path.exists(split_train_file_last):
            with open(original_train_file) as f:
                data = json.load(f)['data']
            print('Splitting train file. Total Item %d' % len(data))
            cur_item_count=0
            split_count = math.ceil(len(data) / split_file_num)
            for i in tqdm(range(split_file_num),desc="Splitting train file"):
                cur_data = []
                j = 0
                while j < split_count and cur_item_count<len(data):
                    cur_data.append(data[cur_item_count])
                    cur_item_count+= 1
                    j +=1
                split_train_file = os.path.join(args.data_dir, '%s_%d'% (args.train_file, i))
                with open(split_train_file, 'w') as f:
                    ujson.dump({'data': cur_data},  f)
                    #json.dump({'data': cur_data},  f)
                print('Splited part %d, item num %d' % (i, j))
        
        # selected_split_num = [0,1,2,3,4,5,6,7,8,9]
        selected_split_num = [6,7,8,9]

        old_train_file =args.train_file
        for i in selected_split_num:
            args.train_file =  os.path.join(args.data_dir, '%s_%d'% (old_train_file, i))
            output_file = "%s/%s_qa_loss_%s_%d.json" % (save_loss_dir, model_name, args.mode,i)
            if os.path.exists(output_file):
                print('Skip Splited train file %s' % args.train_file)
            else:
                print('Processing Splited train file %s' % args.train_file)
            
                train_dataset, all_qas_id = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
                loss_dict = qa_loss(args, train_dataset, model, tokenizer, all_qas_id)

                with open(output_file, 'w') as f:
                    json.dump(loss_dict ,f)
                    logger.info(f"Successfully wrote content in {output_file}")
    elif args.mode == 'question':
        model_name = args.model_name_or_path.split('/')[-1] #
        output_file = "%s/%s_qa_loss_%s.json" % (save_loss_dir, model_name, args.mode)
        train_dataset, all_qas_id = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        loss_dict = qa_loss(args, train_dataset, model, tokenizer, all_qas_id)

        with open(output_file, 'w') as f:
            json.dump(loss_dict ,f)
            logger.info(f"Successfully wrote content in {output_file}")

    else:
        logger.error(f"err mode:{args.mode}")
if __name__ == "__main__":
    main()

