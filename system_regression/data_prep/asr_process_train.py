#Note: 使用asr模型处理training data

"""
compute the NFR between new model and old model
"""

import os
import argparse

import torch
import json
from copy import deepcopy

from system_regression.asr.models.asr_ensemble import ASREnsembleModel, s2t_gen_sample
from system_regression.asr.models.rag_model import S2TRagEnsembleModel
from tqdm import tqdm
from system_regression.asr.utils import get_ref_ids, fill_ref_data, wer_normalize_text, gen_md5_hashmap, filter_eos_token, load_model,set_seed
from system_regression.data_prep.heysquad import get_heysquad_datasets

from torch.utils.data import DataLoader
from system_regression.common.const import Constants
from system_regression.asr.test_model import test
import torch.nn.functional as F
from  copy import deepcopy
import numpy as np

from loguru import logger
from system_regression.common.str2bool import str2bool

def parser_option():

    parser = argparse.ArgumentParser('argument for training')

    # model1
    parser.add_argument('--model', type=str, default='s2t',
                        choices=['s2t', 'wav2vec', 'whisper'])

    parser.add_argument('--model_path', type=str, help='old model snapshot')

    parser.add_argument('--dataset', type=str, default='heysquad_human', choices=['heysquad_human', 'heysquad_machine'], help='dataset')

    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--mode', type=str, choices=['question', 'token', 'cascade'])

    opt = parser.parse_args()

    return opt

#TODO: 实现pos level的 fill
def fill_ref_data_pos(ref_data, trans_res):
    missing_count = 0
    no_answer_count = 0
    total_count =0
    empty_question_count =0
    #remove not found trans
    for d in ref_data['data']:
        for pag in d['paragraphs']:
            new_qas = []
            for qa in pag['qas']:
                #for squad 2.0, there are many empty answers but plasusible answers without true answer
                total_count +=1
                if qa['id'] in trans_res:
                    if len(qa['answers']) == 0:
                        no_answer_count += 1
                    else:
                        for pos, pos_v in trans_res[qa['id']].items():
                            for pred_i, pred_v in pos_v.items():
                                cur_qa = deepcopy(qa)
                                if len(pred_v)<2:
                                    cur_qa['question'] ='empty'
                                    empty_question_count +=1
                                else:
                                    cur_qa['question'] = pred_v['question']
                                cur_qa['pos'] = pos
                                cur_qa['pred_weight'] = pred_v['weight']
                                weight_str=  format(pred_v['weight'], '.5f')
                                #新的id是组合了question id与position的
                                cur_qa['id'] = '%s_%d_%d_%s' % (cur_qa['id'], pos, pred_i,weight_str)
                                # skip plausible answer
                                # qa['answers'] = qa['plausible_answers']
                                new_qas.append(cur_qa)
                else:
                    missing_count +=1
                    #print('%s not in trans_res' % qa['id'])
                    #raise Exception()
            pag['qas'] = new_qas
    print('Ref count %d, Trans count %d, Missing count %d, No Answer count %d, Empty Question count %d' % (total_count, len(trans_res), missing_count, no_answer_count, empty_question_count))
    return

def load_pos_data(output_file):
    with open(output_file) as f:
        data = json.load(f)
    res = {}
    for d in data['data']:
        for pag in d['paragraphs']:
            for qa in pag['qas']:
                i = qa['id'].split('_')[0]
                if i not in res:
                    res[i] = {}
                res[i][qa['pos']] = qa['question']
    return res

def process_train_question_level(model_name, model_path, train_ref_ids,train_ref_hashmap, human_machine, opt):
    model, processor, data_collator = load_model(model_name, model_path)
    model = model.to(Constants.device)
    model.eval()
  
    train_set, _, test_set, test_val_set = get_heysquad_datasets(processor, model=model_name, ref_train_ids=train_ref_ids, ref_dev_ids=None, debugging=False, test_only=False, human_machine=human_machine, ref_train_hashmap=train_ref_hashmap, ref_dev_hashmap=None)
    train_loader = DataLoader(train_set,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                collate_fn=data_collator)
    _,res = test(model_name, model, train_loader, processor)
    filtered_res = {}
    filter_count = 0
    for key, value in res.items():
        if len(value['transcribe']) > 1:
            filtered_res[key] = value
        else:
            filter_count += 1
    #check empty res
    print('Filter %d empty  question' % filter_count)
    return filtered_res

#只替换当前的一个token
#TODO: 可能考虑将当前每次生成token时的概率记录下来，作为权重参数乘到qa loss上
def process_train_token_level_single(model_name, model_path, train_ref_ids,train_ref_hashmap, human_machine, opt, output_file,train_data, special_token_map=None):
    # special_token_map= None
    #TODO: 添加whisper的special token map计算
    # if 's2t' not in model_path_names[i]:
    #     special_token_map = load_special_token(os.path.join(pretrained_path, model_path_names[i]))
    pred_token_num = 5

    model, processor, data_collator = load_model(model_name, model_path)
    model = model.to(Constants.device)
    model.eval()
    set_seed(42)
    res = {} #{qid:{pos:transcibed_asr}}
    logging = {}#{(qid,pos):{'pred_token': xx, 'gt_token':xx}}
  
    train_set, _, test_set, test_val_set = get_heysquad_datasets(processor, model=model_name, ref_train_ids=train_ref_ids, ref_dev_ids=None, debugging=False, test_only=False, human_machine=human_machine, ref_train_hashmap=train_ref_hashmap, ref_dev_hashmap=None, train_shuffle=False)
    train_loader = DataLoader(train_set,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                collate_fn=data_collator)
    tokenizer=processor.tokenizer
    cur_batch_count=-1
    #TODO: cache机制取消后看是否会导致少搞训练数据
    # if os.path.exists(output_file):
    #     print('Load Cache')
    #     res = load_pos_data(output_file)
    # else:
    #     print('No Cache')

    cache_batch_count = len(res)/opt.batch_size

    for data in tqdm(train_loader):
        cur_batch_count += 1
        # if cur_batch_count < cache_batch_count:
        #     continue

        batch_size = data["input_features"].shape[0]
        input_features = data["input_features"].to(Constants.device)
        attention_mask  = data['attention_mask'].to(Constants.device) if 'attention_mask' in data  else None
        #For whisper model
        data['labels'][data['labels'] == -100] = processor.tokenizer.pad_token_id
        data['labels'] =  data['labels'].to(Constants.device)
        labels = data['labels']
        question_ids = data['ids']
        # which is best?
        # what is best
        # which are best
        # which is good

        # what is bese best best
        if  's2t' in model_name:
            labels =labels[:, 1:]
            shifted_input_ids = labels.new_zeros(labels.shape)
            shifted_input_ids[:, 1:] = labels[:, :-1].clone()
            shifted_input_ids[:, 0] = tokenizer.eos_token_id
            labels =shifted_input_ids
            start_idx = 1
        else:
            #TODO: 可能需要shift labels
            shifted_input_ids = labels.new_zeros(labels.shape)
            shifted_input_ids[:, 1:] = labels[:, :-1].clone()
            shifted_input_ids[:, 0] = model.generation_config.decoder_start_token_id
            labels =shifted_input_ids
            #TODO: 添加special token map
            start_idx = 1
            while labels[0][start_idx].cpu().item() in special_token_map:
                start_idx  += 1
        cur_decoder_input_ids = labels[:, :start_idx]
        valid_lenths = labels.ne(tokenizer.pad_token_id).sum(-1)
        max_valid_length = torch.max(valid_lenths).cpu().item()
                

        for position in range(start_idx,max_valid_length):
            with torch.no_grad():
                if 's2t' in model_name:
                    outputs = model(input_features=input_features, attention_mask=attention_mask,  decoder_input_ids=cur_decoder_input_ids)
                else:
                    outputs = model(input_features=input_features,
                    decoder_input_ids=cur_decoder_input_ids)
            #logits = outputs.logits.view(-1, outputs.logits.size(-1))
            logits = outputs.logits[:, -1, :]
            
            # probs = F.softmax(logits, dim=-1)
            # _, pred = torch.max(probs, dim=-1)

            top_k_logits, top_k_indices = torch.topk(logits, pred_token_num, dim=1)
            top_k_probs =  F.softmax(top_k_logits, dim=-1)
            # ground_truth = int(inputs_1d[position].item())
            pred_input_id = deepcopy(cur_decoder_input_ids)

            #TODO: 在这里循环pred
            for pred_i in range(pred_token_num):
                pred = top_k_indices[:, pred_i]
                pred_input_id = torch.cat([cur_decoder_input_ids, torch.reshape(pred, (batch_size,1))], dim=-1)
                if position < max_valid_length-1:
                    pred_input_id = torch.cat([pred_input_id, torch.reshape(labels[:, position+1:], (batch_size,-1))], dim=-1)
                for batch_idx in  range(batch_size):
                    if position < valid_lenths[batch_idx]: 
                        pred_question=tokenizer.decode(pred_input_id[batch_idx],skip_special_tokens=True)
                        if question_ids[batch_idx] not in res:
                            res[question_ids[batch_idx]] = {}
                        if position not in res[question_ids[batch_idx]]:
                            res[question_ids[batch_idx]][position] = {}

                        res[question_ids[batch_idx]][position][pred_i] = {
                            'question': wer_normalize_text([pred_question])[0],
                            'weight': top_k_probs[batch_idx][pred_i].cpu().item()
                        } 
                 
                        # logging[(question_ids[batch_idx], position-1)] = [pred_input_id[batch_idx][position].cpu().item(), labels[batch_idx][position].cpu().item()]
            cur_decoder_input_ids = torch.cat([cur_decoder_input_ids, torch.reshape(labels[:, position], (batch_size,1))], dim=-1)

        if cur_batch_count % 50 == 0:
            train_trans = deepcopy(train_data)
            fill_ref_data_pos(train_trans, res)
            with open(output_file, 'w') as f:
                json.dump(train_trans, f)
    
    train_trans = deepcopy(train_data)
    fill_ref_data_pos(train_trans, res)
    with open(output_file, 'w') as f:
        json.dump(train_trans, f)
    print('Total processed train item in token level: %d' % len(res))
    return res


#替换当前的token, 以及后续token也使用模型生成
def process_train_token_level_cascade(model_name, model_path, train_ref_ids,train_ref_hashmap, human_machine, special_token_map, opt):
    pass

#TODO: 添加一个process training data question level还是token level的区别，token level的数据
def main():
    opt = parser_option()
    logger.info(opt)
    if opt.dataset.startswith('heysquad'):
        # print("loading dataset...")
        human_machine = 'human'
        ref_dev_file = os.path.join(Constants.heysquad_json_dir, 'dev-common-original-1002.json')
        if 'machine' in opt.dataset:
            human_machine = 'machine'
            ref_dev_file = os.path.join(Constants.heysquad_json_dir, 'dev-v1.1.json')
        
        ref_train_file = os.path.join(Constants.heysquad_json_dir,'train-common-original-48849.json')
        with open(ref_train_file, 'r') as f:
            train_data = json.load(f)
        train_ref_ids = get_ref_ids(train_data)
        train_ref_hashmap = gen_md5_hashmap(train_data)

        # with open(ref_dev_file, 'r') as f:
        #     dev_data = json.load(f)
        # dev_ref_ids = get_ref_ids(dev_data)
        # dev_ref_hashmap = gen_md5_hashmap(dev_data)

    else:
        raise NotImplementedError(opt.dataset)
    
    
    predict_dir = "%s/%s_transcribed/train_%s" %(Constants.asr_prediction_dir, human_machine, opt.mode)
    if not os.path.exists(predict_dir):
        os.mkdir(predict_dir)

    output_file = "%s/%s.json" %(predict_dir, opt.model_path)
    model_path = os.path.join(Constants.pretrained_model_dir, opt.model_path)

    
    if opt.mode == 'question':
        print('processing the training data at question level:')
        res = process_train_question_level(opt.model, model_path, train_ref_ids=train_ref_ids, train_ref_hashmap=train_ref_hashmap, human_machine=human_machine, opt=opt)
        train_trans = deepcopy(train_data)
        fill_ref_data(train_trans, res)
        with open(output_file, 'w') as f:
            json.dump(train_trans, f)
        logger.info(f"Successfully wrote content to {output_file}")
    elif opt.mode == 'token':
        print('processing the training data at token single level:')
        res = process_train_token_level_single(opt.model, model_path, train_ref_ids=train_ref_ids, train_ref_hashmap=train_ref_hashmap, human_machine=human_machine, output_file=output_file,train_data=train_data, opt=opt)
        train_trans = deepcopy(train_data)
        fill_ref_data_pos(train_trans, res)
        with open(output_file, 'w') as f:
            json.dump(train_trans, f)
        logger.info(f"Successfully wrote content to {output_file}")
    elif opt.mode == 'cascade':
        print('processing the training data at token cascade level:')
        pass
    else:
        logger.error(f'err mode:{opt.mode}')
        # print('err mode')



if __name__ == '__main__':
    main()