# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from copy import deepcopy, copy
import os
import pickle
import time

import pandas as pd
import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from transformers import Speech2TextForConditionalGeneration, WhisperForConditionalGeneration, PreTrainedModel,Speech2TextPreTrainedModel, Speech2TextConfig,  WhisperConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
import os
from scipy.stats import norm
import json
from system_regression.common.const import Constants
from system_regression.asr.models.model_utils import *

def filter_eos_token(generate_ids, eos_token_id):
    batch_size, seq_length = generate_ids.size()
    for i in range(batch_size):
        # Find the position of the first eos_token_id in the row
        eos_positions = (generate_ids[i] == eos_token_id).nonzero(as_tuple=True)[0]
        
        if eos_positions.numel() >= 2:
            second_eos_pos = eos_positions[1].item()
            # Set all token IDs after the second eos_token_id to eos_token_id
            generate_ids[i, second_eos_pos + 1:] = eos_token_id
    return generate_ids


def count_seq_end(generate_ids, eos_token_id):
    # Create a boolean mask where elements equal to eos_token_id are True
    eos_mask = (generate_ids == eos_token_id)
    
    # Sum the boolean mask along the seq_length dimension to count eos_token_id occurrences in each row
    eos_count_per_row = eos_mask.sum(dim=1)
    
    # Count the rows where the sum is greater than 2
    rows_with_more_than_two_eos = (eos_count_per_row >= 2).sum().item()
    return rows_with_more_than_two_eos

# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class S2TRagEnsembleModel(nn.Module):
    
    def __init__(self, model_name, model1, model2, processor, save_inter_file=None, logging=False, mode='asr'):
        super(S2TRagEnsembleModel, self).__init__()
        self.model_name = model_name
        self.model1 = model1
        self.model2 = model2
        self.processor = processor
        self.save_inter_file = save_inter_file #将中途生成每个token对应的old model: {old_token_id:prob, }, new model: {}, ensemble:  {}
        self.logging = logging
        self.mode = mode #asr, question, token, cascade
    
    # for asr knn cache
    def load_index(self,old_knn_dir, knn_dir, asr_lambda=1.0, pt_lambda=1.0,pt_step=10, knn_neighbors_num=20, knn_rag_weight=0.0):
        knn_infos_path = os.path.join(knn_dir,'knn_infos.pkl')
        index_save_path = os.path.join(knn_dir,'knn.index')
        index_save_path_old = os.path.join(old_knn_dir,'knn.index')
        vecs_path = os.path.join(knn_dir,'vecs.pkl')
        self.knn_neighbors_num = knn_neighbors_num
        self.asr_lambda = asr_lambda
        self.knn_rag_weight = knn_rag_weight #用来控制使用knn_rag的比例参数

        with open(vecs_path, 'rb') as f:
            self.vecs = pickle.load(f)
        with open(knn_infos_path, 'rb') as f:
            self.knn_infos = pickle.load(f)
        self.index = faiss.read_index(index_save_path)
        self.index_old = faiss.read_index(index_save_path_old)
        assert self.index.is_trained

        #load old knn info
        knn_infos_path = os.path.join(old_knn_dir,'knn_infos.pkl')
        with open(knn_infos_path, 'rb') as f:
            # train_id, position, groud_truth, pred_id
            self.old_knn_infos=pickle.load(f)
            self.old_knn_losses =self.old_knn_infos['asr_losses']
        #loss diff越大，cdf计算出的loss_diff_weight越大，给老模型的权重越大
        self.loss_diff = self.knn_infos['asr_losses'] - self.old_knn_losses
        self.loss_diff_mean = np.mean(self.loss_diff)
        self.loss_diff_std = np.std(self.loss_diff)

    
    #for qa loss 
    #qa_asr_weight代表qa loss算出来的logit ensemble与asr_loss算出来的logit ensemble的权重
    #qa_lambda与knn_lambda一样，是调整的weight
    def load_qa_loss(self, old_qa_loss_file, new_qa_loss_file, qa_lambda=1.0, qa_asr_weight=0.5):
        def get_loss_item(loss_dict, key, default, loss_name):
            if key in loss_dict:
                return loss_dict[key][loss_name]
            else:
                return default
        
        def merge_pred_qa_loss(loss_file):

            new_loss_dict= {} # {'id': }
            split_num = 10
            loss_file = loss_file.split('.')[0]
            for i in range(split_num):
                cur_split_file = '%s_%d.json' % (loss_file, i)
                if os.path.exists(cur_split_file):
                    print('Loading from %s' % cur_split_file)
                    with open(cur_split_file) as f:
                        loss_dict = json.load(f)
                    for key, value in loss_dict.items():
                        i = '_'.join(key.split('_')[:2])
                        # print(key)#
                        cur_weight = float(key.split('_')[3])
                        if i not in new_loss_dict:
                            new_loss_dict[i] = {
                                'start_loss_list': [],
                                'end_loss_list': [],
                                'weight_list': []
                            }
                        new_loss_dict[i]['start_loss_list'].append(value['start_loss'])
                        new_loss_dict[i]['end_loss_list'].append(value['end_loss'])
                        new_loss_dict[i]['weight_list'].append(cur_weight)
                else:
                    print('Not found %s' % cur_split_file)
            
            for key, value in new_loss_dict.items():
                new_loss_dict[key]['start_loss'] = np.mean(new_loss_dict[key]['start_loss_list'])
                new_loss_dict[key]['end_loss'] = np.mean(new_loss_dict[key]['end_loss_list'])
                new_loss_dict[key]['total_loss'] = (new_loss_dict[key]['start_loss'] + new_loss_dict[key]['end_loss'])/2
                #TODO: 移除start_loss_list与end_loss_list以节约内存
            return new_loss_dict

        self.qa_lambda = qa_lambda
        self.qa_asr_weight = qa_asr_weight
        if self.mode == 'question' or self.mode == 'question-merge' or self.mode == 'asr' or self.mode =='simple-asr' or self.mode=='simple-question':
            with open(old_qa_loss_file) as f:
                self.old_qa_loss = json.load(f)
            with open(new_qa_loss_file) as f:
                self.new_qa_loss = json.load(f)
        else:
            self.old_qa_loss = merge_pred_qa_loss(old_qa_loss_file)
            self.new_qa_loss = merge_pred_qa_loss(new_qa_loss_file)

        start_loss_diff = {}
        end_loss_diff = {}
        total_loss_diff = {}
        max_qa_loss_diff ={}
        #TODO: 可能要在这里一下根据pred num进行加权平均，合并一下pred的loss，还是构建一个token为key的loss dict


        max_old_total_loss = max(self.old_qa_loss[key]['total_loss'] for key in  self.old_qa_loss)
        max_old_start_loss = max(self.old_qa_loss[key]['start_loss'] for key in  self.old_qa_loss)
        max_old_end_loss = max(self.old_qa_loss[key]['end_loss'] for key in  self.old_qa_loss)
        self.max_old_start_loss=max_old_start_loss
        self.max_old_end_loss=max_old_end_loss

        max_new_total_loss = max(self.new_qa_loss[key]['total_loss'] for key in  self.new_qa_loss)
        max_new_start_loss = max(self.new_qa_loss[key]['start_loss'] for key in  self.new_qa_loss)
        max_new_end_loss = max(self.new_qa_loss[key]['end_loss'] for key in  self.new_qa_loss)
        self.max_new_start_loss=max_new_start_loss
        self.max_new_end_loss=max_new_end_loss


        key_set = set(self.old_qa_loss.keys())
        key_set =  key_set.union(set(self.new_qa_loss.keys()))
        for key in  key_set:
            start_loss_diff[key] =  get_loss_item(self.new_qa_loss, key, max_new_start_loss, 'start_loss') - get_loss_item(self.old_qa_loss, key, max_old_start_loss, 'start_loss')
            end_loss_diff[key] = get_loss_item(self.new_qa_loss, key, max_new_end_loss, 'end_loss') - get_loss_item(self.old_qa_loss, key, max_old_end_loss, 'end_loss')
            total_loss_diff[key] = get_loss_item(self.new_qa_loss, key, max_new_total_loss, 'total_loss') - get_loss_item(self.old_qa_loss, key, max_old_total_loss, 'total_loss')
            max_qa_loss_diff[key]=  np.maximum(start_loss_diff[key],end_loss_diff[key]) #计算两者的max
       
        self.start_qa_loss_diff = start_loss_diff
        self.end_qa_loss_diff = end_loss_diff
        self.max_qa_loss_diff = max_qa_loss_diff
        self.start_qa_loss_diff_metrics = [np.mean(list(self.start_qa_loss_diff.values())), np.std(list(self.start_qa_loss_diff.values()))]
        self.end_qa_loss_diff_metrics = [np.mean(list(self.end_qa_loss_diff.values())), np.std(list(self.end_qa_loss_diff.values()))]
        self.max_qa_loss_diff_metrics = [np.mean(list(self.max_qa_loss_diff.values())), np.std(list(self.max_qa_loss_diff.values()))]

    
    def get_loss_diff_weight_asr(self, cur_loss_diff, distance_weight, qa_diff_weight=None):
        #one sigma shift to increase weight for old model

        
        #0,0.1,0.3,0.5,0.7
        if qa_diff_weight is None:
            loss_diff_weight = norm.cdf(cur_loss_diff, loc=self.loss_diff_mean-self.asr_lambda*self.loss_diff_std, scale=self.loss_diff_std)
        else:
            loc_base_ratio = - self.asr_lambda
            #qa_diff_weight shift range from (-qa_asr_weight*loss_diff_std, 0), 不能往右移动
            # qa_asr_weight: 0-2
            qa_shift_loc_ratio = 2*(0.5-qa_diff_weight) * self.qa_asr_weight
            if qa_shift_loc_ratio > 0:
                qa_shift_loc_ratio = 0

            loc_shift = (loc_base_ratio + qa_shift_loc_ratio)*self.loss_diff_std

            loss_diff_weight = norm.cdf(cur_loss_diff, loc=self.loss_diff_mean + loc_shift, scale=self.loss_diff_std)

        return np.sum(np.multiply(loss_diff_weight, distance_weight))
    
    def get_loss_diff_weight_qa(self, start_loss_diff, end_loss_diff, distance_weight):
        
        start_loss_diff_mean=  self.start_qa_loss_diff_metrics[0]
        start_loss_diff_std=  self.start_qa_loss_diff_metrics[1]
        end_loss_diff_mean=  self.end_qa_loss_diff_metrics[0]
        end_loss_diff_std= self.end_qa_loss_diff_metrics[1]
        max_loss_diff_mean=  self.max_qa_loss_diff_metrics[0]
        max_loss_diff_std= self.max_qa_loss_diff_metrics[1]
        
        # max_loss_diff = np.maximum(start_loss_diff, end_loss_diff)
        # max_loss_diff_weight  = norm.cdf(max_loss_diff, loc=max_loss_diff_mean-self.qa_lambda*max_loss_diff_std, scale=max_loss_diff_std)
        # return np.sum(np.multiply(max_loss_diff_weight, distance_weight))
        
        #qa_lambda: [-2, 2]
        if self.mode == 'question' or self.mode == 'token' or self.mode == 'asr': 
            start_loss_diff_weight = norm.cdf(start_loss_diff, loc=start_loss_diff_mean-self.qa_lambda*start_loss_diff_std, scale=start_loss_diff_std)
            end_loss_diff_weight = norm.cdf(end_loss_diff, loc=end_loss_diff_mean-self.qa_lambda*end_loss_diff_std, scale=end_loss_diff_std)
        else:
            start_loss_diff_weight = norm.cdf(start_loss_diff, loc=start_loss_diff_mean-self.qa_lambda*start_loss_diff_std, scale=start_loss_diff_std)
            end_loss_diff_weight = norm.cdf(end_loss_diff, loc=end_loss_diff_mean-self.qa_lambda*end_loss_diff_std, scale=end_loss_diff_std)

        # total_loss_diff_weight  = (start_loss_diff_weight + end_loss_diff_weight)/2
        total_loss_diff_weight  = np.maximum(start_loss_diff_weight, end_loss_diff_weight)
        return np.sum(np.multiply(total_loss_diff_weight, distance_weight))
    
    def update_logits_knn_test1(self, last_hidden_states_old,last_hidden_states, old_logits, logits):
        batch_size = logits.shape[0]
        logits_final = deepcopy(logits)
        vecs_old = last_hidden_states_old.cpu().numpy()
        vecs = last_hidden_states.cpu().numpy()
        old_probs = F.softmax(old_logits, dim=-1)
        _, old_pred =  torch.max(old_probs, dim=-1)
        new_probs = F.softmax(logits, dim=-1)
        _, new_pred =  torch.max(new_probs, dim=-1)

        # faiss.normalize_L2(vecs)
        # faiss.normalize_L2(vecs_old)

        #如果新旧模型都没有要结束，就过滤掉neighbor中结束的token
        distances_old, neighbors_ids_old = self.index_old.search(vecs_old, 50*self.knn_neighbors_num)
        distances, neighbors_ids = self.index.search(vecs, 50*self.knn_neighbors_num)




        res=[]
        new_distances= [] #batch_size, neighbor
        new_neighbor_ids=[]
        for batch_num in range(batch_size):
            if old_pred[batch_num]!=self.processor.tokenizer.eos_token_id and new_pred[batch_num]!=self.processor.tokenizer.eos_token_id:
                neighbors_id = neighbors_ids[batch_num]
                distance = distances[batch_num]
                neighbor_gts = self.knn_infos['gt_token_ids'][neighbors_id]
                cur_neighbor_ids = []
                cur_distance = []
                for i, token in enumerate(neighbor_gts):
                    if token != self.processor.tokenizer.eos_token_id:
                        cur_neighbor_ids.append(neighbors_id[i])
                        cur_distance.append(distance[i])
                    if len(cur_neighbor_ids) == self.knn_neighbors_num:
                        break
                if len(cur_neighbor_ids) <self.knn_neighbors_num:
                    print('warning: not enough non terminate neighbors: %d' % len(cur_neighbor_ids))
                    # new_neighbor_ids.append(neighbors_ids[batch_num][:self.knn_neighbors_num])
                    # new_distances.append(distances[batch_num][:self.knn_neighbors_num])
                    for i in range(self.knn_neighbors_num-len(cur_neighbor_ids)):
                        cur_neighbor_ids.append(neighbors_id[i])
                        cur_distance.append(distance[i])
                
                new_neighbor_ids.append(cur_neighbor_ids)
                new_distances.append(cur_distance)
            else:
                new_neighbor_ids.append(neighbors_ids[batch_num][:self.knn_neighbors_num])
                new_distances.append(distances[batch_num][:self.knn_neighbors_num])
        distances = np.array(new_distances)
        neighbors_ids = np.array(new_neighbor_ids)
    
        for batch_num in range(batch_size):
            distance = distances[batch_num]
            neighbors_id = neighbors_ids[batch_num]

            distance_old = distances_old[batch_num]
            neighbors_id_old = neighbors_ids_old[batch_num]

            # def curve(input):
            #     return [2 / (1 + np.exp(-5 * x)) - 1 for x in input]

            # distance=curve(distance)
            epsilon = 1e-8

            weights = -(distance  - np.mean(distance))/(np.std(distance)+epsilon)
            weights = np.exp(weights) / np.exp(weights).sum(axis=-1, keepdims=True)

            a=np.std(distance_old)+epsilon
            weights_old = -(distance_old  - np.mean(distance_old))/(np.std(distance_old)+epsilon)
            weights_old = np.exp(weights_old) / np.exp(weights_old).sum(axis=-1, keepdims=True)


            
            loss_new=sum([weights[i]*self.knn_infos['asr_losses'][id] for i,id in enumerate(neighbors_id)])
            loss_old=sum([weights_old[i]*self.old_knn_infos['asr_losses'][id] for i,id in enumerate(neighbors_id_old)])
            loss_diff=loss_new-loss_old

            # knn搜索邻居部分
            neighbors_vecs = self.vecs[neighbors_id]
            # tem_weights=weights.unsqueeze(0)
            neighbors_vecs2 = deepcopy(neighbors_vecs)
            faiss.normalize_L2(neighbors_vecs2)
            vecs2=deepcopy(vecs)
            faiss.normalize_L2(vecs2)
            cosine_distance=sum(np.multiply(neighbors_vecs2[0],vecs2[batch_num]))

            neighbor_gts = self.knn_infos['gt_token_ids'][neighbors_id]
            neight_loss_diff = self.loss_diff[neighbors_id]

            neighbor_gts_old = self.old_knn_infos['gt_token_ids'][neighbors_id_old]
            neight_loss_diff_old = self.loss_diff[neighbors_id_old]

            neight_question_id = [self.knn_infos['question_ids'][i] for i in neighbors_id]



            logits_x = logits[batch_num].squeeze(0)
            old_logits_x = old_logits[batch_num].squeeze(0)

            least_distance=distance[0]
            distance_thresh=0.93


            least_distance_old=distance_old[0]
            distance_thresh_old=0.5

            # # 权重和最大
            # hashtable={}
            # tem_token=neighbor_gts[0]
            # tem_weight=weights[0]
            # for i,weight in enumerate(weights):
            #     token_i=neighbor_gts[i]
            #     if token_i not in hashtable:
            #         hashtable[token_i]=weight
            #     else:
            #         hashtable[token_i]+=weight
            #     if hashtable[token_i]>tem_weight:
            #         tem_token=token_i
            #         tem_weight=hashtable[token_i]

            # # 权重和最大old
            # hashtable_old={}
            # tem_token_old=neighbor_gts_old[0]
            # tem_weight=weights_old[0]
            # for i,weight in enumerate(weights_old):
            #     token_i=neighbor_gts_old[i]
            #     if token_i not in hashtable_old:
            #         hashtable_old[token_i]=weight
            #     else:
            #         hashtable_old[token_i]+=weight
            #     if hashtable_old[token_i]>tem_weight:
            #         tem_token_old=token_i
            #         tem_weight=hashtable_old[token_i]

            # tem_token=new_pred[batch_num]
            # # 众数
            # hashtable={}
            # tem_token=neighbor_gts[0]
            # tem_weight=1
            # for i,weight in enumerate(weights):
            #     token_i=neighbor_gts[i]
            #     if token_i not in hashtable:
            #         hashtable[token_i]=1
            #     else:
            #         hashtable[token_i]+=1
            #     if hashtable[token_i]>tem_weight:
            #         tem_token=token_i
            #         tem_weight=hashtable[token_i]
            
            # token_choosed=0
            # if least_distance_old>distance_thresh:
            #     token_choosed=tem_token_old
            #     # if least_distance_old<2.5:
            #     #     token_choosed=tem_token_old
            # else:
            #     # self.asr_lambda=0.0
            #     # asr_diff_weight_before = self.get_loss_diff_weight_asr(neight_loss_diff, weights)
            #     # _,token_choosed=torch.max(old_logits_x *asr_diff_weight_before + (1-asr_diff_weight_before)*logits_x,dim=-1)
            #     # token_choosed=new_pred[batch_num]
            #     token_choosed=old_pred[batch_num]



            # alpha=0.6
            # _,token_choosed=torch.max(old_logits_x *alpha + (1-alpha)*logits_x,dim=-1) 
            # res.append(token_choosed)
            # self.asr_lambda=-1.5
            # asr_diff_weight_before = self.get_loss_diff_weight_asr(neight_loss_diff, weights)
            # _,token_choosed=torch.max(old_logits_x *asr_diff_weight_before + (1-asr_diff_weight_before)*logits_x,dim=-1)

                            # asr_diff_weight_before = self.get_loss_diff_weight_asr(neight_loss_diff, weights)
                # _,token_choosed=torch.max(old_logits_x *asr_diff_weight_before + (1-asr_diff_weight_before)*logits_x,dim=-1)

            # self.asr_lambda=0.4

            # old_logits_x=F.softmax(old_logits_x,dim=-1)
            # logits_x=F.softmax(logits_x,dim=-1) 

            # distance_nor=1.3
            # if least_distance>distance_nor:
            #     alpha=(least_distance-distance_nor)/(1-distance_nor)
            # else:
            #     alpha=(least_distance-distance_nor)/distance_nor

            # if least_distance_old>distance_nor:
            #     alpha_old=(least_distance_old-distance_nor)/(1-distance_nor)
            # else:
            #     alpha_old=(least_distance_old-distance_nor)/distance_nor
            
            # assert least_distance <1.1

            # alpha=alpha*10
            # alpha_old=alpha_old*10

            # if old_pred[batch_num]!=new_pred[batch_num]:
            #     logits_x[tem_token]=logits_x[tem_token]*alpha

            # old_logits_x[tem_token_old]=old_logits_x[tem_token_old]*alpha_old

            loss_diff_weight = norm.cdf(neight_loss_diff, loc=self.loss_diff_mean-self.asr_lambda*self.loss_diff_std, scale=self.loss_diff_std)
            asr_diff_weight_before = np.sum(np.multiply(loss_diff_weight, weights))

            # asr_diff_weight_before=norm.cdf(loss_diff, loc=self.loss_diff_mean-self.asr_lambda*self.loss_diff_std, scale=self.loss_diff_std)
            # asr_diff_weight_before = norm.cdf(loss_diff, loc=self.loss_diff_mean-self.asr_lambda*self.loss_diff_std, scale=self.loss_diff_std)
            final_logit=old_logits_x *asr_diff_weight_before + (1-asr_diff_weight_before)*logits_x
            # _,token_choosed=torch.max(old_logits_x *asr_diff_weight_before + (1-asr_diff_weight_before)*logits_x,dim=-1)
            #根据logits和old logits的pred token id, 判断其中一个有终止符的时候再打开knn rag ensemble开关

            #TODO: 这里想要解除判定限制的话，需要衡量knn检索出的内容是否靠谱，例如首个token knn neighbor直接返回终止符的情况就不太靠谱
            if old_pred[batch_num]==self.processor.tokenizer.eos_token_id or new_pred[batch_num]==self.processor.tokenizer.eos_token_id:
                logits_x = final_logit
                probs_x = F.softmax(logits_x, dim=-1)
                probs_knn = torch.zeros_like(probs_x)
                probs_x = probs_x * (1 - self.knn_rag_weight)
                for i in range(len(neighbors_id)):
                    probs_knn[neighbor_gts[i]] += weights[i]
                probs_x += self.knn_rag_weight * probs_knn
                final_logit = torch.log(probs_x)

            # elif cosine_distance>0.95 and self.asr_lambda<0:
            #     neighbor_rag_weight=self.asr_lambda*(-0.5)
            #     logits_x = final_logit
            #     probs_x = F.softmax(logits_x, dim=-1)
            #     probs_knn = torch.zeros_like(probs_x)
            #     probs_x = probs_x * (1 - neighbor_rag_weight)
            #     for i in range(len(neighbors_id)):
            #         probs_knn[neighbor_gts[i]] += weights[i]
            #     probs_x += neighbor_rag_weight * probs_knn
            #     final_logit = torch.log(probs_x)

            elif cosine_distance>0.95:
                neighbor_rag_weight=0.7
                logits_x = final_logit
                probs_x = F.softmax(logits_x, dim=-1)
                probs_knn = torch.zeros_like(probs_x)
                probs_x = probs_x * (1 - neighbor_rag_weight)
                for i in range(len(neighbors_id)):
                    probs_knn[neighbor_gts[i]] += weights[i]
                probs_x += neighbor_rag_weight * probs_knn
                final_logit = torch.log(probs_x)
            _,token_choosed=torch.max(final_logit,dim=-1)

            res.append(token_choosed)

        return res

    #TODO:增加一个按照是否成功预测token_id算的，这个需要create knn的时候除了loss记录一下model预测的token_id。或许我们应该只考虑新模型预测token_id, 旧模型预测token_id,  ground truth token id三个单词分布的loss.
    
    #TODO: 添加考虑qa loss的部分，这部分如果相应的neighbor_id不在qa loss file中的话，说明question为空，则loss设置为最大
    def update_logits_incrementally_knn(self,last_hidden_states_old,last_hidden_states, old_logits, logits):
        batch_size = logits.shape[0]
        logits_final = deepcopy(logits)
        vecs_old = last_hidden_states_old.cpu().numpy()
        vecs = last_hidden_states.cpu().numpy()
        old_probs = F.softmax(old_logits, dim=-1)
        _, old_pred =  torch.max(old_probs, dim=-1)
        new_probs = F.softmax(logits, dim=-1)
        _, new_pred =  torch.max(new_probs, dim=-1)

        #如果新旧模型都没有要结束，就过滤掉neighbor中结束的token
        distances_old, neighbors_ids_old = self.index_old.search(vecs_old, 50*self.knn_neighbors_num)
        distances, neighbors_ids = self.index.search(vecs, 50*self.knn_neighbors_num)

        new_distances= [] #batch_size, neighbor
        new_neighbor_ids=[]

        old_distances=[]
        old_neighbor_ids=[]

        for batch_num in range(batch_size):
            if old_pred[batch_num]!=self.processor.tokenizer.eos_token_id and new_pred[batch_num]!=self.processor.tokenizer.eos_token_id:

                # process new 
                neighbors_id = neighbors_ids[batch_num]
                distance = distances[batch_num]
                neighbor_gts = self.knn_infos['gt_token_ids'][neighbors_id]
                cur_neighbor_ids = []
                cur_distance = []
                for i, token in enumerate(neighbor_gts):
                    if token != self.processor.tokenizer.eos_token_id:
                        cur_neighbor_ids.append(neighbors_id[i])
                        cur_distance.append(distance[i])
                    if len(cur_neighbor_ids) == self.knn_neighbors_num:
                        break
                if len(cur_neighbor_ids) <self.knn_neighbors_num:
                    print('warning: not enough non terminate neighbors: %d' % len(cur_neighbor_ids)) 
                    # new_neighbor_ids.append(neighbors_ids[batch_num][:self.knn_neighbors_num])
                    # new_distances.append(distances[batch_num][:self.knn_neighbors_num])
                    for i in range(self.knn_neighbors_num-len(cur_neighbor_ids)):
                        cur_neighbor_ids.append(neighbors_id[i])
                        cur_distance.append(distance[i])
                
                new_neighbor_ids.append(cur_neighbor_ids)
                new_distances.append(cur_distance)

                # process old
                neighbors_id_old = neighbors_ids_old[batch_num]
                distance_old = distances_old[batch_num]
                neighbor_gts_old = self.old_knn_infos['gt_token_ids'][neighbors_id_old]
                cur_neighbor_ids_old = []
                cur_distance_old = []
                for i, token in enumerate(neighbor_gts_old):
                    if token != self.processor.tokenizer.eos_token_id:
                        cur_neighbor_ids_old.append(neighbors_id_old[i])
                        cur_distance_old.append(distance_old[i])
                    if len(cur_neighbor_ids_old) == self.knn_neighbors_num:
                        break
                if len(cur_neighbor_ids_old) <self.knn_neighbors_num:
                    print('warning: not enough non terminate neighbors: %d' % len(cur_neighbor_ids_old)) 
                    # new_neighbor_ids.append(neighbors_ids[batch_num][:self.knn_neighbors_num])
                    # new_distances.append(distances[batch_num][:self.knn_neighbors_num])
                    for i in range(self.knn_neighbors_num-len(cur_neighbor_ids_old)):
                        cur_neighbor_ids_old.append(neighbors_id_old[i])
                        cur_distance_old.append(distance_old[i])
                
                old_neighbor_ids.append(cur_neighbor_ids_old)
                old_distances.append(cur_distance_old)
            else:
                new_neighbor_ids.append(neighbors_ids[batch_num][:self.knn_neighbors_num])
                new_distances.append(distances[batch_num][:self.knn_neighbors_num])

                old_neighbor_ids.append(neighbors_ids_old[batch_num][:self.knn_neighbors_num])
                old_distances.append(distances_old[batch_num][:self.knn_neighbors_num])

        distances = np.array(new_distances)
        neighbors_ids = np.array(new_neighbor_ids)
        
        distances_old = np.array(old_distances)
        neighbors_ids_old = np.array(old_neighbor_ids)

        neighbor_gts_full =  np.zeros((batch_size, self.knn_neighbors_num), dtype=np.int32)
        neighbor_weights = np.zeros((batch_size, self.knn_neighbors_num), dtype=np.float32)
        #only capture weighted sum loss now
        neighbor_new_loss =  np.zeros((batch_size, 1))
        neighbor_old_loss =  np.zeros((batch_size, 1))

       

        neighbors_log_data = []#(batch_size, )
        neighbors_log_data_old = []
        diff_weights=[]
        for batch_num in range(batch_size):
            distance = distances[batch_num]
            neighbors_id = neighbors_ids[batch_num]

            distance_old = distances_old[batch_num]
            neighbors_id_old = neighbors_ids_old[batch_num]

            # knn搜索邻居部分
            
            neighbors_vecs = self.vecs[neighbors_id]
            neighbors_vecs2 = deepcopy(neighbors_vecs)
            faiss.normalize_L2(neighbors_vecs2)
            vecs2=deepcopy(vecs)
            faiss.normalize_L2(vecs2)
            cosine_distance=sum(np.multiply(neighbors_vecs2[0],vecs2[batch_num]))

            # dist=[]
            # for i in range(20):
            #     neighbor_vec=neighbors_vecs[i]
            #     vec=vecs[batch_num]
            #     neighbor_vec = torch.from_numpy(neighbor_vec)
            #     vec = torch.from_numpy(vec)
            #     l2_distance = torch.norm(neighbor_vec - vec, p=2)
            #     dist.append(l2_distance)

            neighbor_gts = self.knn_infos['gt_token_ids'][neighbors_id]
            neight_loss_diff = self.loss_diff[neighbors_id]

            neighbor_gts_old = self.old_knn_infos['gt_token_ids'][neighbors_id_old]
            neight_loss_diff_old = self.loss_diff[neighbors_id_old]

            neight_new_loss =  self.knn_infos['asr_losses'][neighbors_id]
            neight_old_loss =  self.old_knn_losses[neighbors_id]

            neight_pos= self.knn_infos['poss'][neighbors_id]

            neight_question_id = [self.knn_infos['question_ids'][i] for i in neighbors_id]
            if self.mode =='token' or self.mode =='token-merge':
                neight_question_pos_id = []
                for q_id, pos in zip(neight_question_id, neight_pos):
                    neight_question_pos_id.append('%s_%s'%(q_id,pos))
                neight_question_id = neight_question_pos_id

            # normalize  weight by softmax
            # 10100， 10200
            # loss_diff = new_model_loss-old_model loss
            # new_model_loss/(new_model_loss-old model lss) 

            epsilon = 1e-8

            weights = -(distance  - np.mean(distance))/(np.std(distance)+epsilon)
            weights = np.exp(weights) / np.exp(weights).sum(axis=-1, keepdims=True)

            weights_old = -(distance_old  - np.mean(distance_old))/(np.std(distance_old)+epsilon)
            weights_old = np.exp(weights_old) / np.exp(weights_old).sum(axis=-1, keepdims=True)

            loss_new=sum([weights[i]*self.knn_infos['asr_losses'][id] for i,id in enumerate(neighbors_id)])
            loss_old=sum([weights_old[i]*self.old_knn_infos['asr_losses'][id] for i,id in enumerate(neighbors_id)])

            qa_start_loss_new=sum([weights[i]*self.new_qa_loss[id]['start_loss'] if id in self.new_qa_loss else weights[i]*self.max_new_start_loss for i,id in enumerate(neight_question_id) ])
            qa_start_loss_old=sum([weights[i]*self.old_qa_loss[id]['start_loss'] if id in self.old_qa_loss else weights[i]*self.max_old_start_loss for i,id in enumerate(neight_question_id) ])

            qa_end_loss_new=sum([weights[i]*self.new_qa_loss[id]['end_loss'] if id in self.new_qa_loss else weights[i]*self.max_new_end_loss for i,id in enumerate(neight_question_id)] )
            qa_end_loss_old=sum([weights[i]*self.old_qa_loss[id]['end_loss'] if id in self.old_qa_loss else weights[i]*self.max_old_end_loss for i,id in enumerate(neight_question_id) ])
            
            logits_x = logits[batch_num].squeeze(0) 
            old_logits_x = old_logits[batch_num].squeeze(0)

            neight_qa_start_loss_diff = []
            neight_qa_end_loss_diff = []
            neight_qa_total_loss_diff = []
            qa_weights = []

            neight_qa_start_token_loss_diff = []
            neight_qa_end_token_loss_diff = []

            neighbor_data = []
            neighbor_data_old = []
            for i, q_id in enumerate(neight_question_id):
                if q_id in self.start_qa_loss_diff:
                    # neight_qa_start_token_loss_diff.append(self.start_token_qa_loss_diff[neighbors_id[i]])
                    # neight_qa_end_token_loss_diff.append(self.end_token_qa_loss_diff[neighbors_id[i]])

                    neight_qa_start_loss_diff.append(self.start_qa_loss_diff[q_id])
                    neight_qa_end_loss_diff.append(self.end_qa_loss_diff[q_id])
                    # neight_qa_total_loss_diff.append(self.total_loss_diff[q_id])
                    qa_weights.append(weights[i])
                    
                    #Note: 这里用来log相关信息
                    neighbor_data.append(NeighborData(
                                            distance=distance[i]
                                            ,distance_weight=weights[i],
                                            # prefix=neight_prefix[i],
                                            gt_token=neighbor_gts[i],
                                            asr_loss_diff=neight_loss_diff[i],
                                            qa_start_loss_diff=self.start_qa_loss_diff[q_id],
                                            qa_end_loss_diff=self.end_qa_loss_diff[q_id],
                                            ))
                    
                    neighbor_data_old.append(NeighborData(
                                            distance=distance_old[i]
                                            ,distance_weight=weights_old[i],
                                            # prefix=neight_prefix[i],
                                            gt_token=neighbor_gts_old[i],
                                            asr_loss_diff=neight_loss_diff_old[i],
                                            qa_start_loss_diff=self.start_qa_loss_diff[q_id],
                                            qa_end_loss_diff=self.end_qa_loss_diff[q_id],
                                            ))
                else:
                    #TODO: 解决这个不一致的问题
                    pass
                    #print('Error: not found q_id in start_qa_loss_diff %s' %q_id)

            neighbors_log_data.append(neighbor_data)
            neighbors_log_data_old.append(neighbor_data_old)

            qa_diff_weight = self.get_loss_diff_weight_qa(neight_qa_start_loss_diff, neight_qa_end_loss_diff, qa_weights)
            logits_e_qa = old_logits_x *qa_diff_weight + (1-qa_diff_weight)*logits_x

            asr_diff_weight_before = self.get_loss_diff_weight_asr(neight_loss_diff, weights)
            asr_diff_weight_adjust = self.get_loss_diff_weight_asr(neight_loss_diff, weights, qa_diff_weight)

            if self.mode == 'asr':
                logits_final[batch_num] = old_logits_x *asr_diff_weight_before + (1-asr_diff_weight_before)*logits_x 
            #TODO: question的应该也在这里，只不过是用question level的qa loss
            elif self.mode == 'token' or self.mode == 'question':
                logits_final[batch_num] = logits_e_qa
            #token level的qa ensemble融到asr里了
            #TODO: question merge的应该也在这里，只不过是用question level的qa loss
            elif self.mode == 'token-merge' or self.mode == 'question-merge':
                logits_final[batch_num] = old_logits_x *asr_diff_weight_adjust + (1-asr_diff_weight_adjust)*logits_x 
            elif self.mode == "simple-asr":
                simple_loss_weight=loss_new/(loss_old+loss_new)
                logits_final[batch_num] = old_logits_x *simple_loss_weight + (1-simple_loss_weight)*logits_x
            elif self.mode == "simple-question":
                simple_loss_weight_start=qa_start_loss_new/(qa_start_loss_new+qa_start_loss_old)
                simple_loss_weight_end=qa_end_loss_new/(qa_end_loss_new+qa_end_loss_old)

                simple_loss_weight=max(simple_loss_weight_start,simple_loss_weight_end)

                logits_final[batch_num] = old_logits_x *simple_loss_weight + (1-simple_loss_weight)*logits_x

            #根据logits和old logits的pred token id, 判断其中一个有终止符的时候再打开knn rag ensemble开关
            #TODO: 这里想要解除判定限制的话，需要衡量knn检索出的内容是否靠谱，例如首个token knn neighbor直接返回终止符的情况就不太靠谱
            if old_pred[batch_num]==self.processor.tokenizer.eos_token_id or new_pred[batch_num]==self.processor.tokenizer.eos_token_id:
                logits_x = logits_final[batch_num]
                probs_x = F.softmax(logits_x, dim=-1)
                probs_knn = torch.zeros_like(probs_x)
                probs_x = probs_x * (1 - self.knn_rag_weight)
                for i in range(len(neighbors_id)):
                    probs_knn[neighbor_gts[i]] += weights[i]
                probs_x += self.knn_rag_weight * probs_knn
                logits_final[batch_num] = torch.log(probs_x)

            elif cosine_distance>0.95 :#and self.asr_lambda<0:
                # neighbor_rag_weight=self.asr_lambda*(-0.5)
                neighbor_rag_weight=0.7
                logits_x = logits_final[batch_num]
                probs_x = F.softmax(logits_x, dim=-1)
                probs_knn = torch.zeros_like(probs_x)
                probs_x = probs_x * (1 - neighbor_rag_weight)
                for i in range(len(neighbors_id)):
                    probs_knn[neighbor_gts[i]] += weights[i]
                probs_x += neighbor_rag_weight * probs_knn
                logits_final[batch_num] = torch.log(probs_x)

            diff_weights.append([asr_diff_weight_before, qa_diff_weight, asr_diff_weight_adjust])

            neighbor_new_loss[batch_num] = np.sum(np.multiply(neight_new_loss, weights))
            neighbor_old_loss[batch_num] = np.sum(np.multiply(neight_old_loss, weights))
            neighbor_gts_full[batch_num] = neighbor_gts
            neighbor_weights[batch_num] = weights

        #asr weight, qa question weight, qa token weight

        return logits_final, neighbor_old_loss, neighbor_new_loss, neighbor_gts_full, neighbor_weights, neighbors_log_data, neighbors_log_data_old, diff_weights

    def forward(self, input_features, attention_mask, decoder_input_ids, output_loss=False):
        with torch.no_grad():
            outputs1 = self.model1(input_features=input_features, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask,output_hidden_states=True)
            outputs2 = self.model2(input_features=input_features,decoder_input_ids=decoder_input_ids, attention_mask=attention_mask,output_hidden_states=True)

        #最后一个logit才是推理输出的logit，前面都是context logit
        logits1 = outputs1.logits[:, -1, :]                                
        logits2 = outputs2.logits[:, -1, :]                                
        decoder_states_old = outputs1.decoder_hidden_states[-1][:,-1,:]               
        decoder_states = outputs2.decoder_hidden_states[-1][:,-1,:]                          
        batch_size = input_features.shape[0]                                 

        # mode="old"
        # if mode=="old":
        merge_logits, neighbor_old_loss, neighbor_new_loss, neighbor_gts_full, neighbor_weights, neighbors_log_data, neighbors_log_data_old, diff_weights = self.update_logits_incrementally_knn(decoder_states_old,decoder_states,logits1, logits2)     
        # merge_logits = (logits1 + logits2)/2
        probs_merge = F.softmax(merge_logits, dim=-1)
        _, new_token_id = torch.max(probs_merge, dim=-1)


        # log top 5
        log_num=5

        probs_old=F.softmax(logits1,dim=-1)
        probs_new=F.softmax(logits2,dim=-1)

        topk_values_merge, topk_indices_merge = torch.topk(probs_merge, k=log_num, dim=-1)
        topk_values_old, topk_indices_old = torch.topk(probs_old, k=log_num, dim=-1)
        topk_values_new, topk_indices_new = torch.topk(probs_new, k=log_num, dim=-1)

        topk_logit_values_merge, topk_logit_indices_merge = torch.topk(merge_logits, k=log_num, dim=-1)
        topk_logit_values_old, topk_logit_indices_old = torch.topk(logits1, k=log_num, dim=-1)
        topk_logit_values_new, topk_logit_indices_new = torch.topk(logits2, k=log_num, dim=-1) 

        # assert topk_logit_indices_merge.tolist()==topk_indices_merge.tolist()

        decoder_input_ids = torch.cat([decoder_input_ids, torch.reshape(new_token_id, (batch_size,1))], dim=-1)
        seq_end_count = count_seq_end(decoder_input_ids, self.processor.tokenizer.eos_token_id)
        batch_step_data = []
        if self.logging:
            # batch_step_data = [] #(batch_num, )
            old_probs = F.softmax(logits1, dim=-1)
            _, old_pred =  torch.max(old_probs, dim=-1)
            new_probs = F.softmax(logits2, dim=-1)
            _, new_pred =  torch.max(new_probs, dim=-1)
            for batch_idx in range(batch_size):
                if new_token_id[batch_idx].cpu().item() != self.processor.tokenizer.eos_token_id:
                # if True:
                    step_data = RagEnsembleLogStep(prefix=decoder_input_ids[batch_idx][:-1],
                                                preds=[old_pred[batch_idx], new_pred[batch_idx], new_token_id[batch_idx]],
                                                preds_probs={"old":{"logits":topk_logit_values_old[batch_idx].tolist(),"probs":topk_values_old[batch_idx].tolist(),"token_ids":topk_indices_old[batch_idx].tolist()},
                                                            "new":{"logits":topk_logit_values_new[batch_idx].tolist(),"probs":topk_values_new[batch_idx].tolist(),"token_ids":topk_indices_new[batch_idx].tolist()},
                                                            "ensemble":{"logits":topk_logit_values_merge[batch_idx].tolist(),"probs":topk_values_merge[batch_idx].tolist(),"token_ids":topk_indices_merge[batch_idx].tolist()}},
                                                knn_neighbors=neighbors_log_data[batch_idx],
                                                knn_neighbors_old=neighbors_log_data_old[batch_idx],
                                                asr_diff_weight=diff_weights[batch_idx][0],
                                                qa_diff_weight=diff_weights[batch_idx][1],
                                                qa_attr_diff_weight=diff_weights[batch_idx][2]
                                                )
                    batch_step_data.append(step_data)
                else:
                    batch_step_data.append(None)

        if output_loss:
            return decoder_input_ids, seq_end_count==batch_size, neighbor_old_loss, neighbor_new_loss, neighbor_gts_full, neighbor_weights, batch_step_data
        else:
            return decoder_input_ids, seq_end_count==batch_size, batch_step_data
        # elif mode=="test":
        #     new_token_id=self.update_logits_knn_test1(decoder_states_old,decoder_states,logits1, logits2)
        #     new_token_id=torch.tensor(new_token_id)
        #     new_token_id = new_token_id.cuda(0)
        #     decoder_input_ids = torch.cat([decoder_input_ids, torch.reshape(new_token_id, (batch_size,1))], dim=-1)
        #     seq_end_count = count_seq_end(decoder_input_ids, self.processor.tokenizer.eos_token_id)
        #     self.logging=False
        #     return decoder_input_ids, seq_end_count==batch_size, None

    def generate_sample_batch(self, input_features, attention_mask=None, max_length=Constants.max_question_len):
    
    #greedy generation now, may use other generation methods like beam search
        batch_size =  input_features.shape[0]
        eos_token_id = self.processor.tokenizer.eos_token_id
        # decoder_input_ids = torch.tensor([batch_size* [eos_token_id]]).reshape(batch_size,-1).to(self.model1.device)
        if self.model_name == 's2t':
           decoder_input_ids = torch.tensor([batch_size* [eos_token_id]]).reshape(batch_size,-1).to(self.model1.device)
        else:
            decoder_input_ids = torch.tensor([batch_size* [self.model1.generation_config.decoder_start_token_id]]).reshape(batch_size,-1).to(self.model1.device)
        
        #TODO: 针对每个question生成，记录每个step推理过程中的knn neighbor信息，以及每个step old model与new model的logit数据

        logging_data = [[] for _ in range(batch_size)] #(batch, step_size)
        for i in range(max_length):
            decoder_input_ids, is_terminate, batch_step_data = self.forward(input_features=input_features, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask)
            
            if self.logging:
                for batch_idx in range(batch_size):
                    logging_data[batch_idx].append(batch_step_data[batch_idx])
            
            if is_terminate:
                break
        if self.model_name == 'whisper':
            decoder_input_ids[decoder_input_ids == -100] = self.processor.tokenizer.pad_token_id
        decoder_input_ids = filter_eos_token(decoder_input_ids, self.processor.tokenizer.eos_token_id)
        return  self.processor.batch_decode(decoder_input_ids, skip_special_tokens=True), logging_data
    