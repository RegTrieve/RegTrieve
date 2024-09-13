from __future__ import print_function

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from system_regression.common.const import Constants
from tqdm  import tqdm
import os
import pickle

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
    
class BayesianEnsemble(nn.Module):

    def __init__(self, model1, model2, model_name, processor,  combine='replace', c=5.0, special_token_map=None):
        super(BayesianEnsemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.models = [model1, model2]
        self.combine = combine
        self.processor = processor
        self.n_cls = processor.tokenizer.vocab_size
        self.model_name = model_name  #TODO: 支持vocab align之后应该把这里更新一下
        self.specail_token_map = special_token_map

        #n_cls 从tokenizer的vocab数量里获得
        if self.combine == 'cost_ratio':
            self.c = c
        # self.prior = torch.ones(inf_data.shape[0], opt.n_cls) * (1/opt.n_cls)

    #TODO: 应该使用train loader计算pi, 但是这样耗时多少需要评估，计算一次之后可以保存到cache file里面，后续加载后再计算即可
    def compute_pi(self, data_loader, special_token_map, cache_file):
        # cache_file = os.path.join(Constants.bayesian_cache, )
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.pi_list = pickle.load(f)
                return

        self.pi_list = []
        for model in self.models:
            #TODO: 这里的问题在于token space的类别会比较多，所以估计要用整个training data做compute_pi。以及每一个token生成的step，都要计算一下比例(类似create_knn_asr里的代码)
            pi = torch.zeros(self.n_cls, self.n_cls)
            pred_list = []
            label_list = []

            device = self.model1.device
            for data in tqdm(data_loader):
                input_features = data["input_features"].to(device)
                attention_mask  = data['attention_mask'].to(device) if 'attention_mask' in data  else None
                #For whisper model
                data['labels'][data['labels'] == -100] = self.processor.tokenizer.pad_token_id
                data['labels'] =  data['labels'].to(device)
                decoder_input_ids = data['labels']
                with torch.no_grad():
                    if 's2t' in self.model_name:
                        outputs = model(input_features=input_features, attention_mask=attention_mask,  decoder_input_ids=decoder_input_ids, labels=decoder_input_ids)
                    else:
                        outputs = model(input_features=input_features,  decoder_input_ids=decoder_input_ids, labels=decoder_input_ids)

                logits = outputs.logits
                valid_lenths = decoder_input_ids.ne(self.processor.tokenizer.pad_token_id).sum(-1)
                batch_idx = 0
                #batch_size = decoder_input_ids.shape[0]
                for (inputs_1d, valid_lenth) \
                in zip(decoder_input_ids, valid_lenths):
                #Note: 这里必须要保证以<s>开头，skip掉第一个输出的token
                    start_idx = 0
                    if special_token_map:
                        while inputs_1d[start_idx].cpu().item() in special_token_map:
                            start_idx  += 1
                        start_idx -= 1 
                    for position in range(start_idx, valid_lenth-1):
                        groud_truth = int(inputs_1d[position+1].item())
                        pred = logits[batch_idx][position]
                        pred = torch.argmax(pred)
                        pred_list.append(pred.cpu().item())
                        label_list.append(groud_truth)
                batch_idx += 1
            
            labels = np.array(label_list)
            preds = np.array(pred_list)
            print('filling pi')
            for k in tqdm(range(self.n_cls)):
                for i in range(self.n_cls):
                    pi[i,k] = np.logical_and(preds == i, labels == k).sum() / ((labels == k).sum() +1) # p(pred = i | label = k)
            self.pi_list.append(pi)
        with open(cache_file, 'wb') as f:
            pickle.dump(self.pi_list, f)
            

    def compute_posterior(self, pred1, pred2):
        #TODO: 我们这里的计算方式不能是所有的preds都计算好了再更新，应该是针对生成的每个batch的token进行一次更新，这种更新不考虑old model与new model的logit，只针对输出的token id进行ensemble        
        for i, preds in enumerate([pred1, pred2]):                
            if i == 0:
                self.preds = preds.clone()
                self.prior = torch.ones(preds.shape[0], self.n_cls) * (1/self.n_cls)
                posterior = torch.zeros(preds.shape[0], self.n_cls)
            pi = self.pi_list[i]
            for y in range(preds.shape[0]):
                norm = (pi[preds[y], :] * self.prior[y, :]).sum()
                posterior[y, :] =  pi[preds[y], :] * self.prior[y, :] / norm 
            if self.combine == 'replace':
                self.preds = preds
            if self.combine == 'max_belief':
                _, preds = torch.max(posterior, 1)
                self.preds = preds
            elif self.combine == 'mbme':
                pre_entropy = - (self.prior * torch.log(self.prior + 1e-5)).sum(dim=1)
                post_entropy = - (posterior * torch.log(posterior + 1e-5)).sum(dim=1)
                _, preds = torch.max(posterior, 1)
                idx = pre_entropy > post_entropy
                preds = preds.to(self.preds.device)
                self.preds[idx] = preds[idx]
            elif self.combine == 'cost_ratio':
                tmp = F.one_hot(self.preds, self.n_cls)
                p_nf = (posterior * tmp).sum(dim=1)
                p_pf, preds = torch.max(posterior, 1)
                idx = (p_pf/p_nf) > self.c
                self.preds[idx] = preds[idx]
            self.prior = posterior.clone()
        self.preds = self.preds.to(pred1.device)
        return self.preds
    
    def forward(self, input_features, attention_mask, decoder_input_ids):
        with torch.no_grad():
            outputs1 = self.model1(input_features=input_features, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask)
            outputs2 = self.model2(input_features=input_features,decoder_input_ids=decoder_input_ids, attention_mask=attention_mask,output_hidden_states=True)

        #最后一个logit才是推理输出的logit，前面都是context logit
        logits1 = outputs1.logits[:, -1, :]
        logits2 = outputs2.logits[:, -1, :]
        decoder_states = outputs2.decoder_hidden_states[-1][:,-1,:]
        batch_size = input_features.shape[0]
        #TODO: 根据 lgoits1 和logits2算出model1与model pred的token id
        #new_token_id =
        probs1 = F.softmax(logits1, dim=-1)
        _, new_token_id1 = torch.max(probs1, dim=-1)
        probs2 = F.softmax(logits2, dim=-1)
        _, new_token_id2 = torch.max(probs2, dim=-1)
        new_token_id = self.compute_posterior(new_token_id1, new_token_id2)
        decoder_input_ids = torch.cat([decoder_input_ids, torch.reshape(new_token_id, (batch_size,1))], dim=-1)
        seq_end_count = count_seq_end(decoder_input_ids, self.processor.tokenizer.eos_token_id)

        return decoder_input_ids, seq_end_count==batch_size

    def generate_sample_batch(self, input_features, attention_mask=None, max_length=Constants.max_question_len):
        batch_size =  input_features.shape[0]
        eos_token_id = self.processor.tokenizer.eos_token_id
        # decoder_input_ids = torch.tensor([batch_size* [eos_token_id]]).reshape(batch_size,-1).to(self.model1.device)
        if self.model_name == 's2t':
           decoder_input_ids = torch.tensor([batch_size* [eos_token_id]]).reshape(batch_size,-1).to(self.model1.device)
        else:
            decoder_input_ids = torch.tensor([batch_size* [self.model1.generation_config.decoder_start_token_id]]).reshape(batch_size,-1).to(self.model1.device)
        
        for _ in range(max_length):
            decoder_input_ids, is_terminate = self.forward(input_features=input_features, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask)
            if is_terminate:
                break
        if self.model_name == 'whisper':
            decoder_input_ids[decoder_input_ids == -100] = self.processor.tokenizer.pad_token_id
        decoder_input_ids = filter_eos_token(decoder_input_ids, self.processor.tokenizer.eos_token_id)
        return  self.processor.batch_decode(decoder_input_ids, skip_special_tokens=True)

                


