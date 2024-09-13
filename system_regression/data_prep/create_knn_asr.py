import argparse
import json
import logging
import os
import pickle

import faiss
import numpy as np
import torch
from tqdm import tqdm

from system_regression.asr.utils import get_ref_ids, load_model, set_seed, load_special_token
from system_regression.data_prep.heysquad import get_heysquad_datasets
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from system_regression.common.const import Constants
import math
from transformers import (
    AutoTokenizer
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
device = torch.device("cuda")

def softmax_weight(cur_w,  weights):
    avg_weights = np.average(weights)
    std_weights = np.std(weights)
    shift_weights = [(w-avg_weights)/std_weights for w in weights]
    sum_shift_weights = sum([math.exp(w) for w in shift_weights])

    return math.exp((cur_w-avg_weights)/std_weights) /sum_shift_weights

def knn_reg_relationship(data_loader, model, processor, knn_save_dir, save_model_name, hidden_size, tokenizer_name, special_token_map, distance_type='L2'):
   
    #knn_index
    # max_size = len(train_dataset 48849)*100
    max_size = 4884900
    vecs = np.zeros((max_size, hidden_size)).astype('float32')
    #info里面存的是ground truth的token id
    knn_infos = {
        'question_ids': [],
        'poss': np.zeros(max_size).astype('int16'), #记录token在当前question里的offset
        'gt_token_ids': np.zeros(max_size).astype('int32'),
        'asr_losses': np.zeros(max_size).astype('float32'),
        # 'qa_losses': np.zeros(max_size).astype('float32'),
        # 'start_attr': np.zeros(max_size).astype('float32'),
        # 'end_attr': np.zeros(max_size).astype('float32'),
        # 'prefix': []
    }

    model.eval()
    set_seed(42)
    knn_cnt = 0
    tokenizer=processor.tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(os.path.join(Constants.pretrained_model_dir, 'whisper-tiny.en'))
    loss_fct = CrossEntropyLoss(reduction='none')

    #将ground truth里的每个token切分变成sample
    batch_size = data_loader.batch_size
    for data in tqdm(data_loader):
        input_features = data["input_features"].to(device)
        attention_mask  = data['attention_mask'].to(device) if 'attention_mask' in data  else None
        #For whisper model
        data['labels'][data['labels'] == -100] = processor.tokenizer.pad_token_id
        data['labels'] =  data['labels'].to(device)
        decoder_input_ids = data['labels']
        question_ids = data['ids']
        with torch.no_grad():
            if 's2t' in save_model_name:
                outputs = model(input_features=input_features, output_hidden_states=True, attention_mask=attention_mask,   labels=decoder_input_ids) #模型自动将labels右移，添加decoder_start_token_id, 然后再作为decoder_input_ids
            else:
                outputs = model(input_features=input_features, output_hidden_states=True, labels=decoder_input_ids)

        valid_lenths = decoder_input_ids.ne(tokenizer.pad_token_id).sum(-1)
        last_hidden_states = outputs['decoder_hidden_states'][-1]
        logits = outputs.logits.view(-1, outputs.logits.size(-1))
        #TODO: 检查这里的loss的index是否能跟label index对的上
        #TODO: 似乎应该是每次生成一个token, 然后计算当前的loss, 这种计算方式与我们直接计算所有的有什么区别吗？
        loss = loss_fct(logits, data['labels'].view(-1))
        loss_matrix = loss.view(data['labels'].size())

        #TODO: 这里计算embedding似乎也要调整成跟decoding时类似的token, 添加decoding start toekn

        batch_idx = 0
        for (inputs_1d, last_hidden_states_1d, valid_lenth) \
            in zip(decoder_input_ids, last_hidden_states, valid_lenths):
            #Note: 这里必须要保证以<s>开头，skip掉第一个输出的token
            # inputs_1d = inputs_1d[1:valid_lenth]
            # last_hidden_states_1d = last_hidden_states_1d[:valid_lenth-1]
            q_id = question_ids[batch_idx]
            
            #这里是把whisper的前面几个token过滤掉，直到出现正常的token为止， 需要skip start token来设置start pos
            if special_token_map:
                start_idx = 0
                while inputs_1d[start_idx].cpu().item() in special_token_map:
                    start_idx  += 1
            else:
                start_idx = 1

            #TODO: 这里的position要考虑到s2t model与whisper model的label token shift。以及检查在推理/whisper train阶段，是否也应用了这个decoder_start_logit在第一位？           
            for position in range(start_idx, valid_lenth):
                groud_truth = int(inputs_1d[position].item())
                vecs[knn_cnt] = last_hidden_states_1d[position].cpu().numpy()
                knn_infos['gt_token_ids'][knn_cnt] = groud_truth
                knn_infos['question_ids'].append(q_id)
                knn_infos['asr_losses'][knn_cnt] = loss_matrix[batch_idx][position]
                knn_infos['poss'][knn_cnt] = position
                # knn_infos['prefix'].append(inputs_1d[:position])

                # if attr_data is None or q_id not in attr_data or tokenizer_name not in attr_data[q_id]:
                #     knn_infos['start_attr'][knn_cnt] = 1/(valid_lenth)
                #     knn_infos['end_attr'][knn_cnt] = 1/(valid_lenth)
                # else:
                #     if len(attr_data[q_id][tokenizer_name]['start_attrs']) < valid_lenth-start_idx:
                #         print('not enough q_id: %s attrs %d, %d' % (q_id, len(attr_data[q_id][tokenizer_name]['start_attrs']), valid_lenth-start_idx)) 
                #         knn_infos['start_attr'][knn_cnt] = 1/(valid_lenth)
                #         knn_infos['end_attr'][knn_cnt] = 1/(valid_lenth)
                #     else:
                #         # print('check pass')
                #         knn_infos['start_attr'][knn_cnt] = softmax_weight(attr_data[q_id][tokenizer_name]['start_attrs'][position-start_idx], attr_data[q_id][tokenizer_name]['start_attrs'])
                #         knn_infos['end_attr'][knn_cnt] = softmax_weight(attr_data[q_id][tokenizer_name]['end_attrs'][position-start_idx], attr_data[q_id][tokenizer_name]['end_attrs'])

                #要基于create_qa处理save好的每个token的ig
                #TODO: 在这里找到每个question_id里每个token对应的ig attribute score(ground truth asr file), 再匹配到当前的token id.
                #每个token的ig attribtuon要在ensemble的时候用来加权model qa loss
                
                knn_cnt += 1
                if knn_cnt >= max_size:
                    break
            batch_idx += 1
            if knn_cnt >= max_size:
                break
        if knn_cnt >= max_size:
            break
    print("knn_cnt:{}".format(knn_cnt))
    vecs = vecs[:knn_cnt]
    knn_infos['gt_token_ids'] = knn_infos['gt_token_ids'][:knn_cnt]
    knn_infos['asr_losses'] = knn_infos['asr_losses'][:knn_cnt]
    # knn_infos['start_attr'] = knn_infos['start_attr'][:knn_cnt]
    # knn_infos['end_attr'] = knn_infos['end_attr'][:knn_cnt]
    knn_infos['poss'] = knn_infos['poss'][:knn_cnt]
    # save vecs, knn_info_list to file
    if not os.path.exists(knn_save_dir):
        os.makedirs(knn_save_dir)
    # 保存数据到pickle文件中
    save_file = os.path.join(knn_save_dir, "vecs.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(vecs, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_file = os.path.join(knn_save_dir, "knn_infos.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(knn_infos, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"vecs saved to {knn_save_dir}")
    # knn 部分
    d = len(vecs[1])  # dimension
    nlist = 20  # 量化的聚类中心数，可以调整这个值
    quantizer = faiss.IndexFlatL2(d)  # 使用FlatL2作为量化器
    
    train_points_num = 10000
    train_vecs = vecs[np.random.choice(vecs.shape[0], train_points_num, replace=False)]
    if distance_type == 'L2':
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    elif distance_type == 'Inner_Product' :
        faiss.normalize_L2(train_vecs)
        faiss.normalize_L2(vecs)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    # 如果没有训练量化器，IVF 索引需要被训练
    assert not index.is_trained
    index.train(train_vecs)
    assert index.is_trained
    # 主数据是 self.vecs
    index.add(vecs)
    # index.search()

    # 保存knn_index到文件中
    index_save_path = os.path.join(knn_save_dir, "knn.index")
    faiss.write_index(index, index_save_path)
    print(f"knn.index saved to {knn_save_dir}")


#TODO:添加inner proeduct距离
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process save and data directories")
    parser.add_argument('--dataset', type=str, default='heysquad_human', choices=['heysquad_human'], help='dataset')

    opt = parser.parse_args()

    device = torch.device("cuda")
    
    pretrained_path = Constants.pretrained_model_dir

    model_list = ['s2t', 's2t', 's2t', 'whisper', 'whisper', 'whisper', 'whisper', 'whisper', 'whisper']
    model_path_names = ['s2t-small-librispeech-asr', 's2t-medium-librispeech-asr', 's2t-large-librispeech-asr', 'whisper-small.en',  'distil-small.en', 'whisper-tiny.en', 'whisper-tiny','whisper-base','whisper-base.en']
    model_tokenizer_name = ['s2t', 's2t', 's2t',  'whisper', 'whisper', 'whisper',  'whisper', 'whisper', 'whisper']
    hidden_size_dict = {'s2t-small-librispeech-asr':256, 's2t-medium-librispeech-asr': 512,'s2t-large-librispeech-asr':1024, 'whisper-tiny.en': 384, 'whisper-tiny': 384,"whisper-base":512,'whisper-base.en': 512, 'whisper-small.en': 768, 'distil-small.en': 768}
    #metric_types = ['L2', 'Inner_Product']
    metric_types = ['L2']
    selected_i = [0]
    #logger.info(f"")

    for i  in selected_i:
        for m_type in metric_types:
            knn_save_dir = os.path.join(Constants.asr_knn_cache_dir, m_type, model_path_names[i], opt.dataset)
            if os.path.exists(knn_save_dir):
                print('Skip cache %s' %  knn_save_dir)
            else:
                print('Generating %s' %  knn_save_dir)
            model, processor, data_collator = load_model(model_list[i], os.path.join(pretrained_path, model_path_names[i]))
            special_token_map= None
            if 's2t' not in model_path_names[i]:
                special_token_map = load_special_token(os.path.join(pretrained_path, model_path_names[i]))

            model = model.to(device)
            batch_size = 16
            if opt.dataset == 'heysquad_human':
                ref_dev_file = os.path.join(Constants.heysquad_json_dir, 'dev-common-original-1002.json')
                with open(ref_dev_file, 'r') as f:
                    dev_data = json.load(f)
                dev_ref_ids = get_ref_ids(dev_data)
                ref_train_file = os.path.join(Constants.heysquad_json_dir, 'train-common-original-48849.json')
                with open(ref_train_file, 'r') as f:
                    train_data = json.load(f)
                train_ref_ids = get_ref_ids(train_data)
                
                #TODO: 这里应该将training data中的question(label id)做一下wer normalize， 只有在构建asr train data时去除标点，正常推理时不去除？。因为正常推理时，whisper模型。训练数据肯定要去除
                train_set, val_set, test_set, test_val_set = get_heysquad_datasets(processor, model=model_list[i], ref_dev_ids=dev_ref_ids, ref_train_ids=train_ref_ids, debugging=False, test_only=False)
            
            data_loader = DataLoader(train_set,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        collate_fn=data_collator)

            knn_reg_relationship(data_loader=data_loader,model=model, knn_save_dir= knn_save_dir, processor=processor, save_model_name=model_path_names[i], hidden_size=hidden_size_dict[model_path_names[i]], tokenizer_name=model_tokenizer_name[i], special_token_map=special_token_map,distance_type=m_type)

    print("finished")
   
