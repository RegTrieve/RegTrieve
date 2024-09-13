import argparse
import json
import logging
import os
import pickle
import random

import faiss
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import sys
sys.path.append('../')

from utils import get_ref_ids, fill_ref_data, whisper_wer_normailize_text, load_model, set_seed

from utils import get_ref_ids,fill_ref_data, gen_md5_hashmap, filter_eos_token
from dataset.heysquad import get_heysquad_datasets
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from models.rag_model import S2TRagEnsembleModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
device = torch.device("cuda")


torch.manual_seed(42)
random.seed(42)

def evaluate_rag_model(oldmodel_name, oldmodel_path, newmodel_name, newmodel_path, dev_ref_ids, dev_ref_hashmap, human_machine, opt):
    # oldmodel, old_processor, old_data_collator = load_model(oldmodel_name, oldmodel_path)
    # oldmodel = oldmodel.to(device)
    # oldmodel.eval()
    #TODO: 后续新加载的模型需要考虑老模型的预测结果来减少regression bug
    newmodel, new_processor, new_data_collator = load_model(newmodel_name, newmodel_path, opt=opt, index_dir=opt.index_dir)
    newmodel = newmodel.to(device)
    newmodel.eval()
    oldmodel, old_processor, old_data_collator = load_model(oldmodel_name, oldmodel_path)
    oldmodel = oldmodel.to(device)
    oldmodel.eval()
    
    
    model_e = S2TRagEnsembleModel(newmodel_name,oldmodel,  newmodel, new_processor)
    model_e.load_index(old_knn_dir=opt.old_index_dir, knn_dir=opt.index_dir)
    
    #TODO: 目前只支持同类model的ensemble, 因此只支持一种processor。后续要支持多个
    batch_size = 8
    model_id = newmodel_name.split('-')[0]
    _, _, test_set, test_val_set = get_heysquad_datasets(new_processor, model=model_id, ref_dev_ids=dev_ref_ids, ref_train_ids=None, debugging=False, test_only=True, human_machine=human_machine, ref_dev_hashmap=dev_ref_hashmap)
    test_loader = DataLoader(test_set,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=new_data_collator)
    
    
    for data in tqdm(test_loader):
        input_features = data["input_features"].to(device)
        attention_mask  = data['attention_mask'].to(device) if 'attention_mask' in data  else None
        #For whisper model
        data['labels'][data['labels'] == -100] = new_processor.tokenizer.pad_token_id
        data['labels'] =  data['labels'].to(device)
        decoder_input_ids = data['labels']
        question_ids = data['ids']
        #add start token
        start_input_ids = torch.tensor(batch_size*[[new_processor.tokenizer.eos_token_id]]).to(device)
        decoder_input_ids = torch.cat([start_input_ids,decoder_input_ids], dim=-1)
        with torch.no_grad():
            for seq_len in range(1, decoder_input_ids.shape[1]-1):
                cur_decoder_input_ids, _, neighbor_old_loss, neighbor_new_loss, neighbor_gts, neighbor_weights = model_e(input_features=input_features, output_loss=True, attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids[:, :seq_len]
                                  )
                ground_truth = decoder_input_ids[:, seq_len+1].cpu().numpy()
                neight_gt_corr_num = np.sum(ground_truth == neighbor_gts).item()
                print(1)


    return

def knn_reg_relationship(data_loader, model, processor, knn_save_dir, save_model_name, hidden_size):
   
    #knn_index
    # max_size = len(train_dataset 48849)*100
    max_size = 4884900
    vecs = np.zeros((max_size, hidden_size)).astype('float32')
    #info里面存的是ground truth的token id
    knn_infos = {
        'question_ids': [],
        'gt_token_ids': np.zeros(max_size).astype('int32'),
        'model_losses': np.zeros(max_size).astype('float32')
    }

    model.eval()
    set_seed(42)
    knn_cnt = 0
    tokenizer=processor.tokenizer
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
                outputs = model(input_features=input_features, output_hidden_states=True, attention_mask=attention_mask,  decoder_input_ids=decoder_input_ids, labels=decoder_input_ids)
            else:
                outputs = model(input_features=input_features, output_hidden_states=True,  decoder_input_ids=decoder_input_ids, labels=decoder_input_ids)


            valid_lenths = decoder_input_ids.ne(tokenizer.pad_token_id).sum(-1)
            last_hidden_states = outputs['decoder_hidden_states'][-1]
            logits = outputs.logits.view(-1, outputs.logits.size(-1))
            loss = loss_fct(logits, data['labels'].view(-1))
            loss_matrix = loss.view(data['labels'].size())


            batch_idx = 0
            for (inputs_1d, last_hidden_states_1d, valid_lenth) \
                in zip(decoder_input_ids, last_hidden_states, valid_lenths):
                
                for position in range(valid_lenth-1):
                    groud_truth = int(inputs_1d[position+1].item())
                    vecs[knn_cnt] = last_hidden_states_1d[position].cpu().numpy()
                    knn_infos['gt_token_ids'][knn_cnt] = groud_truth
                    knn_infos['question_ids'].append(question_ids[batch_idx])
                    knn_infos['model_losses'][knn_cnt] = loss_matrix[batch_idx][position+1]
                    
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
    knn_infos['model_losses'] = knn_infos['model_losses'][:knn_cnt]
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
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process save and data directories")
    parser.add_argument('--old_model', type=str, default='s2t')
    # model2
    parser.add_argument('--new_model', type=str, default='s2t')



    parser.add_argument('--oldmodel_path', type=str, help='old model snapshot')
    parser.add_argument('--newmodel_path', type=str, help='new model snapshot')
    # parser.add_argument('--datadir', type=str, default="/data/c/base/RAPT/CodeCompletion/dataset/",
    #                     help='Directory with dataset')
    parser.add_argument('--dataset', type=str, default='heysquad_human', choices=['heysquad_human'], help='dataset')
    parser.add_argument('--rag_type', type=str, default='knn')
    parser.add_argument('--knn_lambda', type=float, default=1.0)
    parser.add_argument('--pt_lambda', type=float, default=1.0)
    parser.add_argument('--index_dir', type=str)
    parser.add_argument('--old_index_dir', type=str, default=None)
    opt = parser.parse_args()
    device = torch.device("cuda")
    
    pretrained_path = '/path_to/pretrained_models/'#TODO
    
    if opt.dataset.startswith('heysquad'):
        human_machine = 'human'
        ref_dev_file = '/path_to/HeySQuAD_json/HeySQuAD_test/dev-common-original-1002.json'#TODO
        if 'machine' in opt.dataset:
            human_machine = 'machine'
            ref_dev_file = '/path_to/HeySQuAD_json/HeySQuAD_test/dev-v1.1.json'#TODO
        with open(ref_dev_file, 'r') as f:
            dev_data = json.load(f)
        dev_ref_ids = get_ref_ids(dev_data)
        dev_ref_hashmap = gen_md5_hashmap(dev_data)

    else:
        raise NotImplementedError(opt.dataset)

   
    print('eval the rag model:')
    test_ensmeble_wers,res_ensemble = evaluate_rag_model(opt.old_model, opt.oldmodel_path, opt.new_model, opt.newmodel_path, dev_ref_ids=dev_ref_ids, dev_ref_hashmap=dev_ref_hashmap, human_machine=human_machine, opt=opt)

    

    # knn_reg_relationship(data_loader=data_loader,model=model, knn_save_dir= knn_save_dir, processor=processor, save_model_name=model_path_names[i], hidden_size=hidden_size_dict[model_path_names[i]])
   

