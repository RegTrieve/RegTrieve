from __future__ import print_function

import os
import argparse

# import tensorboard_logger as tb_logger
import torch
import evaluate
import json
from copy import deepcopy


from system_regression.asr.test_model import nfr_validate, test, eval_single_model
from numpy import random
from system_regression.data_prep.heysquad import get_heysquad_datasets

from system_regression.asr.utils import get_ref_ids,fill_ref_data, gen_md5_hashmap, load_model, get_knn_output_file_name
from torch.utils.data import  DataLoader
from system_regression.common.const import Constants
from system_regression.asr.models.rag_model import S2TRagEnsembleModel
from system_regression.asr.models.model_utils import *
from system_regression.common.str2bool import str2bool

torch.manual_seed(42)
random.seed(42)

def parser_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')

    # model1
    parser.add_argument('--old_model', type=str, default='s2t')
    # model2
    parser.add_argument('--new_model', type=str, default='s2t')



    parser.add_argument('--oldmodel_path', type=str, help='old model snapshot')
    parser.add_argument('--newmodel_path', type=str, help='new model snapshot')

    parser.add_argument('--old_qa_loss_file', type=str, help='old model snapshot', default=None)
    parser.add_argument('--new_qa_loss_file', type=str, help='old model snapshot', default=None)

    parser.add_argument('--dataset', type=str, default='heysquad_human', choices=['heysquad_machine', 'heysquad_human'], help='dataset')
   
    parser.add_argument('--dataset_size', type=float, default=1.0, choices=[1.0, 0.5, 0.25])
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--rag_type', type=str, default='knn')
    parser.add_argument('--asr_lambda', type=float, default=None)
    parser.add_argument('--qa_asr_weight', type=float, default=None)
    parser.add_argument('--knn_rag_weight', type=float, default=0.5)
    parser.add_argument('--qa_lambda', type=float, default=None)
    parser.add_argument('--index_metric', type=str, default="L2")
    parser.add_argument('--mode', type=str, default='asr')
    parser.add_argument('--model_logging', type=str2bool, default=True)
    

    opt = parser.parse_args()
    print('old model: ', opt.old_model)
    print('new model: ', opt.new_model)
    
    return opt

def evaluate_rag_model(oldmodel_name, oldmodel_path, newmodel_name, newmodel_path, dev_ref_ids, dev_ref_hashmap, human_machine, asr_lambda, qa_lambda, qa_asr_weight, opt):
    old_index_dir = os.path.join(Constants.asr_knn_cache_dir, opt.index_metric, opt.oldmodel_path, opt.dataset) 
    index_dir = os.path.join(Constants.asr_knn_cache_dir, opt.index_metric, opt.newmodel_path, opt.dataset)
    #TODO: 后续新加载的模型需要考虑老模型的预测结果来减少regression bug
    newmodel, new_processor, new_data_collator = load_model(newmodel_name, newmodel_path, opt=opt)
    newmodel = newmodel.to(Constants.device)
    newmodel.eval()
    oldmodel, old_processor, old_data_collator = load_model(oldmodel_name, oldmodel_path)
    oldmodel = oldmodel.to(Constants.device)
    oldmodel.eval()
    
    
    model_e = S2TRagEnsembleModel(newmodel_name,oldmodel,  newmodel, new_processor, logging=opt.model_logging, mode=opt.mode)
    model_e.load_index(old_knn_dir=old_index_dir, knn_dir=index_dir,asr_lambda=asr_lambda,knn_rag_weight=opt.knn_rag_weight)

    if opt.old_qa_loss_file is not None:
        old_qa_loss_file = os.path.join(Constants.qa_knn_cache_dir, opt.oldmodel_path, opt.old_qa_loss_file)
        new_qa_loss_file = os.path.join(Constants.qa_knn_cache_dir, opt.newmodel_path, opt.new_qa_loss_file)
        model_e.load_qa_loss(old_qa_loss_file, new_qa_loss_file, qa_lambda=qa_lambda, qa_asr_weight=qa_asr_weight)
    
   
    #TODO: 目前只支持同类model的ensemble, 因此只支持一种processor。后续要支持多个
    model_id = newmodel_name.split('-')[0]
    _, _, test_set, test_val_set = get_heysquad_datasets(new_processor, model=model_id, size=opt.dataset_size, ref_dev_ids=dev_ref_ids, ref_train_ids=None, debugging=False, test_only=True, human_machine=human_machine, ref_dev_hashmap=dev_ref_hashmap)
    test_loader = DataLoader(test_set,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                collate_fn=new_data_collator)
    
    newmodel.eval()
    
    test_wers,res, logging_data = test(newmodel_name, model_e, test_loader, new_processor,return_logging=False) #return_logging=True)
    return test_wers, res, logging_data

def main():
    opt = parser_option()

    print(opt)
     # dataloader
    if opt.dataset.startswith('heysquad'):
        human_machine = 'human'
        ref_dev_file =  os.path.join(Constants.heysquad_json_dir, 'dev-common-original-1002.json')
        if 'machine' in opt.dataset:
            human_machine = 'machine'
            ref_dev_file = os.path.join(Constants.heysquad_json_dir, 'dev-v1.1.json')
        with open(ref_dev_file, 'r') as f:
            dev_data = json.load(f)
        dev_ref_ids = get_ref_ids(dev_data)
        dev_ref_hashmap = gen_md5_hashmap(dev_data)
    else:
        raise NotImplementedError(opt.dataset)
    
      #TODO: add the nfr compute for val data
     #Avg wer 0.465785, Err rate 0.489022
    
    old_path = os.path.join(Constants.pretrained_model_dir, opt.oldmodel_path)
    new_path = os.path.join(Constants.pretrained_model_dir, opt.newmodel_path)

    print('eval the old model:')
    test_old_wers,res_old = eval_single_model(opt.old_model, old_path, dev_ref_ids=dev_ref_ids, dev_ref_hashmap=dev_ref_hashmap, human_machine=human_machine, opt=opt)

    print('eval the rag model:')
    predict_dir = "%s/%s_transcribed/eval/ensemble" %(Constants.asr_prediction_dir, human_machine)
    if not os.path.exists(predict_dir):
        os.mkdir(predict_dir)
    logging_dir = "%s/%s_transcribed/eval/ensemble/knn_logs" %(Constants.asr_prediction_dir, human_machine)
    if not os.path.exists(logging_dir):
        os.mkdir(logging_dir)
    
    #TODO: 将这个参数添加到arg parse里
    #0-2, 每隔0.1
    asr_lambda_searchs = [i/10 for i in range(-20,21,1)]
    qa_lambda_searchs = [i/10 for i in range(-20,21,1)]
    #qa_lambda_searchs = [0.8,1.0,1.2,1.4,1.6,1.8,2.0]
    #qa_asr_weight_searchs = [0, 0.3, 0.6, 0.9, 1.1, 1.3, 1.5, 1.7]
    qa_asr_weight_searchs = [i/10 for i in range(0,21,1)]
    
    default_asr_lambda = -0.6
    default_qa_lambda = 1.6
    default_qa_asr_weight = 0.6
    if opt.mode == 'asr':
        asr_lambdas= asr_lambda_searchs if opt.asr_lambda is None else [opt.asr_lambda]
        qa_lambdas= [opt.qa_lambda] if opt.qa_lambda else [default_qa_lambda]
        qa_asr_weights = [opt.qa_asr_weight] if opt.qa_asr_weight else [default_qa_asr_weight]
    elif opt.mode == 'token' or opt.mode == 'question':
        asr_lambdas = [opt.asr_lambda] if opt.asr_lambda else [default_asr_lambda]
        qa_lambdas= qa_lambda_searchs if opt.qa_lambda is None else [opt.qa_lambda]
        qa_asr_weights = [opt.qa_asr_weight] if opt.qa_asr_weight else [default_qa_asr_weight]
    elif opt.mode == 'token-merge' or opt.mode == 'question-merge':
        # print("asr_lambda",opt.asr_lambda,"qa_lambda",opt.qa_lambda)
        asr_lambdas = [opt.asr_lambda] if opt.asr_lambda else [default_asr_lambda]
        qa_lambdas = [opt.qa_lambda] if opt.qa_lambda else [default_qa_lambda]
        qa_asr_weights = qa_asr_weight_searchs if opt.qa_asr_weight is None else [opt.qa_asr_weight]
    elif opt.mode=="simple-asr" or opt.mode=="simple-question":
        asr_lambdas = [default_asr_lambda]
        qa_lambdas = [default_qa_lambda]
        qa_asr_weights = [default_qa_asr_weight]
    #TODO: 添加question与question-merge的代码

    
    for asr_lambda  in asr_lambdas:
        for qa_lambda in qa_lambdas:
            for qa_asr_weight in qa_asr_weights:
                
                # asr_lambda=0.5
                # print("asr_lambda",asr_lambda)

                cur_file = get_knn_output_file_name(opt.mode, asr_lambda, qa_lambda, qa_asr_weight)
                # qa_asr_weight=0.2
                # cur_file = "test" 
                ensemble_file  = '%s/%s-%s-%s.json' % (predict_dir,  opt.oldmodel_path,  opt.newmodel_path,  cur_file)

                cmp_file = '%s/%s-%s-%s-cmp.json' % ((predict_dir, opt.oldmodel_path, opt.newmodel_path, cur_file))

                logging_file = '%s/%s-%s-%s.json' % ((logging_dir, opt.oldmodel_path, opt.newmodel_path, cur_file))

                if os.path.exists(ensemble_file):
                    print('Overriding %s' % ensemble_file)
                else:
                    print('Generating %s' % ensemble_file)
                test_ensmeble_wers,res_ensemble, logging_data = evaluate_rag_model(opt.old_model, old_path, opt.new_model, new_path, dev_ref_ids=dev_ref_ids, dev_ref_hashmap=dev_ref_hashmap, human_machine=human_machine,asr_lambda=asr_lambda,qa_lambda=qa_lambda, qa_asr_weight=qa_asr_weight, opt=opt)

                # print('eval the old model:')
                # test_old_wers, res_old = eval_single_model(opt.old_model, old_path, dev_ref_ids=dev_ref_ids, dev_ref_hashmap=dev_ref_hashmap, human_machine=human_machine, opt=opt)

                #TODO: 查看一下为啥这里的trans数量为7k，而不是1k
                dev_trans_ensemble = deepcopy(dev_data)
                fill_ref_data(dev_trans_ensemble, res_ensemble)
                with open(ensemble_file, 'w') as f:
                    json.dump(dev_trans_ensemble, f)

                cmp_map = nfr_validate(test_old_wers, test_ensmeble_wers)
                with open(cmp_file, 'w') as f:
                    json.dump(cmp_map, f)

                # print(opt.oldmodel_path,  opt.newmodel_path,cur_file,asr_lambda)
                if logging_data:
                    with open(logging_file, 'w') as f:
                        json.dump(to_json(logging_data), f)
  
  


if __name__ == '__main__':
    main()


# Avg wer 0.071078
# Avg wer 0.110294