from __future__ import print_function

import os
import argparse

# import tensorboard_logger as tb_logger
import torch
import json
import copy

from system_regression.asr.models.nb_combiner import BayesianEnsemble
from system_regression.asr.test_model import nfr_validate, test, eval_single_model, load_model
from numpy import random
from system_regression.data_prep.heysquad import get_heysquad_datasets

from system_regression.asr.utils import get_ref_ids,fill_ref_data, gen_md5_hashmap, load_special_token
from torch.utils.data import  DataLoader
from system_regression.common.const import Constants

from system_regression.common.str2bool import str2bool

torch.manual_seed(42)
random.seed(42)

def parser_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')

    # model1
    parser.add_argument('--old_model', type=str, default='s2t',
                        choices=['s2t', 'whisper', 'wav2vec'])

    # model2
    parser.add_argument('--new_model', type=str, default='s2t',
                        choices=['s2t', 'whisper', 'wav2vec'])
    parser.add_argument('--ensemble', type=str, default='max', choices=['replace', 'max_belief', 'mbme', 'cost_ratio'])

    parser.add_argument('--oldmodel_path', type=str, help='old model snapshot')
    parser.add_argument('--newmodel_path', type=str, help='new model snapshot')

    parser.add_argument('--dataset', type=str, default='heysquad_human', choices=['heysquad_machine', 'heysquad_human'], help='dataset') 

    parser.add_argument('--record', type=int, default=0)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--weight_adj', type=str2bool, default=False)

    opt = parser.parse_args()
    print('old model: ', opt.old_model)
    print('new model: ', opt.new_model)
    
    return opt

def evaluate_bayesian_model(oldmodel_name, oldmodel_path, newmodel_name, newmodel_path, dev_ref_ids, dev_ref_hashmap, human_machine, opt):
    oldmodel, old_processor, old_data_collator = load_model(oldmodel_name, oldmodel_path, back_model_path=newmodel_path)
    oldmodel = oldmodel.to(Constants.device)
    oldmodel.eval()
    newmodel, new_processor, new_data_collator = load_model(newmodel_name, newmodel_path, back_model_path=oldmodel_path)
    newmodel = newmodel.to(Constants.device)
    newmodel.eval()
    
    #TODO: 目前只支持同类model的ensemble, 因此只支持一种processor。后续要支持多个
    _, _, test_set, test_val_set = get_heysquad_datasets(old_processor, model=oldmodel_name, ref_dev_ids=dev_ref_ids, ref_train_ids=None, debugging=False, test_only=True, human_machine=human_machine, ref_dev_hashmap=dev_ref_hashmap)
    test_loader = DataLoader(test_set,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                collate_fn=old_data_collator)
    
    special_token_map= None
    if 's2t' not in oldmodel_name:
        special_token_map = load_special_token(os.path.join(Constants.pretrained_path, oldmodel_name))
    
    model_e = BayesianEnsemble(model_name=opt.old_model, model1=oldmodel, model2=newmodel, combine=opt.ensemble, processor=old_processor, special_token_map=special_token_map)
    
    #TODO: 给whisper模型对应修改为token map
    cache_file = os.path.join(Constants.bayesian_cache ,'%s-%s.pkl' % (opt.oldmodel_path, opt.newmodel_path))
    

    model_e.compute_pi(test_loader, special_token_map=special_token_map, cache_file=cache_file)

    
    test_wers,res = test(oldmodel_name, model_e, test_loader, old_processor)
    return test_wers, res

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
    
    predict_dir = "%s/%s_transcribed/eval/ensemble" %(Constants.asr_prediction_dir, human_machine)
    if not os.path.exists(predict_dir):
        os.mkdir(predict_dir)
  
  
    ensemble_file  = '%s/%s-%s-%s.json' % (predict_dir, opt.oldmodel_path, opt.newmodel_path, opt.ensemble)
    cmp_file = '%s/%s-%s-%s-cmp.json' % ((predict_dir, opt.oldmodel_path, opt.newmodel_path, opt.ensemble))

    old_path = os.path.join(Constants.pretrained_model_dir, opt.oldmodel_path)
    new_path = os.path.join(Constants.pretrained_model_dir, opt.newmodel_path)


    opt.weight_adj = False
    print('weight_adj %s' % str(opt.weight_adj))

    print('eval the old model:')
    test_old_wers,res_old = eval_single_model(opt.old_model, old_path, dev_ref_ids=dev_ref_ids, dev_ref_hashmap=dev_ref_hashmap, human_machine=human_machine, opt=opt, back_model_path=new_path)

    print('eval the ensemble model:')
    test_ensmeble_wers,res_ensemble = evaluate_bayesian_model(opt.old_model, old_path, opt.new_model, new_path, dev_ref_ids=dev_ref_ids, dev_ref_hashmap=dev_ref_hashmap, human_machine=human_machine, opt=opt)




    dev_trans_ensemble = copy.deepcopy(dev_data, res_ensemble)
    fill_ref_data(dev_trans_ensemble, res_ensemble)



    with open(ensemble_file, 'w') as f:
        json.dump(dev_trans_ensemble, f)

    cmp_map = nfr_validate(test_old_wers, test_ensmeble_wers)
    with open(cmp_file, 'w') as f:
        json.dump(cmp_map, f)
  
  


if __name__ == '__main__':
    main()


# Avg wer 0.071078
# Avg wer 0.110294