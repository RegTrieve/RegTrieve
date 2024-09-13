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
from system_regression.asr.models.nb_combiner import BayesianEnsemble
from tqdm import tqdm
from system_regression.asr.utils import get_ref_ids, fill_ref_data, wer_normalize_text, gen_md5_hashmap, filter_eos_token, load_model
from system_regression.data_prep.heysquad import get_heysquad_datasets

from torch.utils.data import DataLoader
from system_regression.common.const import Constants
from system_regression.common.str2bool import str2bool
from system_regression.asr.align.align_wrapper import AlignWrapper
from system_regression.asr.models.model_utils import *
from loguru import logger
import time

def parser_option():

    parser = argparse.ArgumentParser('argument for training')

    # model1
    parser.add_argument('--old_model', type=str, default='s2t',
                        choices=['s2t', 'wav2vec', 'whisper'])
    # model2
    parser.add_argument('--new_model', type=str, default='s2t',
                        choices=['s2t', 'wav2vec', 'whisper'])

    parser.add_argument('--oldmodel_path', type=str, help='old model snapshot')
    parser.add_argument('--newmodel_path', type=str, help='new model snapshot')

    parser.add_argument('--dataset', type=str, default='heysquad_human', choices=['heysquad_human', 'heysquad_machine'], help='dataset')

    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--align', type=str, default=None)
    parser.add_argument('--vocab_path', type=str, default=None)

    opt = parser.parse_args()

    return opt


#将logging data里所有的token id都decode为token str
def decode_logging_data(all_logging_data, tokenizer):
    for k, v in all_logging_data.items():
        new_steps = []
        for step in v['log_step']:
            if step is not None:
                step.preds_str = tokenizer.decode(step.preds)
                step.prefix_str = tokenizer.decode(step.prefix)
                step.preds_probs["old"]["tokens"]=[tokenizer.decode(int(id)) for id in step.preds_probs["old"]["token_ids"]]
                step.preds_probs["new"]["tokens"]=[tokenizer.decode(int(id)) for id in step.preds_probs["new"]["token_ids"]]
                step.preds_probs["ensemble"]["tokens"]=[tokenizer.decode(int(id)) for id in step.preds_probs["ensemble"]["token_ids"]]
                for neigh in step.knn_neighbors:
                    neigh.gt_token_str = tokenizer.decode(neigh.gt_token)
                    if neigh.prefix is  not None:
                        neigh.prefix_str = tokenizer.decode(neigh.prefix)
                        
                for neigh in step.knn_neighbors_old:
                    neigh.gt_token_str = tokenizer.decode(neigh.gt_token)
                    if neigh.prefix is  not None:
                        neigh.prefix_str = tokenizer.decode(neigh.prefix)
                
                new_steps.append(step)
            else:
                break
        v['log_step'] = new_steps

def test(model_name, model, data_loader, processor, return_logging=False):
    #TODO: add loss for asr model training
    
    model.eval()
    wers = {}
    predictions = []
    refs = []
    ids = []
    res = {}
    max_len = Constants.max_question_len
    
    all_logging_data = {} #q_id: logging_data
    total_infer_time=0
    with torch.no_grad():
        for data in tqdm(data_loader):

            input_features = data["input_features"].to(Constants.device)
            attention_mask  = data['attention_mask'].to(Constants.device) if 'attention_mask' in data  else None

            if isinstance(model, ASREnsembleModel) or isinstance(model, S2TRagEnsembleModel) or isinstance(model, BayesianEnsemble):
                '''or isinstance(model, DTWEnsembleModel)''' 
                #针对s2t目前似乎只能batch size=1评估，不知道后续实现继承ASREnsemble接口后会不会好点
                if input_features.shape[0] == 1:
                    start_time=time.time()
                    transcription_asr = model.generate_sample(input_features=input_features, attention_mask=attention_mask)
                    end_time=time.time()
                    total_infer_time+=end_time-start_time
                else:
                    #TODO: 添加fill qa loss的脚本，处理logging data，将old model, new model的str, wer, qa loss， qa attr都合并。
                    #这个logging data是跟asr file绑定的，应该起一个asr file的名字
                    start_time=time.time()
                    transcription_asr, batch_logging_data = model.generate_sample_batch(input_features=input_features, attention_mask=attention_mask)
                    end_time=time.time()
                    total_infer_time+=end_time-start_time

                    if batch_logging_data is not None: # 只在knn时调用下面代码
                        for i, q_id, logging_data, question in zip(range(len(data['ids'])), data['ids'], batch_logging_data, data['questions']):
                            all_logging_data[q_id] =  {
                                'gt_question': question,
                                'pred_question': transcription_asr[i],
                                'log_step': logging_data
                            }
                            # all_logging_data_json = to_json(all_logging_data)
                            # all_logging_data_back = from_json(all_logging_data_json)
                        
                if model_name=='whisper':
                    questions=processor.batch_decode(data['labels'], skip_special_tokens=True)
                elif model_name=='s2t':
                    questions=data['questions']
    
            #single model inference
            elif model_name=='whisper' or model_name=='whisper-rag':
                generated_ids = filter_eos_token(model.generate(input_features, max_length=max_len), processor.tokenizer.eos_token_id)
                generated_ids[generated_ids == -100] = processor.tokenizer.pad_token_id
                transcription_asr = processor.batch_decode(generated_ids, skip_special_tokens=True)

                # ###### Add the align ######
                # if vocab_aligner is None:
                #     transcription_asr = processor.batch_decode(generated_ids, skip_special_tokens=True)
                # else:
                #     transcription_asr = processor.batch_decode(generated_ids, skip_special_tokens=True)
                #     logger.debug(f'prev: {transcription_asr}')
                #     transcription_asr = vocab_aligner.run(generated_ids, skip_special_tokens=True)
                #     logger.debug(f"after: {transcription_asr}")

                questions = processor.batch_decode(data['labels'], skip_special_tokens=True)

            elif model_name =='s2t':
                #TODO: 默认的decoding方式，所以目前可能会有很多重复的单词，导致wer极大
                #TODO: s2t batch推理时可以使用eos_token_id截断后面的内容，这样应该与迭代推理的结果差不多？
                if input_features.shape[0] == 1:
                    transcription_asr = s2t_gen_sample(model, processor, input_features, attention_mask, max_length=max_len)
                else:
                    #generated_ids = model.generate(input_features)
                    #TODO: do_sample=False, num_beams=1
                    start_time=time.time()
                    output=model.generate(input_features, attention_mask=attention_mask, max_length=max_len, do_sample=False, num_beams=1)
                    end_time=time.time()
                    total_infer_time+=end_time-start_time

                    generated_ids = filter_eos_token(output, processor.tokenizer.eos_token_id)
                    transcription_asr = processor.batch_decode(generated_ids, skip_special_tokens=True)

                    # ###### Add the align ######
                    # if vocab_aligner is None:
                    #     transcription_asr = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    # else:
                    #     transcription_asr = vocab_aligner.run(generated_ids, skip_special_tokens=True)
                questions=data['questions']

            elif model_name =='s2t-rag' or model_name =='s2t-rag-reg':
                #generated_ids = model.generate(input_features)
                #TODO: 使用greedy decoding的话
                generated_ids = filter_eos_token(model.generate(input_features, attention_mask=attention_mask, max_length=max_len, do_sample=False, num_beams=1), processor.tokenizer.eos_token_id)
                transcription_asr = processor.batch_decode(generated_ids, skip_special_tokens=True)
                questions=data['questions']
            
            transcription_asr = wer_normalize_text(transcription_asr) 
            predictions.extend(transcription_asr)
            questions = wer_normalize_text(questions)
            for idx, _ in enumerate(data["ids"]): 
                res[_] = {
                    'question': questions[idx],
                    # 'context':data["context"][0],
                    'transcribe': transcription_asr[idx],
                }

            refs.extend(questions)
            ids.extend(data["ids"])

        
        if len(all_logging_data) > 0:
            decode_logging_data(all_logging_data, processor.tokenizer)

        wer_thresh = 0.2
        err_count = 0
        empty_ref_ids = []
        # for i in range(len(ids)):
        for i in tqdm(range(len(ids)), desc="Processing WER calculation"):
            if len(refs[i]) == 0:
                empty_ref_ids.append(ids[i])
                wers[ids[i]] = 1
            else:
                wers[ids[i]] = Constants.wer.compute(predictions=[predictions[i]], references=[refs[i]])
            res[ids[i]]['wer'] = wers[ids[i]]
            if wers[ids[i]] > wer_thresh:
                err_count += 1
        # avg_wer = Constants.wer.compute(predictions=predictions, references=refs)
        avg_wer = sum(list(wers.values()))/len(wers)
        #TODO: add variance of wer and histogram
        logger.info('cases num: %d, total infer time: %f, per infer time:%f' %(len(ids)-len(empty_ref_ids),total_infer_time,total_infer_time/(len(ids)-len(empty_ref_ids))))
        logger.info('Avg wer %f, Err rate %f' % (avg_wer, err_count/len(refs)))
        logger.info('Empty Ref id num: %d' % len(empty_ref_ids))
        # if isinstance(model, ASREnsembleModel) or isinstance(model, S2TRagEnsembleModel) or isinstance(model, BayesianEnsemble):
        #     exit(0)
    if return_logging:
        return wers, res, all_logging_data
    else:
        return wers, res

def nfr_validate(old_wers, new_wers, wer_thresh=0.2):
    """validation"""
    nfr_ids = []
    nfr_amount = []
    cmp_map = {}
    nfr_ratio = [] # (new_wer - old_wer) / old_wer ?, old_wer=0
    res_keys = ['TT', 'FT', 'TF', 'FF']
    res_map = {k: 0 for k in res_keys}
    for idx, wer in old_wers.items():
        cur_key = []
        if old_wers[idx]<= wer_thresh:
            cur_key.append('T')
        else:
            cur_key.append('F')
        if new_wers[idx]<= wer_thresh:
            cur_key.append('T')
        else:
            cur_key.append('F')
        res_map[''.join(cur_key)] += 1

        if old_wers[idx] <= wer_thresh < new_wers[idx]:
            nfr_ids.append(idx)
            nfr_amount.append(new_wers[idx] - old_wers[idx])
        cmp_map[idx] = [old_wers[idx], new_wers[idx]]
    logger.info('nfr rate: %f' % (len(nfr_ids)/len(old_wers)))
    logger.info('nfr amount avg: %f' % (sum(nfr_amount)/len(old_wers)))
    logger.info(res_map)
    return cmp_map

def eval_single_model(model_name, model_path, dev_ref_ids,dev_ref_hashmap, human_machine, opt, align_wrapper: Optional[AlignWrapper]=None, back_model_path=None):
    if align_wrapper is not None and align_wrapper.use_align:
        logger.info('Load align model')
        test_wers, res = align_wrapper.test_single(dev_ref_ids, human_machine, dev_ref_hashmap, opt, debug=True)
        return test_wers, res
    else:
        model, processor, data_collator = load_model(model_name, model_path, back_model_path=back_model_path)
        model = model.to(Constants.device)
        model.eval()

        _, _, test_set, test_val_set = get_heysquad_datasets(processor, model=model_name, ref_dev_ids=dev_ref_ids, ref_train_ids=None, debugging=False, test_only=True, human_machine=human_machine, ref_dev_hashmap=dev_ref_hashmap)
        test_loader = DataLoader(test_set,
                                    batch_size=opt.batch_size,
                                    shuffle=False,
                                    collate_fn=data_collator)
        test_wers,res = test(model_name, model, test_loader, processor)
        return test_wers, res

def main():
    opt = parser_option()

  #TODO: 使用asr model生成human以及machine updated file, 就跟heysquad一样的json(同时计算wer regression)。然后再用下游模型去计算
    
    

    if opt.dataset.startswith('heysquad'):
        human_machine = 'human'
        ref_dev_file = os.path.join(Constants.heysquad_json_dir, 'dev-common-original-1002.json')
        if 'machine' in opt.dataset:
            human_machine = 'machine'
            ref_dev_file = os.path.join(Constants.heysquad_json_dir, 'dev-v1.1.json')
        # TODO: change the processor load path
        #  human_machine=human

        with open(ref_dev_file, 'r') as f:
            dev_data = json.load(f)
        dev_ref_ids = get_ref_ids(dev_data)
        dev_ref_hashmap = gen_md5_hashmap(dev_data)


    else:
        raise NotImplementedError(opt.dataset)
    
    
    predict_dir = "%s/%s_transcribed/eval" %(Constants.asr_prediction_dir, human_machine)
    # predict_dir = "%s/%s_transcribed/eval-xxw-test" %(Constants.asr_prediction_dir, human_machine)
    if not os.path.exists(predict_dir):
        os.mkdir(predict_dir)

    old_file = "%s/%s.json" %(predict_dir, opt.oldmodel_path)

    if opt.align:
        new_file = "%s/%s-%s.json" %(predict_dir, opt.newmodel_path, opt.align)
        cmp_file = "%s/%s-%s-%s-cmp.json" % (predict_dir, opt.oldmodel_path, opt.newmodel_path, opt.align)
        log_file = "%s/%s-%s-%s.log" % (predict_dir, opt.oldmodel_path, opt.newmodel_path, opt.align)
    else:
        new_file = "%s/%s.json" %(predict_dir, opt.newmodel_path)
        cmp_file = "%s/%s-%s-cmp.json" % (predict_dir, opt.oldmodel_path, opt.newmodel_path)
        log_file = "%s/%s-%s.log" % (predict_dir, opt.oldmodel_path, opt.newmodel_path)

    if os.path.exists(log_file):
        os.remove(log_file)
    _ = logger.add(log_file, level='DEBUG')

    old_path = os.path.join(Constants.pretrained_model_dir, opt.oldmodel_path)
    new_path = os.path.join(Constants.pretrained_model_dir, opt.newmodel_path)

    logger.info(f'eval the old model:{opt.oldmodel_path}')
    test_old_wers, res_old = eval_single_model(
        opt.old_model,
        old_path,
        dev_ref_ids=dev_ref_ids,
        dev_ref_hashmap=dev_ref_hashmap,
        human_machine=human_machine,
        back_model_path=new_path,
        opt=opt,
        align_wrapper=None
    )


    dev_trans_old = deepcopy(dev_data)
    fill_ref_data(dev_trans_old, res_old)
    with open(old_file, 'w') as f:
        json.dump(dev_trans_old, f)

    # load align info
    align_wrapper = AlignWrapper.load_from_config(opt)
    logger.info(f'eval the new model:{opt.newmodel_path}')
    test_new_wers, res_new = eval_single_model(
        opt.new_model,
        new_path,
        dev_ref_ids=dev_ref_ids,
        dev_ref_hashmap=dev_ref_hashmap,
        human_machine=human_machine,
        back_model_path=old_path,
        opt=opt,
        align_wrapper=align_wrapper
    )
    dev_trans_new = deepcopy(dev_data)
    fill_ref_data(dev_trans_new, res_new)
    with open(new_file, 'w') as f:
        json.dump(dev_trans_new, f)
    # nfr compute
    # nfr_validate(val_loader, old_model, new_model, opt)
    cmp_map = nfr_validate(test_old_wers, test_new_wers)
    #cmp_map = nfr_validate(test_new_wers, test_old_wers)

    with open(cmp_file, 'w') as f:
        json.dump(cmp_map, f)


if __name__ == '__main__':
    main()