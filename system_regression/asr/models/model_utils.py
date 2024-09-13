from dataclasses import dataclass
from typing import List, Optional,Dict
from pydantic import BaseModel
from copy import copy, deepcopy
import json

#TODO: 把neighbor的prefix str在构建索引的时候记录下来
#TODO: 或许可以使用uncertainty度量的指标衡量probs，然后再分析结果 
class NeighborData(BaseModel):
    distance: Optional[float] = None
    distance_weight: Optional[float] = None
    prefix: Optional[List[int]] = None
    prefix_str: Optional[str] = None
    gt_token: Optional[int] = None
    gt_token_str: Optional[str] = None
    asr_loss_diff: Optional[float] = None # new_loss - old_loss
    qa_start_loss_diff: Optional[float] = None
    qa_end_loss_diff: Optional[float] = None
    qa_start_attr_loss_diff: Optional[float] = None
    qa_end_attr_loss_diff: Optional[float] = None


class  RagEnsembleLogStep(BaseModel):
    prefix: Optional[List[int]] = None
    prefix_str: Optional[str] = None
    preds: Optional[List[int]] = None # [old_pred, new_pred, ensemble_pred]
    preds_str: Optional[str] = None
    preds_probs: Optional[Dict[str, Dict[str, List]]] = None
    knn_neighbors: Optional[List[NeighborData]] = None
    knn_neighbors_old: Optional[List[NeighborData]] = None
    asr_diff_weight: Optional[float] = None
    qa_diff_weight: Optional[float] = None
    qa_attr_diff_weight: Optional[float] = None

#
def to_json(logging_data):
    all_res =deepcopy(logging_data)
    for key, value in logging_data.items():
        log_steps = []
        for step in value['log_step']:
            if step is not None:
                neighbors = [neighbor_data.model_dump_json() for neighbor_data in step.knn_neighbors]
                neighbors_old = [neighbor_data_old.model_dump_json() for neighbor_data_old in step.knn_neighbors_old]
                log_steps.append({
                    'prefix': step.prefix,
                    'prefix_str': step.prefix_str,
                    'preds': step.preds,
                    'preds_str': step.preds_str,
                    'preds_probs': step.preds_probs,
                    'knn_neighbors':neighbors,
                    'knn_neighbors_old':neighbors_old,
                    'asr_diff_weight': step.asr_diff_weight,
                    'qa_diff_weight': step.qa_diff_weight,
                    'qa_attr_diff_weight': step.qa_attr_diff_weight
                })
        res = {
            'gt_question': value['gt_question'],
            'pred_question': value['pred_question'],
            'log_step': log_steps
        }
        all_res[key] =  res
    return all_res
#return logging data
def from_json(logging_data_json):
    all_res = deepcopy(logging_data_json)
    for key, value in logging_data_json.items():
        log_steps = []
        for step in value['log_step']:
            neighbors = [NeighborData(**json.loads(neighbor_data)) for neighbor_data in step['knn_neighbors']]
            neighbors_old = [NeighborData(**json.loads(neighbor_data)) for neighbor_data in step['knn_neighbors_old']]
            log_steps.append(RagEnsembleLogStep(
                prefix=step['prefix'],
                prefix_str=step['prefix_str'],
                preds=step['preds'],
                preds_str=step['preds_str'],
                preds_probs=step['preds_probs'],
                knn_neighbors=neighbors,
                knn_neighbors_old=neighbors_old,
                asr_diff_weight=step['asr_diff_weight'],
                qa_diff_weight=step['qa_diff_weight'],
                qa_attr_diff_weight=step['qa_attr_diff_weight']
            ))
        res = {
            'gt_question': value['gt_question'],
            'pred_questrion': value['pred_question'],
            'log_step': log_steps
        }
        all_res[key] = res
    return all_res