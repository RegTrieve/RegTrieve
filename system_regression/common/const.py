import os
import torch
import evaluate

class Constants:
    project_dir = '/path_to/system_regression'#TODO
    pretrained_model_dir = os.path.join(project_dir, 'pretrained_models') 
    device = torch.device("cuda")
    wer = evaluate.load(os.path.join(project_dir, 'system_regression/common/wer')) 
    heysquad_json_dir = os.path.join(project_dir, 'system_regression/data/HeySQuAD_json')
    heysquad_human_dir = os.path.join(project_dir, 'system_regression/data/HeySQuAD_human/data')
    heysquad_machine_dir = os.path.join(project_dir, 'system_regression/data/HeySQuAD_machine/data')

    asr_prediction_dir = os.path.join(project_dir, 'system_regression/predictions/asr')
   

    qa_prediction_dir = os.path.join(project_dir, 'system_regression/predictions/qa') 
    qa_data_dir = os.path.join(project_dir, 'system_regression/predictions/qa/data_cache') 

    asr_knn_cache_dir = os.path.join(project_dir, 'system_regression/knn_cache/asr')
    qa_knn_cache_dir = os.path.join(project_dir, 'system_regression/knn_cache/qa')

    bayesian_cache =  os.path.join(project_dir, 'system_regression/bayesian_cache/')

    #模型生成question时最长的token数量
    max_question_len= 50