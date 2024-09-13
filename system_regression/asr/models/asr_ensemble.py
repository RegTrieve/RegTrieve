#TODO: 1. implement average and maximum average (how to get the predict digit with asr model? should I modify generation method iteratively.)
# https://github.com/SebastianBodza/EnsembleForecasting/blob/main/EnsembleModel_torch.py
# 先尝试简单logit ensemble, 再尝试更加复杂的算法。We should get the predicted token distribution of LM and ensemble them. (参考Purifying Large Language Models by Ensembling a Small Language Model?, Regression Bugs Are In Your Model! Measuring, Reducing and Analyzing Regressions In NLP Model Updates里的ensemble可能不具备参考价值，因为只是对最终的Logit进行ensemble)
#TODO:2. impplement dropout and perturb average

#直接ensemble似乎是greedy decoding ensemble的概念，实际可能需要beam ensemble,然后算最后概率高的结果。直接ensemble的baseline是单模型greeedy decoding。
#这里的logit ensemble只适合tokenizer一样的情况(small model-> large model, add model finetuning data, 不适合tokenizer变化的情况)

import torch 
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch.nn.functional as F
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from system_regression.asr.utils import wer_normalize_text, filter_eos_token, count_seq_end
from system_regression.common.const import Constants



def s2t_gen_sample(model, processor, input_features, attention_mask, max_length=Constants.max_question_len):
    eos_token_id = processor.tokenizer.eos_token_id
    decoder_input_ids = torch.tensor([[eos_token_id]]).to(model.device)
    for _ in range(max_length):
        logits = model.forward(input_features=input_features, decoder_input_ids=decoder_input_ids).logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        _, new_token_id = torch.max(probs, dim=-1)
        decoder_input_ids  = torch.cat([decoder_input_ids, torch.reshape(new_token_id, (1,1))], dim=-1)
        if new_token_id.cpu().item() == processor.tokenizer.eos_token_id:
            break
    return processor.batch_decode(decoder_input_ids, skip_special_tokens=True)


def asr_gen_sample_batch(model, tokenizer, input_features, max_length=200):
    #greedy generation now, may use other generation methods like beam search
    batch_size = input_features.shape[0]
    decoder_input_ids = torch.tensor(batch_size*[[tokenizer.eos_token_id]]).to(model.device)
    for _ in range(max_length):
        logits = model(input_features=input_features, decoder_input_ids=decoder_input_ids).logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        _, new_token_id = torch.max(probs, dim=-1)
        decoder_input_ids  = torch.cat([decoder_input_ids, torch.reshape(new_token_id, (batch_size,1))], dim=-1)
        count = 0 
        for i in new_token_id:
            if i == tokenizer.eos_token_id or i == tokenizer.pad_token_id:
                count +=1
        if count == batch_size:
            break
    return decoder_input_ids
    
#TODO: 为了实现不同的decoding方式，需要ensemble model继承Transformers Pretraned model，然后实现forward方法，继承默认的generate方法。
class ASREnsembleModel(nn.Module):
    #现在只支持一种model架构的model进行ensemble，后续需要实现vocab对其与token 
    def __init__(self, model_name, model1, model2, processor, ensemble_method, weight_adj=False):
        super(ASREnsembleModel, self).__init__()
        self.model_name = model_name
        self.model1 = model1
        self.model2 = model2
        self.processor = processor
        self.ensemble_method = ensemble_method 
        self.weight_adj = weight_adj

    def forward(self, input_features, attention_mask, decoder_input_ids):
        with torch.no_grad():
            outputs1 = self.model1(input_features=input_features, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask)
            outputs2 = self.model2(input_features=input_features,decoder_input_ids=decoder_input_ids, attention_mask=attention_mask)

        #最后一个logit才是推理输出的logit，前面都是context logit
        logits1 = outputs1.logits[:, -1, :]
        logits2 = outputs2.logits[:, -1, :]
        batch_size = input_features.shape[0]

        new_token_id = None
        if self.ensemble_method=='max':
            probs1 = F.softmax(logits1, dim=-1)
            probs2 = F.softmax(logits2, dim=-1)
            merged_prob = torch.max(probs1, probs2)
            #TODO: 添加sampling generation的方式，先取max，再sample
            max_prob1, max_indices1 = torch.max(merged_prob, dim=-1)
            # max_prob2, max_indices2 = torch.max(probs2, dim=-1)
            new_token_id = max_indices1
        elif self.ensemble_method=='avg':
            merge_logits = (logits1 + logits2)/2
            probs = F.softmax(merge_logits, dim=-1)
            _, new_token_id = torch.max(probs, dim=-1)
        elif self.ensemble_method=='dropout':
            loops = 10
            # old_confs = F.softmax(output1, dim=-1)
            # new_confs = F.softmax(output2, dim=-1)
            pred1 = []
            pred2 = []
            #这里设置为train就将dropout打开了
            self.model1.train()
            self.model2.train()
            for _ in range(loops):
                pred1.append(F.softmax(self.model1(input_features, decoder_input_ids=decoder_input_ids,attention_mask=attention_mask).logits[:, -1, :], dim=-1))
                pred2.append(F.softmax(self.model2(input_features, decoder_input_ids=decoder_input_ids,attention_mask=attention_mask).logits[:, -1, :], dim=-1))
            var_pred1 = torch.zeros(logits1.shape).cuda()
            var_pred2 = torch.zeros(logits2.shape).cuda()
            mean_pred1 = torch.zeros(logits1.shape).cuda()
            mean_pred2 = torch.zeros(logits2.shape).cuda()
            for i in range(loops):
                mean_pred1 += pred1[i]
                mean_pred2 += pred2[i]
                var_pred1 += pred1[i].square()
                var_pred2 += pred2[i].square()
            var_pred1 = (var_pred1/loops) - (mean_pred1/loops).square()
            var_pred2 = (var_pred2/loops) - (mean_pred2/loops).square()
            # clip for numerical stability
            #TODO: check 截断是否会导致weight 1和weight2经常都是1
            weight1 = torch.clamp(var_pred1.sum(dim=-1), min=1e-3, max=1-1e-3) 
            weight2 = torch.clamp(var_pred2.sum(dim=-1), min=1e-3, max=1-1e-3)  
            # add a priori with weight1=0.0, weight2=1.0.
            if self.weight_adj:
                w1 = torch.unsqueeze((weight2 / (2*(weight1+weight2))+0.0), -1)
                w2 = torch.unsqueeze((weight1 / (2*(weight1+weight2))+0.5), -1)
            else:
                w1 = torch.unsqueeze((weight2 / (weight1+weight2)), -1)
                w2 = torch.unsqueeze((weight1 / (weight1+weight2)), -1)
            probs = torch.mul(mean_pred1 / loops, w1) + torch.mul(mean_pred2 / loops, w2)
            _, new_token_id = torch.max(probs, dim=-1)
            self.model1.eval()
            self.model2.eval()

        elif self.ensemble_method == 'pertub':
            loops = 10
            # old_confs = F.softmax(output1, dim=-1)
            # new_confs = F.softmax(output2, dim=-1)
            pred1 = []
            pred2 = []
            for _ in range(loops):
                pert = torch.randn(input_features.shape).cuda()
                pred1.append(F.softmax(self.model1(input_features + 0.05 * pert, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask).logits[:, -1, :], dim=-1))
                # pert = torch.randn(x.shape).cuda()
                pred2.append(F.softmax(self.model2(input_features + 0.05 * pert, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask).logits[:, -1, :], dim=-1))
            var_pred1 = torch.zeros(logits1.shape).cuda()
            var_pred2 = torch.zeros(logits2.shape).cuda()
            mean_pred1 = torch.zeros(logits1.shape).cuda()
            mean_pred2 = torch.zeros(logits2.shape).cuda()
            for _ in range(loops):
                mean_pred1 += pred1[_]
                mean_pred2 += pred2[_]
                var_pred1 += pred1[_].square()
                var_pred2 += pred2[_].square()
            var_pred1 = (var_pred1/loops) - (mean_pred1/loops).square()
            var_pred2 = (var_pred2/loops) - (mean_pred2/loops).square()
            # clip for numerical stability
            weight1 = torch.clamp(var_pred1.sum(dim=-1), min=1e-5, max=1-1e-5) 
            weight2 = torch.clamp(var_pred2.sum(dim=-1), min=1e-5, max=1-1e-5)  
            # set priori weight as w1=0.0, w2=1.0
            if self.weight_adj:
                w1 = torch.unsqueeze((weight2 / (2*(weight1+weight2))+0.0), -1)
                w2 = torch.unsqueeze((weight1 / (2*(weight1+weight2))+0.5), -1)
            else:
                w1 = torch.unsqueeze((weight2 / (weight1+weight2)), -1)
                w2 = torch.unsqueeze((weight1 / (weight1+weight2)), -1)
            probs = torch.mul(mean_pred1 / loops, w1) + torch.mul(mean_pred2 / loops, w2)
            _, new_token_id = torch.max(probs, dim=-1)

        # term_count = 0
        # for t_id in new_token_id:
        #     if t_id.cpu().item()==self.processor.tokenizer.eos_token_id:
        #         term_count += 1
        decoder_input_ids = torch.cat([decoder_input_ids, torch.reshape(new_token_id, (batch_size,1))], dim=-1)
        seq_end_count = count_seq_end(decoder_input_ids, self.processor.tokenizer.eos_token_id)
        return decoder_input_ids, seq_end_count==batch_size 

        
    def generate_sample(self, input_features,attention_mask, max_length=Constants.max_question_len):
        #return generate ids
        eos_token_id = self.processor.tokenizer.eos_token_id
        if self.model_name == 's2t':
            decoder_input_ids = torch.tensor([[eos_token_id]]).to(self.model1.device)
        else:
            decoder_input_ids = torch.tensor([[self.model1.generation_config.decoder_start_token_id]]).to(self.model1.device)

        for _ in range(max_length):
            decoder_input_ids, is_terminate = self.forward(input_features=input_features, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask)
            if is_terminate:
                break
        if self.model_name == 'whisper':
            decoder_input_ids[decoder_input_ids == -100] = self.processor.tokenizer.pad_token_id
        
        #for  debugging
        # logits1 = self.model1(input_features=input_features, decoder_input_ids=decoder_input_ids[:,:-1], attention_mask=attention_mask).logits[:, -1, :]
        # prob1 = F.softmax(logits1, dim=-1)
        # _, new_token_id1 = torch.max(prob1, dim=-1)

        # logits2 = self.model2(input_features=input_features, decoder_input_ids=decoder_input_ids[:,:-1], attention_mask=attention_mask).logits[:, -1, :]
        # prob2 = F.softmax(logits2, dim=-1)
        # _, new_token_id2 = torch.max(prob2, dim=-1)

        #TODO: 这个bug似乎跟默认的greedy  generation与我自己实现的不同有关
        res = self.processor.batch_decode(decoder_input_ids, skip_special_tokens=True)
        return res
    
    def generate_sample_batch(self, input_features, attention_mask=None, max_length=Constants.max_question_len):
    #greedy generation now, may use other generation methods like beam search
        batch_size =  input_features.shape[0]
        eos_token_id = self.processor.tokenizer.eos_token_id
    
        if self.model_name == 's2t':
           decoder_input_ids = torch.tensor([batch_size* [eos_token_id]]).reshape(batch_size,-1).to(self.model1.device)
        else:
            # use old model decoder_start_token_id
            decoder_input_ids = torch.tensor([batch_size* [self.model1.generation_config.decoder_start_token_id]]).reshape(batch_size,-1).to(self.model1.device)

        for _ in range(max_length):
            decoder_input_ids, is_terminate = self.forward(input_features=input_features, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask)
            if is_terminate:
                break
        if self.model_name == 'whisper':
            decoder_input_ids[decoder_input_ids == -100] = self.processor.tokenizer.pad_token_id
        decoder_input_ids = filter_eos_token(decoder_input_ids, self.processor.tokenizer.eos_token_id)
        
        #TODO: 记录ensemble过程中的data, 用于进一步分析
        logging_data = None
        return  self.processor.batch_decode(decoder_input_ids, skip_special_tokens=True), logging_data

#TODO: 将ASREnsembleModel继承Speech2TextPreTrainedModel， 同时有Speech2TextForConditionalGeneration的old model与new model，这样可以使用generate方法(greedy decoding, beam decoding等)
