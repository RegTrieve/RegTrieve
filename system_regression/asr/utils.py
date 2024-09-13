import hashlib
import json
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration, AutoProcessor, WhisperForConditionalGeneration
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import  torch
import random
import numpy as np
import  sys
# from system_regression.asr.models.rag_model import S2TRagModel, WhisperRagModel
import os



#for wav2vec
@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = True
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        #将多余的维度移除
        input_features = [{"input_values": feature['input_values'][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"][0]} for feature in features]


        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        
        #Note: training 
        if 'id' in features[0]:
            ids = [feature["id"] for feature in features]
            batch['ids'] = ids

        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

#for s2t
@dataclass
class DataCollatorS2T:
    processor: Any
    def __call__(self, data):
        input_features = [d['audio']['array'] for d in data]
        batch = self.processor(input_features, sampling_rate=16000, return_tensors="pt", padding=True)
        if 'id' in data[0]:
            batch['ids'] = [d["id"] for d in data]
        batch['questions'] = [(d["question"]) for d in data]
        batch['questions'] = wer_normalize_text(batch['questions'])
        pad_id = self.processor.tokenizer.pad_token_id
        
        label_ids = []
        max_seq_length = 200
        max_batch_len = 0
        for x in  batch['questions']:
            input_ids = self.processor.tokenizer.encode(x)[:max_seq_length]
            if input_ids[0] != 0:
                input_ids.insert(0, 0)
            if input_ids[-1] != 2:
                input_ids.append(2)

            label_ids.append(input_ids)
            if len(input_ids) > max_batch_len:
                max_batch_len = len(input_ids)

        batch['labels'] = torch.tensor([l + [pad_id] * (max_batch_len-len(l)) for l in label_ids])
        return batch

#for whisper
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels

        #Note: for evaluation step
        if 'id' in features[0]:
            ids = [feature["id"] for feature in features]
            batch['ids'] = ids
        if  'question' in features[0]:
            batch['questions'] = [(feature["question"]) for feature in features]
        return batch
    

def load_model(model_name, model_path, opt=None, back_model_path=None):
    if 'checkpoint' in model_path:
        processor = AutoProcessor.from_pretrained(back_model_path, language="en", task="transcribe")
    else:
        processor = AutoProcessor.from_pretrained(model_path, language="en", task="transcribe")
    
    if model_name == 'whisper':
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        if 'en' in model_path:
            model.generation_config.language = "en"
        model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids = None
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    elif model_name =='s2t':
        model = Speech2TextForConditionalGeneration.from_pretrained(model_path)
        data_collator = DataCollatorS2T(
        processor=processor
    )
    # elif model_name =='s2t-rag':
    #     model = S2TRagModel.from_pretrained(model_path)
    #     data_collator = DataCollatorS2T(
    #     processor=processor
    # )
    #     model.load_index(opt.index_dir, knn_lambda=opt.knn_lambda, pt_lambda=opt.pt_lambda)
    #     model.set_rag_type(opt.rag_type)
    # elif model_name =='s2t-rag-reg':
    #     model = S2TRagRegModel.from_pretrained(model_path)
    #     data_collator = DataCollatorS2T(
    #     processor=processor
    # )
    #     model.load_index(opt.index_dir, knn_lambda=opt.knn_lambda, pt_lambda=opt.pt_lambda)
    #     model.set_rag_type(opt.rag_type)

    # elif model_name =='whisper-rag':
    #     model = WhisperRagModel.from_pretrained(model_path)
    #     model.generation_config.language = "en"
    #     model.generation_config.task = "transcribe"
    #     model.generation_config.forced_decoder_ids = None
    #     data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    #     processor=processor,
    #     decoder_start_token_id=model.config.decoder_start_token_id,
    # )
        # model.load_index(opt.index_dir, knn_lambda=opt.knn_lambda, pt_lambda=opt.pt_lambda)
        # model.set_rag_type(opt.rag_type)
    return model, processor, data_collator


def load_asr_map(asr_file):
    res  = {}
    with open(asr_file) as f:
        data = json.load(f)
    for d in data['data']:
        for pag in d['paragraphs']:
            for qa in pag['qas']:
                res[qa['id']] = qa['question']
    return res

def load_qa_gt(qa_gt_file):
    res  = {}
    with open(qa_gt_file) as f:
        data = json.load(f)
    for d in data['data']:
        for pag in d['paragraphs']:
            for qa in pag['qas']:
                res[qa['id']] = {
                    'question': qa['question'], 
                    'answer': qa['answers'][0]['text'],
                    'context': pag['context']
                    }
    return res

def load_qa_pred(qa_pred_file):
    with open(qa_pred_file) as f:
        data = json.load(f)
    return data

def get_ref_ids(ref_data):
    res = []
    for d in ref_data['data']:
        for pag in d['paragraphs']:
            for qa in pag['qas']:
                res.append(qa['id'])
    return res

# return {token_id: str}
def load_special_token(dir):
    special_token_path = os.path.join(dir, 'special_tokens_map.json')
    tokenizer_path= os.path.join(dir, 'tokenizer.json')
    with open(special_token_path) as f: 
        data = json.load(f)
        special_tokens = data['additional_special_tokens']
    with open(tokenizer_path) as f: 
        data = json.load(f)
        vocab_dict = {d['content']: d['id'] for d in data['added_tokens']}
    res = {vocab_dict[token]:  token for token in special_tokens}
    return res


def fill_ref_data(ref_data, trans_res):
    missing_count = 0
    no_answer_count = 0
    total_count =0
    empty_question_count =0
    #remove not found trans

    for d in ref_data['data']:
        for pag in d['paragraphs']:
            new_qas = []
            for qa in pag['qas']:
                #for squad 2.0, there are many empty answers but plasusible answers without true answer
                total_count +=1
                if qa['id'] in trans_res:
                    if len(qa['answers']) == 0:
                        no_answer_count += 1
                    else:
                        if len(trans_res[qa['id']]['transcribe'])<2:
                            qa['question'] ='empty'
                            empty_question_count +=1
                        else:
                            qa['question'] = trans_res[qa['id']]['transcribe']
                        # skip plausible answer
                        # qa['answers'] = qa['plausible_answers']
                        new_qas.append(qa)
                else:
                    missing_count +=1
                    #print('%s not in trans_res' % qa['id'])
                    #raise Exception()
            pag['qas'] = new_qas

    print('Ref count %d, Trans count %d, Missing count %d, No Answer count %d, Empty Question count %d' % (total_count, len(trans_res), missing_count, no_answer_count, empty_question_count))
    return

#通过question+context的md5 hash来计算
def gen_md5_hashmap(ref_data):
    res = {}
    for d in ref_data['data']:
        for pag in d['paragraphs']:
            for qa in pag['qas']:
                res[cal_md5(qa['question'], pag['context'])] = qa['id'] 
                   
    return res

def cal_md5(question, context):
    return  hashlib.md5((question+context).encode()).hexdigest()

#Note: 实验发现s2t model batch 后过滤eos token结果不变，但是都跟batch size为1的时候的结果有略微的差异
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
    

def wav2vec_wer_normailize_text(s):
    import string, re
    def remove_articles(text):
        regex = re.compile(r"\b(A|AN|THE)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())
    
    def replace_white_space(text):
        return text.replace("|"," ")
    
    def lower(text):
        return text.lower()
    
    def post_process(text):
        #TODO: 在进行wer评估时，这些特殊的token都应该去掉吗
        special_chars = ['</s>', '<s>', '<pad>', '<unk>']
        for c in special_chars:
            text = text.replace(c," ")
        return text

    return [lower(white_space_fix(replace_white_space((remove_articles((post_process(i))))))) for i in s]

def wav2vec_normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re


    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def upper(text):
        return text.upper()
    
    def replace_white_space(text):
        return text.replace(" ","|")
    

    return [replace_white_space(white_space_fix((remove_punc(upper(i))))) for i in s]

def wer_normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re
    
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    # def replace_white_space(text):
    #     return text.replace(" ","|")
    

    return [lower(white_space_fix((remove_punc(i)))) for i in s]

def whisper_wer_normailize_text(s):
    import string, re
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())
    
    # def replace_white_space(text):
    #     return text.replace("|"," ")
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    

    return [white_space_fix(remove_articles(remove_punc((lower(i))))) for i in s]

def set_seed(param):
    random.seed(param)
    np.random.seed(param)
    torch.manual_seed(param)
    torch.cuda.manual_seed_all(param)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_knn_output_file_name(mode, asr_lambda, qa_lambda, qa_asr_weight):
    
    if mode == 'asr':
        return 'asr-asrlambda%.1f' % asr_lambda
    elif mode == 'question':
        return 'qa-%s-qalambda%.1f' % (mode, qa_lambda)
    elif mode == 'question-merge':
        return 'qa-%s-qalambda%.1f-asrlambda%.1f-qaweight%.1f' % (mode, qa_lambda, asr_lambda, qa_asr_weight)
    elif mode == 'token':
        return 'qa-%s-qalambda%.1f' % (mode, qa_lambda)
    elif mode == 'token-merge':
        return 'qa-%s-qalambda%.1f-asrlambda%.1f-qaweight%.1f' % (mode, qa_lambda, asr_lambda, qa_asr_weight)
    elif mode == 'simple-asr':
        return 'simple-asr'
    elif mode == 'simple-question':
        return 'simple-question'