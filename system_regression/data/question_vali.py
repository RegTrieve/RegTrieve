import json
from transformers import AutoTokenizer
import re

# 假设文件路径
input_path = '/path_to/system_regression/data/HeySQuAD_json/dev-common-original-1002.json'
asr_path = '/path_to/pretrained_models/s2t-medium-librispeech-asr'

print("asr s2t-small input dev-1002")
# 加载模型的tokenizer
tokenizer = AutoTokenizer.from_pretrained(asr_path)


question = "how often does the european council meet how often does the european council meet how often does the european council meet how often does the european council meet how often does the european council meet how often does the european council meet how often does the european council meet"

tokenized_question = tokenizer(question, return_tensors='pt')
token_length = tokenized_question['input_ids'].shape[1]


print(f"长度：{token_length}")
