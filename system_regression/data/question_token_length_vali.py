import json
from transformers import AutoTokenizer
import re

# 假设文件路径
input_path = '/path_to/system_regression/data/HeySQuAD_json/dev-common-original-1002.json'
asr_path = '/path_to/pretrained_models/s2t-small-librispeech-asr'
print()
print("asr s2t-small input dev-1002")
# 加载模型的tokenizer
tokenizer = AutoTokenizer.from_pretrained(asr_path)

# 读取JSON文件
with open(input_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 提取所有问题并计算token长度
max_length = 0
max_question = ""
questions = []

for article in data['data']:
    for paragraph in article['paragraphs']:
        for qa in paragraph['qas']:
            question = qa['question']
            question = re.sub(r'\s+', ' ', question).strip()
            questions.append(question)
            tokenized_question = tokenizer(question, return_tensors='pt')
            token_length = tokenized_question['input_ids'].shape[1]
            if token_length > max_length:
                max_length = token_length
                max_question = question

print(f"最长的token长度: {max_length},问题为: {max_question}")

# 假设文件路径
input_path = '/path_to/system_regression/data/HeySQuAD_json/train-common-original-48849.json'
asr_path = '/path_to/pretrained_models/s2t-small-librispeech-asr'
print()
print("asr s2t-small input train-48849")
# 加载模型的tokenizer
tokenizer = AutoTokenizer.from_pretrained(asr_path)

# 读取JSON文件
with open(input_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 提取所有问题并计算token长度
max_length = 0
max_question = ""
questions = []

for article in data['data']:
    for paragraph in article['paragraphs']:
        for qa in paragraph['qas']:
            question = qa['question']
            question = re.sub(r'\s+', ' ', question).strip()
            questions.append(question)
            tokenized_question = tokenizer(question, return_tensors='pt')
            token_length = tokenized_question['input_ids'].shape[1]
            if token_length > max_length:
                max_length = token_length
                max_question = question

print(f"最长的token长度: {max_length},问题为: {max_question}")

# 假设文件路径
input_path = '/path_to/system_regression/data/HeySQuAD_json/dev-common-original-1002.json'
asr_path = '/path_to/pretrained_models/whisper-tiny'
print()
print("asr whisper-tiny input dev-1002")
# 加载模型的tokenizer
tokenizer = AutoTokenizer.from_pretrained(asr_path)

# 读取JSON文件
with open(input_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 提取所有问题并计算token长度
max_length = 0
max_question = ""
questions = []

for article in data['data']:
    for paragraph in article['paragraphs']:
        for qa in paragraph['qas']:
            question = qa['question']
            question = re.sub(r'\s+', ' ', question).strip()
            questions.append(question)
            tokenized_question = tokenizer(question, return_tensors='pt')
            token_length = tokenized_question['input_ids'].shape[1]
            if token_length > max_length:
                max_length = token_length
                max_question = question

print(f"最长的token长度: {max_length},问题为: {max_question}")

# 假设文件路径
input_path = '/path_to/system_regression/data/HeySQuAD_json/train-common-original-48849.json'
asr_path = '/path_to/pretrained_models/whisper-tiny'
print()
print("asr whisper-tiny input train-48849")
# 加载模型的tokenizer
tokenizer = AutoTokenizer.from_pretrained(asr_path)

# 读取JSON文件
with open(input_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 提取所有问题并计算token长度
max_length = 0
max_question = ""
questions = []

for article in data['data']:
    for paragraph in article['paragraphs']:
        for qa in paragraph['qas']:
            question = qa['question']
            question = re.sub(r'\s+', ' ', question).strip()
            questions.append(question)
            tokenized_question = tokenizer(question, return_tensors='pt')
            token_length = tokenized_question['input_ids'].shape[1]
            if token_length > max_length:
                max_length = token_length
                max_question = question

print(f"最长的token长度: {max_length},问题为: {max_question}")