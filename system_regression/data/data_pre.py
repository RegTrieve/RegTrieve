# deal with continuous space
import json
import re

# 假设文件路径
input_path = '/path_to/train-common-original-48849.json'
new_path = '/path_to/train-common-original-48849.json'


# 读取JSON文件
with open(input_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 处理每个问题
for article in data['data']:
    for paragraph in article['paragraphs']:
        for qa in paragraph['qas']:
            question = qa['question']
            # 将连续的空格替换为一个空格
            question = re.sub(r'\s+', ' ', question).strip()
            qa['question'] = question

# 将处理后的数据保存到新文件
with open(new_path, 'w', encoding='utf-8') as file:
    json.dump(data, file)
