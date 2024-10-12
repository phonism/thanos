import os
import sys
import json
from transformers import AutoTokenizer

# 加载Qwen模型的tokenizer
tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-0.5B/")

out_f = open("./train_data", "w+")
max_length = 1025

BASE_DIR = "../../../datasets/RedPajama-Data-1T-Sample/"
for fl in os.listdir(BASE_DIR):
    if fl.find("jsonl") == -1:
        continue
    file_path = os.path.join(BASE_DIR, fl)
    with open(file_path) as f:
        buf = []  # 用于存储当前的token序列
        for line in f:
            js = json.loads(line)
            text = js["text"]
            tokenized = tokenizer(text, return_tensors="pt", truncation=False)
            input_ids = tokenized['input_ids'][0].tolist()
            input_ids.append(151643)
            buf.extend(input_ids)
            while len(buf) >= max_length:
                chunk = buf[:max_length]
                buf = buf[max_length:]  # 更新缓冲区
                out_f.write(json.dumps({'input_ids': chunk}) + '\n')
    print(fl)
out_f.close()
