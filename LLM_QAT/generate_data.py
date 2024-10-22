# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import sys
import os

from datasets import load_dataset

# 메모리 파편화 방지를 위한 환경 변수 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_name = "meta-llama/LlamaGuard-7b"

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer loaded!")
print("Loading model")
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.cuda()
print("Model loaded!")


print("Loading SQuAD dataset")
squad = load_dataset("squad")


n_samples = 500  # 사용할 샘플 개수
squad_data = squad["train"][:n_samples]  
questions = [entry["question"] for entry in squad_data]
contexts = [entry["context"] for entry in squad_data]

i_start = sys.argv[1]
n_vocab = len(questions)

if os.path.exists("gen_data/gen_squad.chunk."+str(i_start).zfill(2)+".jsonl"):
    with open("gen_data/gen_squad.chunk."+str(i_start).zfill(2)+".jsonl", "r") as f:
        lines = f.readlines()
        inner_loop = len(lines) % n_vocab
        outer_loop = len(lines) // n_vocab
else:
    inner_loop = 0
    outer_loop = 0

if not os.path.exists("gen_data"):
    os.mkdir("gen_data")


for j in range(3 + outer_loop, 6):
    for i in range(int(i_start) * n_vocab + inner_loop, (int(i_start)+1) * n_vocab):
        print(f"Generating for question {i}: {questions[i]}")
        
        
        input_text = f"Context: {contexts[i]} Question: {questions[i]} Answer:"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()

       
        outputs = model.generate(input_ids, do_sample=True, max_length=512) 
        gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        text_dict = {
            "question": questions[i],
            "context": contexts[i],
            "generated_answer": gen_text[0]
        }

        with open("gen_data/gen_squad.chunk."+str(i_start).zfill(2)+".jsonl", "a") as f:
            f.write(json.dumps(text_dict))
            f.write('\n')

        
        torch.cuda.empty_cache()  

print("Text generation complete!")
