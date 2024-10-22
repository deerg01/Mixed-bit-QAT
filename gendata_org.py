########################################################
##           Testing LlamaGuard-7b(Llama 2)           ##
##                                                    ##
##         original code를 해설하는 파일(공부용)        ##
########################################################

# coding=utf-8

from transformers import AutoTokenizer, AutoModelForCausalLM #토크나이저 및 모델 호출
import torch #GPU 연동, tensor 처리
import json #텍스트 모듈을 json으로 저장
import sys
import os

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
print("Tokenizer loaded!")
print("Loading model")
model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
model = model.cuda() #모델을 GPU로 전송
print("Model loaded!")

n_vocab = 500 # GPU에서 한 번에 생성할 토큰의 개수

i_start = sys.argv[1] #starting point
if os.path.exists("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl"): #이미 생성된 텍스트 수를 기반으로 루프 값 지정
    with open("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl", "r") as f:
        lines = f.readlines()
        inner_loop = len(lines) % n_vocab
        outer_loop = len(lines) // n_vocab
else:
    inner_loop = 0
    outer_loop = 0

if not os.path.exists("gen_data"): #생성된 데이터가 들어갈 경로 생성
    os.mkdir("gen_data")

for j in range(3 + outer_loop, 6): #input token i에 대해 반복적으로 텍스트 생성
    for i in range(int(i_start) * n_vocab + inner_loop, (int(i_start)+1) * n_vocab):
        print(i)
        input_ids = torch.tensor([[i]]).cuda() #입력토큰을 텐서화&GPU로 전송
        print("generating")
        outputs1 = model.generate(input_ids, do_sample=False, max_length=j) #최대길이 j만큼의 텍스트 생성
        outputs = model.generate(outputs1, do_sample=True, max_length=2048) 
        gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True) #디코딩
        text_dict = {"text" : gen_text[0]}
        with open("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl", "a") as f:
            f.write(json.dumps(text_dict))
            f.write('\n')
