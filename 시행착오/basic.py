import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.ao.quantization import float_qparams_weight_only_qconfig, prepare_qat, convert
import matplotlib.pyplot as plt
import numpy as np

#source bert/bin/activate

# BERT 모델 및 토크나이저 불러오기
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
model = BertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# 디바이스 설정
device = torch.device("cpu")

# 양자화 준비
model.qconfig = float_qparams_weight_only_qconfig
model.train()

# QAT 준비
model_prepared = prepare_qat(model)
model_prepared.to(device)

# 예시 텍스트 및 레이블 생성
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt").to(device)
labels = torch.tensor([1]).to(device)

# 손실, 활성화 및 양자화 오류 기록을 위한 리스트
losses = []
activations = []
quantization_errors = []

# QAT 적용 모델로 훈련
optimizer = torch.optim.AdamW(model_prepared.parameters(), lr=1e-5)

num_epochs = 7
for epoch in range(num_epochs):
    model_prepared.train()
    optimizer.zero_grad()

    outputs = model_prepared(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    # 손실 기록
    losses.append(loss.item())

    # 활성화 값 기록 (첫 번째 배치의 logits)
    hidden_states = outputs.logits.detach()  # logits를 활성화로 사용

    # 활성화의 평균값과 표준편차 기록
    activations.append((hidden_states.mean().item(), hidden_states.std().item()))

    # 양자화 오류 계산
    original_weights = model_prepared.bert.encoder.layer[0].intermediate.dense.weight.detach()
    scale = original_weights.max() - original_weights.min()
    zero_point = original_weights.min()
    
    # 양자화: 범위를 0-255로 매핑
    quantized_weights = ((original_weights - zero_point) / scale * 255).round().clamp(0, 255).to(torch.int32)
    
    # 양자화된 가중치 재구성
    quantized_weights_q = quantized_weights.float() * (scale / 255) + zero_point
    
    # 양자화 오류 계산
    quantization_error = torch.norm(original_weights - quantized_weights_q).item()
    quantization_errors.append(quantization_error)

# 1. 손실 값 시각화
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(range(1, num_epochs + 1), losses, marker='o', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid()

# 2. 활성화 평균 및 표준편차 시각화
activation_means, activation_stds = zip(*activations)

plt.subplot(3, 1, 2)
plt.plot(range(1, num_epochs + 1), activation_means, marker='o', label='Activation Mean')
plt.plot(range(1, num_epochs + 1), activation_stds, marker='o', label='Activation Std Dev')
plt.xlabel('Epoch')
plt.ylabel('Activation Values')
plt.title('Activation Mean and Std Dev')
plt.legend()
plt.grid()

# 3. 양자화 오류 시각화
plt.subplot(3, 1, 3)
plt.plot(range(1, num_epochs + 1), quantization_errors, marker='o', label='Quantization Error')
plt.xlabel('Epoch')
plt.ylabel('Quantization Error')
plt.title('Quantization Error overview')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
