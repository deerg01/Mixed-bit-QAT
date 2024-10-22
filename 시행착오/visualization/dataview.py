import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BitsAndBytesConfig
from torch.ao.quantization import float_qparams_weight_only_qconfig, prepare_qat
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm  
import numpy as np

# 데이터 로드
df = pd.read_csv('data/imdb.csv')  
texts = df['Series_Title'].tolist()  
labels = df['IMDB_Rating'].apply(lambda x: 1 if x >= 7.0 else 0).tolist()  # 평점 7.0 이상은 긍정(1), 미만은 부정(0)

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.25, random_state=42)

# CPU 사용 설정
device = torch.device("cpu")
print(f"Using device: {device}")

# 모델 설정
config = BertConfig.from_pretrained('google/bert_uncased_L-4_H-512_A-8', output_hidden_states=True, num_labels=2)
tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-4_H-512_A-8')
model = BertForSequenceClassification.from_pretrained('google/bert_uncased_L-4_H-512_A-8', config=config)

model.to(device)

model.qconfig = float_qparams_weight_only_qconfig
model.train()
model_prepared = prepare_qat(model)
model_prepared.to(device)

losses = []
quantization_errors = []

optimizer = torch.optim.AdamW(model_prepared.parameters(), lr=1e-5)  # Adam 사용

num_epochs = 15

for epoch in range(num_epochs):
    model_prepared.train()
    
    epoch_loss = 0.0
    correct_predictions = 0

    with tqdm(total=len(train_texts), desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch') as pbar:
        for text, label in zip(train_texts, train_labels):
            optimizer.zero_grad()

            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            labels_tensor = torch.tensor([label]).to(device)

            # Forward pass
            outputs = model_prepared(**inputs, labels=labels_tensor, output_hidden_states=True)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_predictions += (predictions == labels_tensor).sum().item()

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

    # 평균 손실 계산
    avg_loss = epoch_loss / len(train_texts)
    losses.append(avg_loss)

    # -------- Quantization Error 및 Clipping 적용 ---------
    quant_error_sum = 0
    layer_count = 0

    for idx, (name, module) in enumerate(model_prepared.named_modules()):
        if hasattr(module, 'weight') and module.weight is not None:
            original_weight = module.weight.detach().clone()

            # 클리핑 및 양자화 과정
            weight_sorted = torch.sort(original_weight.flatten())[0]
            num_elements = weight_sorted.numel()
            lower_index = int(num_elements * 0.1)  
            upper_index = int(num_elements * 0.9)  

            clipping_min = weight_sorted[lower_index].item()
            clipping_max = weight_sorted[upper_index].item()

            clipped_weight = torch.clamp(original_weight, min=clipping_min, max=clipping_max)

            quantized_weight = torch.quantize_per_tensor(clipped_weight, scale=0.1, zero_point=0, dtype=torch.qint8)
            dequantized_weight = quantized_weight.dequantize()

            # 양자화 오류 계산
            quant_error = torch.mean(torch.abs(clipped_weight - dequantized_weight)).item()
            quant_error_sum += quant_error
            layer_count += 1

    # Quantization error per epoch
    if layer_count > 0:
        avg_quant_error = quant_error_sum / layer_count
        quantization_errors.append(avg_quant_error)
    else:
        quantization_errors.append(0)  # if no layer

# --------- Visualization ---------

# Weight Distribution 시각화
layer_names = []
layer_weights = []

for idx, (name, module) in enumerate(model_prepared.named_modules()):
    if hasattr(module, 'weight') and module.weight is not None:
        original_weight = module.weight.detach().clone().cpu().numpy().flatten()
        layer_names.append(name)
        layer_weights.append(original_weight)

num_layers = len(layer_weights)
fig, axs = plt.subplots(nrows=1, ncols=num_layers, figsize=(num_layers * 5, 6))

for i, weights in enumerate(layer_weights):
    axs[i].hist(weights, bins=30, alpha=0.5, orientation='horizontal', density=True)
    axs[i].set_title(f'Weight Distribution - {layer_names[i]}')

fig.text(0.5, 0.04, 'Density', ha='center', fontsize=14)
fig.text(0.04, 0.5, 'Weight Value', va='center', rotation='vertical', fontsize=14)


plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # Adjust layout to leave space for common labels
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(12, 12))

# Loss 시각화
axs[0].plot(range(1, num_epochs + 1), losses, label="Loss", marker='o', color='blue')
axs[0].set_title("Loss by Epoch")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].grid(True)
axs[0].legend()

# Quantization Error 시각화
axs[1].plot(range(1, num_epochs + 1), quantization_errors, label="Quantization Error", marker='o', color='orange')
axs[1].set_title("Quantization Error by Epoch")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Quantization Error")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()
