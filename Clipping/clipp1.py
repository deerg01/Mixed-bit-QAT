import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BitsAndBytesConfig
from torch.ao.quantization import float_qparams_weight_only_qconfig, prepare_qat
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm  
import numpy as np


df = pd.read_csv('data/imdb.csv')  

texts = df['Series_Title'].tolist()  
labels = df['IMDB_Rating'].apply(lambda x: 1 if x >= 7.0 else 0).tolist()  # 평점 7.0 이상은 긍정(1), 미만은 부정(0)

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.25, random_state=42)

#set to use CPU
device = torch.device("cpu")
print(f"Using device: {device}")

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

optimizer = torch.optim.AdamW(model_prepared.parameters(), lr=1e-5) #Adam 사용

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

    #get average loss
    avg_loss = epoch_loss / len(train_texts)

    losses.append(avg_loss)

    # -------- Quantization Error 및 Clipping 적용 ---------
    quant_error_sum = 0
    layer_count = 0
    quant_bit = '8' 

    # 시각화용 플롯 설정
    #fig, axs = plt.subplots(len(list(model_prepared.named_modules())), 1, figsize=(12, 18))
    #fig.suptitle(f"Layer-wise Weight Distribution and Clipping (Epoch {epoch + 1})", fontsize=16)

    for idx, (name, module) in enumerate(model_prepared.named_modules()):
        if hasattr(module, 'weight') and module.weight is not None:
            original_weight = module.weight.detach().clone()

            weight_sorted = torch.sort(original_weight.flatten())[0]
            num_elements = weight_sorted.numel()
            lower_index = int(num_elements * 0.1)  
            upper_index = int(num_elements * 0.9)  

            clipping_min = weight_sorted[lower_index].item()
            clipping_max = weight_sorted[upper_index].item()

            clipped_weight = torch.clamp(original_weight, min=clipping_min, max=clipping_max)

            if quant_bit == '8':
                quantized_weight = torch.quantize_per_tensor(clipped_weight, scale=0.1, zero_point=0, dtype=torch.qint8)
            elif quant_bit == '4':
                quantization_config = BitsAndBytesConfig(bnb_4bit_compute_dtype=torch.bfloat16)
            else:
                raise ValueError(f"Invalid quantization bit: {quant_bit}. Choose either '8' or '4'.")

            # Quantization, Dequantization
            dequantized_weight = quantized_weight.dequantize()

            # Quantization error 계산 (원본 weight와 dequantized weight 간 차이)
            quant_error = torch.mean(torch.abs(clipped_weight - dequantized_weight)).item()
            quant_error_sum += quant_error
            layer_count += 1

            # ---- 시각화: Weight distribution & Clipping Range ----
            '''
            axs[idx].hist(original_weight.cpu().numpy().flatten(), bins=50, alpha=0.5, label='Original Weights')
            axs[idx].axvline(clipping_min, color='red', linestyle='--', label='Clipping Min')
            axs[idx].axvline(clipping_max, color='green', linestyle='--', label='Clipping Max')
            axs[idx].set_title(f"Layer: {name}")
            axs[idx].set_xlabel("Weight Value")
            axs[idx].set_ylabel("Frequency")
            axs[idx].legend()
            '''

    #Quantization error per epoch
    if layer_count > 0:
        avg_quant_error = quant_error_sum / layer_count
        quantization_errors.append(avg_quant_error)
    else:
        quantization_errors.append(0)  #if no layer

    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.show()

# --------- Visualization ---------

# Subplots for loss, quantization error, and accuracy
fig, axs = plt.subplots(2, 1, figsize=(12, 18))

#Loss 시각화
axs[0].plot(range(1, num_epochs + 1), losses, label="Loss", marker='o', color='blue')
axs[0].set_title("Loss by Epoch")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].grid(True)
axs[0].legend()

#Quantization Error 시각화
axs[1].plot(range(1, num_epochs + 1), quantization_errors, label="Quantization Error", marker='o', color='orange')
axs[1].set_title("Quantization Error by Epoch")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Quantization Error")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()
