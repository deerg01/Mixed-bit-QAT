import torch
import time
import datetime
import random
import pandas as pd
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, classification_report


############################# 실행 환경 설정 #############################
if torch.cuda.is_available():       
    device = torch.device("cuda")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

############################# 데이터 로딩 및 전처리 #############################
file_path = "./data/mrpc/msr_paraphrase_train.txt" 
data = []

with open(file_path, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        # Split the line into label and sentences
        parts = line.strip().split('\t')
        if len(parts) == 5:
            label = int(parts[0])  
            sentence1 = parts[3]
            sentence2 = parts[4]
            data.append((sentence1, sentence2, label))

df = pd.DataFrame(data, columns=['sentence1', 'sentence2', 'label'])

sentences1 = df.sentence1.values
sentences2 = df.sentence2.values
labels = df.label.values

print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", do_lower_case=True)

max_len = 256 

input_ids = []
attention_masks = []

for sent1, sent2 in zip(sentences1, sentences2):
    encoded_dict = tokenizer.encode_plus(
                        sent1,                      # First sentence to encode.
                        sent2,                      # Second sentence to encode.
                        add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
                        max_length = max_len,       # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True, # Construct attention masks.
                        return_tensors = 'pt',      # Return pytorch tensors.
                   )   
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

batch_size = 32

train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),  # Select batches randomly
            batch_size = batch_size
        )
validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),  # Pull out batches sequentially.
            batch_size = batch_size 
        )

############################# BERT model 호출 및 준비 #############################
model = BertForSequenceClassification.from_pretrained( 
    "google-bert/bert-base-uncased", 
    num_labels = 2,
    output_attentions = False, 
    output_hidden_states = False,
)

quantization_config = torch.ao.quantization.get_default_qconfig("x86")

model.cuda()

############################# 학습 준비 #############################
optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )
epochs = 5

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

############################# training and validation method  #############################
def train_model(epochs, model, train_dataloader, validation_dataloader, optimizer, scheduler):
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    seed_val = 42  # Set the seed value all over the place to make this reproducible.

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []  # training, validation loss, validation accuracy, timings.

    total_t0 = time.time()
    for epoch_i in range(0, epochs):
        #-------- start training
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0

        model.train()  # set model to train mode

        for step, batch in enumerate(tqdm(train_dataloader, desc="Training", leave=True)):
            # unpack training batches from dataloader
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()        

            # forward pass
            output = model(b_input_ids, 
                           token_type_ids=None, 
                           attention_mask=b_input_mask, 
                           labels=b_labels)
        
            # check process output
            if output is None:
                print("There's no output. Check the model's forward method.")
            if output.loss is None:
                print("There's no output.loss. Check the model's forward method.")
            else:
                loss = output.loss
                logits = output.logits
            
                total_train_loss += loss.item()

                # backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
                optimizer.step()  # get gradients
                scheduler.step()  # Update learning rate.


        avg_train_loss = total_train_loss / len(train_dataloader)            
        training_time = format_time(time.time() - t0)
        
        # 결과 출력
        print("--------------------------------------")
        print("Average training loss: {0:.2f}".format(avg_train_loss))
        print("Total training time: {:}".format(training_time))
        print(f"Total layers: {total_layers}, In-range Attention layers: {inrange_att_layers}, In-range FFN layers: {inrange_ffn_layers}")
        print("--------------------------------------")
        total_layers = 0
        inrange_att_layers = 0
        inrange_ffn_layers = 0
        # -------- start validation
        print("Running Validation...")

        t0 = time.time()

        model.eval()  # set model to evaluation mode
 
        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in tqdm(validation_dataloader, desc="Validating", leave=True):
            # Unpack training batches from dataloader.  
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            with torch.no_grad():        
                output = model(b_input_ids, 
                               token_type_ids=None, 
                               attention_mask=b_input_mask,
                               labels=b_labels)
                loss = output.loss
                logits = output.logits
                
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)
            

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)

        # 결과 출력
        print("--------------------------------------")
        print("  *Accuracy: {0:.2f}".format(avg_val_accuracy))
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation time: {:}".format(validation_time))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    ############################# end of training #############################
    print("")
    print("======== Training complete! ========")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    return training_stats
############################# Inference process  #############################
def inf(model, data_loader):
    model.eval()  
    predictions, true_labels = [], []

    with torch.no_grad():
        start_time = time.time()
        for batch in data_loader:
            b_input_ids = batch[0].cuda()  
            b_attention_mask = batch[1].cuda()  
            b_labels = batch[2].cuda()  

            outputs = model(b_input_ids, attention_mask=b_attention_mask)
            logits = outputs.logits 

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            predictions.extend(preds)  
            true_labels.extend(b_labels.detach().cpu().numpy()) 

        end_time = time.time()
        inference_time = end_time - start_time  
        print(f"Inference Time: {inference_time:.4f} seconds")

    return predictions, true_labels
############################# Mixed QAT requirements  #############################
class STEQuantizeFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, input, scale, qmin, qmax):
        ctx.scale = scale
        return torch.round(input / scale).clamp(qmin, qmax) * scale  #rounding 적용
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None #original gradient 그대로 통과
def apply_QAT(layer, precision=8, mode='attention'): #applying QAT on specific layer
    class CustomQuantizationLayer(nn.Module):
        def __init__(self, layer, bits):
            super(CustomQuantizationLayer, self).__init__()
            self.bits = bits
            self.layer = layer
            self.mode = mode  
            self.layer.requires_grad_(True)  # 양자화 레이어의 gradient 활성화
            
        def forward(self, hidden_states, *args, **kwargs):
            if self.training:
                qmax = (2 ** self.bits) - 1
                qmin = -(2 ** self.bits)

                if self.mode == 'attention': #for attention layer
                    query = self.layer.query(hidden_states)
                    key = self.layer.key(hidden_states)
                    value = self.layer.value(hidden_states)
                    
                    query_scale = query.max() - query.min()
                    key_scale = key.max() - key.min()
                    value_scale = value.max() - value.min()
                    
                    query = STEQuantizeFunction.apply(query, query_scale, qmin, qmax)
                    key = STEQuantizeFunction.apply(key, key_scale, qmin, qmax)
                    value = STEQuantizeFunction.apply(value, value_scale, qmin, qmax)

                    query = query / qmax * query_scale
                    key = key / qmax * key_scale
                    value = value / qmax * value_scale
                    
                    attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (query.size(-1) ** 0.5)
                    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
                    attention_output = torch.matmul(attention_probs, value)

                    return (attention_output, )
                
                elif self.mode == 'ffn': #for feed forward network layer
                    layer_output = self.layer(hidden_states)
                    
                    output_scale = layer_output.max() - layer_output.min()
                    
                    layer_output = STEQuantizeFunction.apply(layer_output, output_scale, qmin, qmax)
                    
                    layer_output = layer_output / qmax * output_scale
                    
                    return layer_output

            return self.layer(hidden_states, *args, **kwargs)

    quant_layer = CustomQuantizationLayer(layer=layer, bits=precision)

    return quant_layer
class MixedQATBERT(nn.Module):
    def __init__(self, model):
        super(MixedQATBERT, self).__init__()
        self.bert = model

        #weight distribution based selection. 
        #가중치 분포의 threshold%를 커버하는데 필요한 range의 크기 순서대로 
        # n개는 8bit로, 나머지는 4bit로 처리하는 방식
       
        weightrange = self.sortrange()
        #print(weightrange)
        n = int(len(weightrange) * 0.4)
        print(f"top {n} layers uses 8bit QAT")

        for j, layer in enumerate(self.bert.bert.encoder.layer): 
            for i in range(len(weightrange)): 
                layid, laytype, _ = weightrange[i]

                if i < n: #upper half.
                    if layid == j: #layer의 데이터를 가지고 QAT하세요
                        if laytype == 'att':
                            print(f"upper half 'att'")
                        if laytype == 'ffn':
                            print(f"upper half 'ffn'")
                else:
                    if layid == j:
                        if laytype == 'att':
                            print(f"lower half 'att'")
                        if laytype == 'ffn':
                            print(f"lower half 'ffn'")

                    
    def getrange(self, weights, threshold=0.99):  
        weights = weights.detach().cpu().numpy()
        weights = np.array(weights) 
        #print("weights:", weights)

        mean = np.mean(weights)
        stddev = np.std(weights)
        zscore = np.percentile(weights, threshold)

        lb = mean - zscore * stddev #lower bound
        ub = mean + zscore * stddev #upper bound

        return abs(ub - lb) #return coverage
        
    def sortrange(self):
        ranges=  []

        for i, layer in enumerate(self.bert.bert.encoder.layer):  
            ranges.append((i, 'att', self.getrange(layer.attention.self.query.weight)))
            ranges.append((i, 'ffn', self.getrange(layer.intermediate.dense.weight)))
        
        ranges.sort(key=lambda x: x[2], reverse=True) #내림차순 정렬

        return ranges

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

############################# main #############################

print("Training weight depend Mixed-bit QAT...")
mixed_qat_model = MixedQATBERT(model)
mixed_qat_model.cuda() #Mixed-bit QAT
optimizer = AdamW(mixed_qat_model.parameters(), lr = 1e-4, eps = 1e-7 )
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps )

#training_stats = train_model(epochs, mixed_qat_model, train_dataloader, validation_dataloader, optimizer, scheduler)

predictions, true_labels = inf(model, validation_dataloader)
accuracy = accuracy_score(true_labels, predictions)
print(f"Infered Validation Accuracy: {accuracy:.4f}")
print(classification_report(true_labels, predictions, target_names=["Not Paraphrase", "Paraphrase"]))

'''
print("Training 8bit QAT...")
quantized_model.cuda() #int8bit QAT
optimizer = AdamW(quantized_model.parameters(), lr = 1e-4, eps = 1e-7 )
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps )

training_stats = train_model(epochs, quantized_model, train_dataloader, validation_dataloader, optimizer, scheduler)
'''
'''
print("Training 4bit QAT...")
quantized_4bit_model.cuda() #int4bit QAT
optimizer = AdamW(quantized_4bit_model.parameters(), lr = 1e-4, eps = 1e-7 )
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps )

training_stats = train_model(epochs, quantized_4bit_model, train_dataloader, validation_dataloader, optimizer, scheduler)
'''

pd.set_option('display.precision', 2)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
df_stats.to_csv('table_weight_mix.csv')