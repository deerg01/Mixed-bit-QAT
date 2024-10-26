import torch
import time
import datetime
import random
import pandas as pd
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

############################# 실행 환경 설정 #############################
if torch.cuda.is_available():       
    device = torch.device("cuda")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

############################# 데이터 로딩 및 전처리 #############################
df = pd.read_csv("./data/cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

# Get the lists of sentences and their labels.
sentences = df.sentence.values
labels = df.label.values

print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", do_lower_case=True)

max_len = 0

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
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
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size
        )
validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size 
        )

############################# BERT model 호출 및 준비 #############################
model = BertForSequenceClassification.from_pretrained( 
    "google-bert/bert-base-uncased", 
    num_labels = 2,
    output_attentions = False, 
    output_hidden_states = False,
)
class QuantizedBert(nn.Module): #quantizing input tensors
    def __init__(self, model_fp32):
        super(QuantizedBert, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.model_fp32 = model_fp32

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        x = self.model_fp32(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, labels=labels)
        x = self.dequant(x)
        return x
class myQuantStub(torch.nn.Module):
    def __init__(self):
        super(myQuantStub, self).__init__()
        self.scale = 32.0 / (2**4)  #into 4bit scale
        self.zero_point = 16

    def forward(self, x): # 양자화된 값을 IntTensor로 변환 (4bit or zero point)
        x_int = torch.clamp(torch.round(x / self.scale) + self.zero_point, -16, 15).to(torch.int)
        return x_int

    def dequantize(self, x_int):
        return (x_int - self.zero_point) * self.scale
class Quantized4BitBert(torch.nn.Module):
    def __init__(self, model):
        super(Quantized4BitBert, self).__init__()
        self.model_fp32 = model
        self.quant = myQuantStub()  # 4비트 양자화
        self.dequant = myQuantStub()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        # 원래 모델의 forward 메서드 호출
        outputs = self.model_fp32(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 양자화 해제 후 logits를 float 타입으로 변환하고, requires_grad를 설정
        logits = self.dequant(outputs.logits).float().detach().requires_grad_(True)
        loss = None # 손실 계산
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model_fp32.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )

quantized_model = QuantizedBert(model)
quantized_4bit_model = Quantized4BitBert(model)

quantization_config = torch.ao.quantization.get_default_qconfig("x86")
quantized_model.qconfig = torch.ao.quantization.get_default_qconfig("x86")
quantized_4bit_model.qconfig = torch.ao.quantization.get_default_qconfig("x86")

torch.ao.quantization.prepare_qat(quantized_model, inplace=True)
torch.ao.quantization.prepare_qat(quantized_4bit_model, inplace=True)

quantized_model.cuda()
quantized_4bit_model.cuda()

model.cuda()

############################# 학습 준비 #############################
optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )
epochs = 4

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

############################# training and validation  #############################
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
        print("")
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
        print("")
        print("Average training loss: {0:.2f}".format(avg_train_loss))
        print("Total training time: {:}".format(training_time))
        
        # -------- start validation
        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()  # set model to evaluation mode
 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

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
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
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

############################# Mixed QAT requirements  #############################
class STEQuantizeFunction(torch.autograd.Function): #min-max
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
                    
                    # Quantize query, key, value
                    query_scale = query.max() - query.min()
                    key_scale = key.max() - key.min()
                    value_scale = value.max() - value.min()
                    
                    query = STEQuantizeFunction.apply(query, query_scale, qmin, qmax)
                    key = STEQuantizeFunction.apply(key, key_scale, qmin, qmax)
                    value = STEQuantizeFunction.apply(value, value_scale, qmin, qmax)

                    # Dequantize query, key, value
                    query = query / qmax * query_scale
                    key = key / qmax * key_scale
                    value = value / qmax * value_scale
                    
                    attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (query.size(-1) ** 0.5)
                    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
                    attention_output = torch.matmul(attention_probs, value)

                    return (attention_output, )
                
                elif self.mode == 'ffn': #for ffn layer
                    layer_output = self.layer(hidden_states)
                    
                    # quantize
                    output_scale = layer_output.max() - layer_output.min()
                    
                    layer_output = STEQuantizeFunction.apply(layer_output, output_scale, qmin, qmax)
                    
                    # dequantize
                    layer_output = layer_output / qmax * output_scale
                    
                    return layer_output

            return self.layer(hidden_states, *args, **kwargs)

    quant_layer = CustomQuantizationLayer(layer=layer, bits=precision)

    return quant_layer
class MixedQATBERT(nn.Module):
    def __init__(self, model, attention_bits=8, ffn_bits=4):
        super(MixedQATBERT, self).__init__()
        self.bert = model
        
        for layer in self.bert.bert.encoder.layer: #attention과 FFN에 다른 quantization 적용
            layer.attention.self = apply_QAT(layer.attention.self, precision = 8, mode = 'attention')

            layer.intermediate.dense = apply_QAT(layer.intermediate.dense, precision = 4, mode = 'ffn')
            layer.output.dense = apply_QAT(layer.output.dense, precision = 4, mode = 'ffn')

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

############################# Mixed QAT #############################
mixed_qat_model = MixedQATBERT(model)

mixed_qat_model.cuda()

optimizer = AdamW(mixed_qat_model.parameters(),
                    lr = 1e-4, 
                    eps = 1e-7 
                    )
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps
                                            )

print("Training Mixed QAT BERT on CoLA...")
training_stats = train_model(epochs, mixed_qat_model, train_dataloader, validation_dataloader, optimizer, scheduler)
pd.set_option('display.precision', 2)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
df_stats.to_csv('table_int_mix.csv')