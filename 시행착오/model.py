########################################################
##           Testing LlamaGuard-7b(Llama 2)           ##
##                                                    ##
##         creating sentence after input text         ##
########################################################
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "meta-llama/LlamaGuard-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "In a world where AI and humans coexist,"

input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(input_ids, max_length=50, num_return_sequences=1)``


generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:")
print(generated_text)