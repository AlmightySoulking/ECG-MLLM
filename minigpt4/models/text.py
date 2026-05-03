import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
inputs = tokenizer('Hello? How are u?', return_tensors="pt", return_attention_mask=False)
print(inputs)
embeddings = model.get_input_embeddings()(inputs.input_ids)
outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
