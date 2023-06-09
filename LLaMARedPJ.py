# Attempt to implement LLaMA with RedPajama training set model

# pips transformers
# cuda-pythohn
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade --force-reinstall

import torch
from torch import cuda
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

# Set up to use GPU
device = 'cuda' if cuda.is_available() else 'cpu'
print(f"Torch Version: {torch.__version__}\nModel loaded on: {device}")

instruct = "togethercomputer/RedPajama-INCITE-7B-Instruct"
chat = "togethercomputer/RedPajama-INCITE-7B-Chat"

# Init tokenizer
tokenizer = AutoTokenizer.from_pretrained(chat)
# Init model
model = AutoModelForCausalLM.from_pretrained(chat).to(device)
print(f"Model: {model}")


# Init HF pipeline
print("Starting Hugging Face pipeline...")
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == 'cuda' else -1,
    max_length=256,
    trust_remote_code=True,
    do_sample=True,
    top_k=10,
    eos_token_id=tokenizer.eos_token_id
    #max_new_tokens=236
)

# Use HF pipeline
prompt = """
    Create a list of things to do in San Francisco
    """

sequences = pipeline(prompt)

# Print results
print('eos_token_id:',tokenizer.eos_token_id)
print(sequences[0]['generated_text'])