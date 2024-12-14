import os
import torch
from model_utils import (
    load_tokenizer_from_cache,
    load_model_from_cache,
    load_model_part
)

# Define input and parameters
input_text = "Why are penises on humans sometimes large and sometimes small?"
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.9
DO_SAMPLE = True
cache_directory = r"C:\Users\ben\.cache\huggingface\hub\models--distilgpt2\snapshots\2290a62682d06624634c1f46a6ad5be0f47f38aa"
model_parts_dir = "./model_parts"

# Load tokenizer
tokenizer = load_tokenizer_from_cache(cache_directory)
print("Tokenizer loaded successfully!")

# Load model metadata
model = load_model_from_cache(cache_directory)
print("Model metadata loaded successfully!")

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device.type.upper()}")

# Prepare input IDs
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# Determine number of layers and parts
num_layers = len(model.transformer.h)
num_chunks = len([f for f in os.listdir(model_parts_dir) if f.startswith("rank")])
layers_per_chunk = (num_layers + num_chunks - 1) // num_chunks

# Initialize hidden states
hidden_states = model.transformer.wte(input_ids) + model.transformer.wpe(
    torch.arange(input_ids.size(-1), device=device)
)

# Perform inference with no gradient computation
with torch.no_grad():
    print("\n--- Processing model parts sequentially ---")
    for i in range(num_chunks):
        # Load chunk
        file_path = os.path.join(model_parts_dir, f"rank{i}.pt")
        part_layers = model.transformer.h[i * layers_per_chunk:(i + 1) * layers_per_chunk]

        print(f"[Rank {i}] Loading part {i} into memory.")
        part = load_model_part(file_path, part_layers, device)

        # Forward pass through the current chunk
        for layer in part:
            hidden_states = layer(hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states)

        # Unload the chunk
        del part
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[Rank {i}] Part {i} removed from memory.")

    # Final processing for logits
    hidden_states = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
    hidden_states = model.transformer.ln_f(hidden_states)

    # Generate tokens
    print(f"[Rank {num_chunks}] Generating text using final activations.")
    generated_output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        do_sample=DO_SAMPLE,
    )

# Decode and output result
generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
print(f"\nGenerated text: {generated_text}")
