import os
import torch
from model_utils import (
    load_tokenizer_from_cache,
    load_model_from_cache,
    load_model_part
)

# Define input and parameters
INPUT_TEXT = "What does an air handling unit do inside big commercial buildings?"
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.9
DO_SAMPLE = True
CACHE_DIRECTORY = r"C:\Users\ben\.cache\huggingface\hub\models--meta-llama--Llama-3.1-8B\snapshots\d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
MODEL_PARTS_DIR = "./model_parts"

# Load tokenizer
tokenizer = load_tokenizer_from_cache(CACHE_DIRECTORY)
print("Tokenizer loaded successfully!")

# Load model metadata
model = load_model_from_cache(CACHE_DIRECTORY)
print("Model metadata loaded successfully!")

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device.type.upper()}")

# Prepare input IDs
input_ids = tokenizer(INPUT_TEXT, return_tensors="pt").input_ids.to(device)

# Dynamically detect layers
if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
    layers = model.transformer.h
elif hasattr(model, "model") and hasattr(model.model, "layers"):
    layers = model.model.layers
else:
    raise AttributeError("Cannot determine the model layers.")

num_layers = len(layers)
print(f"num_layers: {num_layers}")

num_chunks = len([f for f in os.listdir(MODEL_PARTS_DIR) if f.startswith("rank")])
print(f"num_chunks: {num_chunks}")

layers_per_chunk = (num_layers + num_chunks - 1) // num_chunks
print(f"layers_per_chunk: {layers_per_chunk}")

# Initialize hidden states with token embeddings only
hidden_states = model.model.embed_tokens(input_ids)
print(f"hidden_states: {hidden_states}")

# Perform inference with no gradient computation
with torch.no_grad():
    print("\n--- Processing model parts sequentially ---")
    for i in range(num_chunks):
        # Load chunk
        file_path = os.path.join(MODEL_PARTS_DIR, f"rank{i}.pt")
        part_layers = layers[i * layers_per_chunk:(i + 1) * layers_per_chunk]

        print(f"[Rank {i}] Loading part {i} into memory.")
        part = load_model_part(file_path, part_layers, device)

        # Forward pass through the current chunk
        for layer in part:
            hidden_states = layer(hidden_states)

        # Unload the chunk
        del part
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[Rank {i}] Part {i} removed from memory.")

    # Final processing for logits
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)

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
