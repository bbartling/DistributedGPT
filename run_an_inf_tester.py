import os
import torch
from model_utils import load_model_part, load_tokenizer_from_cache, load_model_from_cache, build_alibi

# Define input and parameters
INPUT_TEXT = "You are an HVAC professional. Explain in detail what an air handling unit does in large commercial buildings, including its role in air circulation, temperature control, and energy efficiency."

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.6
TOP_K = 50
TOP_P = 0.9
DO_SAMPLE = True
CACHE_DIRECTORY = r"C:\Users\ben\.cache\huggingface\hub\models--tiiuae--falcon-7b\snapshots\ec89142b67d748a1865ea4451372db8313ada0d8"
MODEL_PARTS_DIR = "./model_parts"

# Load tokenizer
print("Loading tokenizer...")
tokenizer = load_tokenizer_from_cache(CACHE_DIRECTORY)
print("Tokenizer loaded successfully!")

# Load model metadata to extract layers
print("Loading model metadata...")
model = load_model_from_cache(CACHE_DIRECTORY)
print("Model metadata loaded successfully!")

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device.type.upper()}")

# Prepare input IDs
input_ids = tokenizer(INPUT_TEXT, return_tensors="pt").input_ids.to(device)
batch_size, seq_len = input_ids.shape

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

# Generate attention mask and alibi
attention_mask = torch.ones((batch_size, seq_len), device=device)
attention_mask = attention_mask[:, None, None, :]  # Expand dimensions to 4D
alibi = build_alibi(batch_size, model.config.num_attention_heads, seq_len, device)

# Perform inference with no gradient computation
with torch.no_grad():
    print("\n--- Processing model parts sequentially ---")
    hidden_states = model.transformer.word_embeddings(input_ids)

    for i in range(num_chunks):
        # Load chunk
        file_path = os.path.join(MODEL_PARTS_DIR, f"rank{i}.pt")
        print(f"[Rank {i}] Loading part {i} into memory.")
        part_layers = layers[i * layers_per_chunk : (i + 1) * layers_per_chunk]

        # Use `load_model_part` to load the specific part
        part = load_model_part(file_path, part_layers, device)

        # Forward pass through the current chunk
        for layer in part:
            output = layer(hidden_states, alibi=alibi, attention_mask=attention_mask)
            if isinstance(output, tuple):  # Handle tuple outputs
                hidden_states = output[0]  # Extract hidden states
            else:
                hidden_states = output

        # Unload the chunk
        del part
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[Rank {i}] Part {i} removed from memory.")

    # Final processing for logits
    hidden_states = model.transformer.ln_f(hidden_states)
    logits = model.lm_head(hidden_states)

    # Generate tokens
    print(f"[Rank {num_chunks}] Generating text using final activations.")
    generated_output = tokenizer.decode(
        torch.argmax(logits, dim=-1)[0], skip_special_tokens=True
    )

# Decode and output result
print(f"\nGenerated text: {generated_output}")

