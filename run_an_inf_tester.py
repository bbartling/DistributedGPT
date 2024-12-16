import os
import torch
from model_utils import load_model_part, load_tokenizer_from_cache, load_model_from_cache, build_alibi
import gc

# Define system message and instruction with structured prompt format
SYSTEM_MESSAGE = "You are a helpful assistant with expertise in HVAC systems."
INSTRUCTION = "Explain in detail what an air handling unit does in large commercial buildings, including its role in air circulation, temperature control, and energy efficiency."
INPUT_TEXT = f"<SYS> {SYSTEM_MESSAGE} <INST> {INSTRUCTION} <RESP> "

# Inference parameters
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.6
CACHE_DIRECTORY = r"C:\Users\ben\.cache\huggingface\hub\models--ericzzz--falcon-rw-1b-instruct-openorca\snapshots\29cc70a0af3ac4826702ec46667931c0b0af340b"
MODEL_PARTS_DIR = "./model_parts"

# Load tokenizer
print("Loading tokenizer...")
tokenizer = load_tokenizer_from_cache(CACHE_DIRECTORY)
print("Tokenizer loaded successfully!")

# Load model metadata
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

# Reconstruct model sequentially for text generation
with torch.no_grad():
    print("\n--- Processing model parts sequentially ---")
    hidden_states = model.transformer.word_embeddings(input_ids)

    for i in range(num_chunks):
        # Load chunk
        file_path = os.path.join(MODEL_PARTS_DIR, f"rank{i}.pt")
        print(f"[Rank {i}] Loading part {i} into memory.")
        part_layers = layers[i * layers_per_chunk : (i + 1) * layers_per_chunk]
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
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[Rank {i}] Part {i} removed from memory.")

    # Final processing for logits
    hidden_states = model.transformer.ln_f(hidden_states)
    logits = model.lm_head(hidden_states)

# Use HuggingFace's `generate` function for robust generation
print(f"[Rank {num_chunks}] Generating text using HuggingFace's generate API.")
model.to(device)

# Generate tokens
output_ids = model.generate(
    input_ids,
    max_length=MAX_NEW_TOKENS + input_ids.size(1),
    temperature=TEMPERATURE,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)

# Decode and output result
generated_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"\nGenerated text: {generated_output}")
