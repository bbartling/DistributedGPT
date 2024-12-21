import os
import torch
from model_utils import (
    load_model_part,
    load_tokenizer_from_cache,
    load_model_from_cache,
    build_alibi,
    get_memory_usage,
    calculate_max_new_tokens
)
import gc
import time

# Metrics Storage
metrics = {"chunks": []}

# Define system message and structured prompt
SYSTEM_MESSAGE = (
    "Answer the question as truthfully as possible using the provided text, "
    "and if the answer is not contained within the text below, respond with \"I can't answer that\""
)

INSTRUCTION_TEMPLATE = (
    ">>CONTEXT<<\n{context}\n\n>>QUESTION<< {question}\n>>ANSWER<< "
)

# Example Context and Question
CONTEXT = (
    "An air handling unit (AHU) is a device used to regulate and circulate air as part of a heating, ventilating, "
    "and air-conditioning (HVAC) system. AHUs typically consist of a blower, heating or cooling elements, "
    "filter racks or chambers, sound attenuators, and dampers. AHUs are connected to ductwork that distributes "
    "the conditioned air through the building and returns it to the AHU."
)
QUESTION = "What does an air handling unit do in large commercial buildings?"

# Assemble the Input Text
INPUT_TEXT = INSTRUCTION_TEMPLATE.format(context=CONTEXT, question=QUESTION)

# Parameters
DEFAULT_MAX_NEW_TOKENS = 300
TEMPERATURE = 0.6

# Paths for model and tokenizer
FALCON_7B_CACHE_DIRECTORY = (
    r"C:\Users\ben\.cache\huggingface\hub\models--tiiuae--falcon-7b-instruct\snapshots\8782b5c5d8c9290412416618f36a133653e85285"
)
FALCON_7B_MODEL_PARTS_DIR = "./7_b_model_parts"

print("Model Parts Directory Content:")
for f in os.listdir(FALCON_7B_CACHE_DIRECTORY):
    print(f)

# Load tokenizer
tokenizer = load_tokenizer_from_cache(FALCON_7B_CACHE_DIRECTORY)
model = load_model_from_cache(FALCON_7B_CACHE_DIRECTORY)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_ids = tokenizer(INPUT_TEXT, return_tensors="pt").input_ids.to(device)

# Call the function to calculate MAX_NEW_TOKENS dynamically
MAX_NEW_TOKENS = calculate_max_new_tokens(
    input_text=INPUT_TEXT,
    tokenizer=tokenizer,
    model=model,
    max_new_tokens_default=DEFAULT_MAX_NEW_TOKENS,
    device=device
)

# Model info
layers = model.transformer.h if hasattr(model.transformer, "h") else model.model.layers
num_layers = len(layers)
num_chunks = len([f for f in os.listdir(FALCON_7B_MODEL_PARTS_DIR) if f.startswith("rank")])
layers_per_chunk = (num_layers + num_chunks - 1) // num_chunks

# Initialize alibi and masks
batch_size, seq_len = input_ids.shape
attention_mask = torch.ones((batch_size, seq_len), device=device)[:, None, None, :]
alibi = build_alibi(batch_size, model.config.num_attention_heads, seq_len, device)

# Process model sequentially
hidden_states = model.transformer.word_embeddings(input_ids)
total_time = time.time()

with torch.no_grad():
    for i in range(num_chunks):
        chunk_start_time = time.time()
        start_memory = get_memory_usage()

        # Load model part
        file_path = os.path.join(FALCON_7B_MODEL_PARTS_DIR, f"rank_{i}.pt")
        part_layers = layers[i * layers_per_chunk : (i + 1) * layers_per_chunk]
        part = load_model_part(file_path, part_layers, device)

        # Process layers in chunk
        layer_times = []
        for layer_idx, layer in enumerate(part):
            layer_start_time = time.time()
            output = layer(hidden_states, alibi=alibi, attention_mask=attention_mask)
            hidden_states = output[0] if isinstance(output, tuple) else output
            layer_time = time.time() - layer_start_time
            layer_times.append(layer_time)

        # Track chunk metrics
        chunk_time = time.time() - chunk_start_time
        end_memory = get_memory_usage()
        metrics["chunks"].append({
            "chunk": i,
            "chunk_time": chunk_time,
            "memory_used_mb": end_memory - start_memory,
            "layer_times": layer_times
        })

        # Clean up
        del part
        gc.collect()
        torch.cuda.empty_cache()

# Final logits and text generation
hidden_states = model.transformer.ln_f(hidden_states)
logits = model.lm_head(hidden_states)
model.to(device)
output_ids = model.generate(
    input_ids, max_length=MAX_NEW_TOKENS + seq_len, temperature=TEMPERATURE, 
    do_sample=True, pad_token_id=tokenizer.eos_token_id
)
generated_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Final time
metrics["total_time"] = time.time() - total_time
hours, remainder = divmod(metrics["total_time"], 3600)
minutes, seconds = divmod(remainder, 60)

print("\nGenerated text:", generated_output)
print("\n--- Metrics ---")
for chunk in metrics["chunks"]:
    print(f"Chunk {chunk['chunk']} - Time: {chunk['chunk_time']:.3f}s, Memory Used: {chunk['memory_used_mb']:.2f}MB")
    for idx, t in enumerate(chunk["layer_times"]):
        print(f"    Layer {idx} Time: {t:.3f}s")
print(f"Total Inference Time: {metrics['total_time']:.3f}s")
print(f"Total Inference Time (HH:MM:SS): {int(hours):02}:{int(minutes):02}:{int(seconds):02}")