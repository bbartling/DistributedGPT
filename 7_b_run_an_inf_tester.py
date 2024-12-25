import os, re
import torch
from model_utils import (
    load_model_part,
    load_tokenizer_from_cache,
    load_model_from_cache,
    build_alibi,
    calculate_max_new_tokens
)
import gc
import time


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
    "You are a Python coding LLM fine tuned to make optimization scripts to save energy in HVAC systems."
)
QUESTION = (
    "Please make one pseudo code in Python of an algorithm to optimize variable supply fans in a large commercial building. "
    "The pseudo code needs to display all math and data structures. " 
    "The algorithm should adjust the duct static pressure setpoint based on air damper positions in the duct system to maintain the dampers approximately 70% open."
)

# Assemble the Input Text
INPUT_TEXT = INSTRUCTION_TEMPLATE.format(context=CONTEXT, question=QUESTION)

# Parameters
DEFAULT_MAX_NEW_TOKENS = 300
TEMPERATURE = 1.0
TOP_P = 1.0
REPETITION_PENALTY = 1.1

# Paths for model and tokenizer
FALCON_7B_CACHE_DIRECTORY = (
    r"C:\Users\ben\.cache\huggingface\hub\models--tiiuae--falcon-7b-instruct\snapshots\8782b5c5d8c9290412416618f36a133653e85285"
)
FALCON_7B_MODEL_PARTS_DIR = "./7_b_model_parts"

print("Model Parts Directory Content:")
for f in os.listdir(FALCON_7B_MODEL_PARTS_DIR):
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

start_time = time.time()
print("Starting inference timer...!")

with torch.no_grad():
    hidden_states = model.transformer.word_embeddings(input_ids)
    batch_size, seq_len = input_ids.shape
    attention_mask = torch.ones((batch_size, seq_len), device=device)[:, None, None, :]
    alibi = build_alibi(batch_size, model.config.num_attention_heads, seq_len, device)

    for i in range(num_chunks):
        file_path = os.path.join(FALCON_7B_MODEL_PARTS_DIR, f"rank_{i}.pt")
        part_layers = layers[i * layers_per_chunk : (i + 1) * layers_per_chunk]
        part = load_model_part(file_path, part_layers, device)

        for layer in part:
            output = layer(hidden_states, alibi=alibi, attention_mask=attention_mask)
            hidden_states = output[0] if isinstance(output, tuple) else output

        del part
        gc.collect()
        torch.cuda.empty_cache()

    hidden_states = model.transformer.ln_f(hidden_states)
    logits = model.lm_head(hidden_states)
    output_ids = model.generate(
        input_ids, max_length=MAX_NEW_TOKENS + seq_len, temperature=TEMPERATURE, 
        top_p=TOP_P, repetition_penalty=REPETITION_PENALTY,
        do_sample=True, pad_token_id=tokenizer.eos_token_id
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)


# Final time
time_delta = time.time() - start_time
print("\n--- Metrics ---")
print(f"Total Inference Time: {time_delta:.3f}s")
print("\nGenerated Output:", output_text)

filename = f"falcon7b_fan_controls.py"

# Extract the Python code block from the output text
match = re.search(r"```python\n(.*?)\n```", output_text, re.DOTALL)
if match:
    python_code = match.group(1)
    # Save the extracted Python code to a file
    with open(filename, "w") as file:
        file.write("# Generated pseudo-code for optimizing variable supply fan controls\n")
        file.write("# Model context: Optimizing HVAC energy efficiency\n\n")
        file.write(python_code)
    print(f"Extracted Python code saved to file: {filename}")
else:
    print("No Python code block found in the output text.")