import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

input_text = "What does an air handling unit do?"
MAX_NEW_TOKENS=150
TEMPERATURE=0.9

# Load pretrained model and tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
print("Number of transformer layers: ", len(model.transformer.h))

# Directory for saving model parts
os.makedirs("./model_parts", exist_ok=True)

# Split model into parts
part_0 = torch.nn.ModuleList(model.transformer.h[:2])
part_1 = torch.nn.ModuleList(model.transformer.h[2:4])
part_2 = torch.nn.ModuleList(model.transformer.h[4:6])

# Save each part
torch.save(part_0.state_dict(), "./model_parts/rank0.pt")
torch.save(part_1.state_dict(), "./model_parts/rank1.pt")
torch.save(part_2.state_dict(), "./model_parts/rank2.pt")

# Verify files are saved
assert os.path.exists("./model_parts/rank0.pt"), "Part 0 not saved!"
assert os.path.exists("./model_parts/rank1.pt"), "Part 1 not saved!"
assert os.path.exists("./model_parts/rank2.pt"), "Part 2 not saved!"
print("Model parts saved successfully!")

# Define a function to load model parts with appropriate layer structures
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_part(file_path, model_part):
    state_dict = torch.load(file_path, map_location=device, weights_only=True)
    part = torch.nn.ModuleList(model_part)
    part.load_state_dict(state_dict)
    return part.to(device)

# Load each part with its corresponding layers
loaded_part_0 = load_model_part("./model_parts/rank0.pt", model.transformer.h[:2])
loaded_part_1 = load_model_part("./model_parts/rank1.pt", model.transformer.h[2:4])
loaded_part_2 = load_model_part("./model_parts/rank2.pt", model.transformer.h[4:6])
print("Model parts loaded successfully!")

# Input string for generation
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# Generate text using the original model
print("\n--- Generating text with full model ---")
generated_output = model.generate(
    input_ids=input_ids,
    attention_mask=torch.ones_like(input_ids),
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_k=50,
    top_p=0.9,
    do_sample=True,
)
generated_text_full = tokenizer.decode(generated_output[0], skip_special_tokens=True)
print("Full model generated text:", generated_text_full)

# Manual text generation using modularized model
def modular_forward(input_ids, parts):
    # Embedding layer
    hidden_states = model.transformer.wte(input_ids) + model.transformer.wpe(torch.arange(input_ids.size(-1), device=device))
    
    # Process through modular parts
    for part in parts:
        for layer in part:
            # Check and handle tuple outputs
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]  # Extract the tensor
            hidden_states = layer(hidden_states)
    
    # Ensure hidden_states is a tensor before applying ln_f
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]
    
    hidden_states = model.transformer.ln_f(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits

def generate_text_modular(input_ids, parts, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
    generated_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = modular_forward(generated_ids, parts)
        next_token_logits = logits[:, -1, :] / temperature
        probabilities = torch.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probabilities, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        if next_token_id.item() == tokenizer.eos_token_id:
            break
    return generated_ids


print("\n--- Generating text with modular model ---")
parts = [loaded_part_0, loaded_part_1, loaded_part_2]
generated_output_modular = generate_text_modular(input_ids, parts)
generated_text_modular = tokenizer.decode(generated_output_modular[0], skip_special_tokens=True)
print("Modular model generated text:", generated_text_modular)

# Compare results
if generated_text_full == generated_text_modular:
    print("\nResults are identical!")
else:
    print("\nResults differ!")
