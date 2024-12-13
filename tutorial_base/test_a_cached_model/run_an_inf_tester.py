import os
import torch
from model_utils import (
    load_tokenizer_from_cache,
    load_model_from_cache,
    load_model_part,
    generate_text_modular
)

input_text = "Why are dogs sometimes brown in color?"
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.9

# Define cache directories
cache_directory = r"C:\Users\ben\.cache\huggingface\hub\models--distilgpt2\snapshots\2290a62682d06624634c1f46a6ad5be0f47f38aa"

# Load tokenizer and model from cache
tokenizer = load_tokenizer_from_cache(cache_directory)
print("Tokenizer loaded successfully!")

model = load_model_from_cache(cache_directory)
print("Model loaded successfully!")

# Verify number of layers
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

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load each part with its corresponding layers
loaded_part_0 = load_model_part("./model_parts/rank0.pt", model.transformer.h[:2], device)
loaded_part_1 = load_model_part("./model_parts/rank1.pt", model.transformer.h[2:4], device)
loaded_part_2 = load_model_part("./model_parts/rank2.pt", model.transformer.h[4:6], device)
print("Model parts loaded successfully!")

# Input string for generation
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# Generate text using modular model
print("\n--- Generating text with modular model ---")
parts = [loaded_part_0, loaded_part_1, loaded_part_2]
generated_output_modular = generate_text_modular(input_ids, parts, model, device, MAX_NEW_TOKENS, TEMPERATURE)
generated_text_modular = tokenizer.decode(generated_output_modular[0], skip_special_tokens=True)
print("Modular model generated text:", generated_text_modular)