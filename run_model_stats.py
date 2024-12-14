import psutil
from model_utils import load_tokenizer_from_cache, load_model_from_cache, calculate_chunks, split_and_save_model

# Define cache directory
cache_directory = r"C:\Users\ben\.cache\huggingface\hub\models--distilgpt2\snapshots\2290a62682d06624634c1f46a6ad5be0f47f38aa"

# Load tokenizer and model
tokenizer = load_tokenizer_from_cache(cache_directory)
print("Tokenizer loaded successfully!")

model = load_model_from_cache(cache_directory)
print("Model loaded successfully!")

# Display model stats
num_layers = len(model.transformer.h)
print(f"\nModel Configuration:")
print(f"- Number of transformer layers: {num_layers}")
print(f"- Hidden size: {model.config.hidden_size}")
print(f"- Vocabulary size: {model.config.vocab_size}")

# Calculate available memory
available_memory_gb = psutil.virtual_memory().available / (1024**3)
print(f"\nAvailable Memory: {available_memory_gb:.2f} GB")

# Calculate number of chunks
model_size_gb = 0.31  # Replace with actual model size if available
num_chunks = calculate_chunks(num_layers, model_size_gb, available_memory_gb)
print(f"\nModel will be split into {num_chunks} chunks.")

# Split and save model into parts
split_and_save_model(model, num_chunks)
