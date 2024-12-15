import psutil
from model_utils import load_tokenizer_from_cache, load_model_from_cache, calculate_chunks, split_and_save_model

# Define cache directory
CACHE_DIRECTORY = r"C:\Users\ben\.cache\huggingface\hub\models--meta-llama--Llama-3.1-8B\snapshots\d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
MODEL_SIZE_GIGS = 14.97  # Model size in GB (from PowerShell output)

# Load tokenizer and model
tokenizer = load_tokenizer_from_cache(CACHE_DIRECTORY)
print("Tokenizer loaded successfully!")

model = load_model_from_cache(CACHE_DIRECTORY)
print("Model loaded successfully!")

# Determine model architecture
if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
    layers = model.transformer.h
elif hasattr(model, "model") and hasattr(model.model, "layers"):
    layers = model.model.layers
else:
    raise AttributeError("Cannot determine the layers for splitting the model.")

# Calculate number of layers
num_layers = len(layers)
print(f"\nModel Configuration:")
print(f"- Number of transformer layers: {num_layers}")
print(f"- Hidden size: {model.config.hidden_size}")
print(f"- Vocabulary size: {model.config.vocab_size}")

# Calculate available memory
available_memory_gb = psutil.virtual_memory().available / (1024**3)
print(f"\nAvailable Memory: {available_memory_gb:.2f} GB")

# Calculate number of chunks
num_chunks = calculate_chunks(num_layers, MODEL_SIZE_GIGS, available_memory_gb)
print(f"\nModel will be split into {num_chunks} chunks.")

# Split and save model into parts
split_and_save_model(model, num_chunks)
