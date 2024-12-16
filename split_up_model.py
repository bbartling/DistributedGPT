import psutil
from model_utils import load_tokenizer_from_cache, load_model_from_cache, calculate_chunks, split_and_save_model

# Define cache directory
CACHE_DIRECTORY = r"C:\Users\ben\.cache\huggingface\hub\models--ericzzz--falcon-rw-1b-instruct-openorca\snapshots\29cc70a0af3ac4826702ec46667931c0b0af340b"
MODEL_SIZE_GIGS = 13.45  # Not needed is passing in HARDCODED_NUM_CHUNKS value comes from PowerShell output
HARDCODED_NUM_CHUNKS = 3  # Example hard-coded value

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


# Use a hard-coded value for num_chunks or calculate dynamically
split_and_save_model(
    model,
    num_chunks=HARDCODED_NUM_CHUNKS,  # Overrides memory calculation
    directory="./model_parts"
)

"""
Example of splitting up model based on memory

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

split_and_save_model(
    model,
    num_chunks=None,  # Allows dynamic calculation
    model_size_gb=MODEL_SIZE_GIGS,
    available_memory_gb=available_memory_gb,
    directory="./model_parts"
)
"""