from model_utils import load_tokenizer_from_cache, load_model_from_cache, split_and_save_model
import gc

# Define cache directory and model parameters
FALCON_1B_CACHE_DIRECTORY = r"C:\Users\ben\.cache\huggingface\hub\models--ericzzz--falcon-rw-1b-instruct-openorca\snapshots\29cc70a0af3ac4826702ec46667931c0b0af340b"
FALCON_1B_MODEL_PARTS_DIR = "./1_b_model_parts"
FALCON_1B_HARDCODED_NUM_CHUNKS = 3

# Load tokenizer and model metadata
print("Loading tokenizer...")
tokenizer = load_tokenizer_from_cache(FALCON_1B_CACHE_DIRECTORY)
print("Tokenizer loaded successfully!")

print("Loading model metadata...")
print(f"Loading model from {FALCON_1B_CACHE_DIRECTORY}...")
model = load_model_from_cache(FALCON_1B_CACHE_DIRECTORY)
print("Model metadata loaded successfully!")

# Use hardcoded number of chunks
print(f"Splitting model into {FALCON_1B_HARDCODED_NUM_CHUNKS} chunks...")
split_and_save_model(model.state_dict(), FALCON_1B_HARDCODED_NUM_CHUNKS, directory=FALCON_1B_MODEL_PARTS_DIR)

# cleanup memory
gc.collect()


FALCON_7B_CACHE_DIRECTORY = r"C:\Users\ben\.cache\huggingface\hub\models--tiiuae--falcon-7b-instruct\snapshots\8782b5c5d8c9290412416618f36a133653e85285"
FALCON_7B_MODEL_PARTS_DIR = "./7_b_model_parts"
FALCON_7B_HARDCODED_NUM_CHUNKS = 8  # Example hard-coded value

# Load tokenizer and model metadata
print("Loading tokenizer...")
tokenizer = load_tokenizer_from_cache(FALCON_7B_CACHE_DIRECTORY)
print("Tokenizer loaded successfully!")

print("Loading model metadata...")
print(f"Loading model from {FALCON_7B_CACHE_DIRECTORY}...")
model = load_model_from_cache(FALCON_7B_CACHE_DIRECTORY)
print("Model metadata loaded successfully!")

# Use hardcoded number of chunks
print(f"Splitting model into {FALCON_7B_HARDCODED_NUM_CHUNKS} chunks...")
split_and_save_model(model.state_dict(), FALCON_7B_HARDCODED_NUM_CHUNKS, directory=FALCON_7B_MODEL_PARTS_DIR)


