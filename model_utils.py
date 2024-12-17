import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import psutil


def load_tokenizer_from_cache(directory):
    """Load the tokenizer from the local cache directory."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")
    return AutoTokenizer.from_pretrained(directory)


def load_model_from_cache(directory):
    """Load the model from the local cache directory."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")
    return AutoModelForCausalLM.from_pretrained(directory)


def split_and_save_model(state_dict, num_chunks, directory="./model_parts"):
    """
    Splits a PyTorch model's state_dict into predefined chunks without loading the full model into memory.

    :param state_dict: The state_dict of the loaded PyTorch model.
    :param num_chunks: Number of chunks to split the model into.
    :param directory: Directory to save model parts.
    """
    print("Splitting the model into parts and saving...")
    # Create the output directory
    os.makedirs(directory, exist_ok=True)
    state_dict_keys = list(state_dict.keys())
    print(f"Total keys in state_dict: {len(state_dict_keys)}")
    keys_per_chunk = (len(state_dict_keys) + num_chunks - 1) // num_chunks  # Round up
    print(f"Keys per chunk: {keys_per_chunk}")
    # Split the state_dict into chunks and save them
    for i in range(0, len(state_dict_keys), keys_per_chunk):
        chunk_keys = state_dict_keys[i:i + keys_per_chunk]
        chunk_state_dict = {key: state_dict[key] for key in chunk_keys}
        # Save the chunk's state_dict
        chunk_path = os.path.join(directory, f"rank_{i // keys_per_chunk}.pt")
        torch.save(chunk_state_dict, chunk_path)
        print(f"Saved chunk {i // keys_per_chunk} with {len(chunk_keys)} keys.")
        # Clean up
        del chunk_state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Chunk {i} deleted from memory and garbage collected success.")
    print("\nModel parts saved successfully!")

def load_model_part(file_path, model_part, device):
    """
    Load a specific model part and adjust the state_dict keys if necessary.

    Args:
        file_path (str): Path to the saved model chunk.
        model_part (list): Subset of model layers to load the weights into.
        device (torch.device): Target device (CPU/GPU).
    """
    print(f"Loading model part from {file_path}...")
    # Wrap the model layers into a ModuleList
    part = torch.nn.ModuleList(model_part)
    # Load state_dict with weights_only=True for safety
    state_dict = torch.load(file_path, map_location=device, weights_only=True)
    # Adjust state_dict keys to remove unnecessary prefixes (if any)
    adjusted_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("transformer.h."):
            new_key = key.replace("transformer.h.", "")  # Adjust prefix
        else:
            new_key = key
        adjusted_state_dict[new_key] = value
    # Load adjusted state_dict into the model part
    try:
        part.load_state_dict(adjusted_state_dict, strict=False)
        print(f"Successfully loaded part from {file_path}.")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        raise
    return part.to(device)

def modular_forward(input_ids, parts, model, device):
    """Perform forward pass through modularized model."""
    hidden_states = model.transformer.wte(input_ids) + model.transformer.wpe(
        torch.arange(input_ids.size(-1), device=device)
    )
    for part in parts:
        for layer in part:
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]  # Extract the tensor
            hidden_states = layer(hidden_states)
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]
    hidden_states = model.transformer.ln_f(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits


def generate_text_modular(input_ids, parts, model, device, max_new_tokens, temperature):
    """Generate text using modularized model parts."""
    generated_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = modular_forward(generated_ids, parts, model, device)
        next_token_logits = logits[:, -1, :] / temperature
        probabilities = torch.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probabilities, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        if next_token_id.item() == model.config.eos_token_id:
            break
    return generated_ids


def build_alibi(batch_size, num_heads, seq_len, device):
    """
    Generate an Attention Linear Bias (ALiBi) tensor for scaled dot-product attention.

    ALiBi introduces positional bias into attention scores to help the model attend to positions 
    in a sequence in a linear manner, without requiring traditional positional embeddings.

    Args:
        batch_size (int): 
            The number of sequences in a batch.
        num_heads (int): 
            The number of attention heads in the model.
        seq_len (int): 
            The length of the input sequence (number of tokens).
        device (torch.device): 
            The target device (CPU or GPU) to store the tensor.

    Returns:
        torch.Tensor: 
            A 3D tensor of shape (batch_size, num_heads, seq_len) containing the positional slopes 
            repeated for each batch and head.
    """
    # Generate a range of slopes for each attention head
    slopes = torch.arange(1, num_heads + 1, dtype=torch.float32, device=device)
    # Expand and repeat slopes for batch_size and sequence length
    alibi = slopes.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, seq_len)
    return alibi


# Function to measure memory usage
def get_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6  # In MB
    else:
        return psutil.Process().memory_info().rss / 1e6  # In MB
    

def calculate_max_new_tokens(input_text, tokenizer, model, max_new_tokens_default, device):
    """
    Dynamically calculate the maximum new tokens for generation based on model's max sequence length.

    Args:
        input_text (str): The input text to tokenize.
        tokenizer (Tokenizer): The tokenizer used for tokenization.
        model (Model): The model with a max_position_embeddings attribute.
        max_new_tokens_default (int): The default desired number of new tokens.
        device (torch.device): Target device (CPU/GPU).

    Returns:
        int: The safe value for MAX_NEW_TOKENS.
    """
    print()
    info = """
    Max Sequence Length: The upper limit for total tokens (input + output). \n
    Tokens: Units into which text is split; their number depends on the tokenizer and input/output length.
    """
    print(info)
    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    # Calculate the number of input tokens
    num_input_tokens = input_ids.size(1)
    # Get the model's maximum sequence length
    max_sequence_length = model.config.max_position_embeddings
    print(f"Max Sequence Length for this model: {max_sequence_length}")
    # Calculate the available tokens for generation
    available_tokens = max_sequence_length - num_input_tokens
    print(f"Number of input tokens: {num_input_tokens}")
    print(f"Tokens available for generation: {available_tokens}")
    print(f"MAX_NEW_TOKENS dynamically set to: {available_tokens}")
    print()
    return available_tokens
