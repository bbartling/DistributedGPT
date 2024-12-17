import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc


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
    """Load a specific model part from saved chunks."""
    part = torch.nn.ModuleList(model_part)
    state_dict = torch.load(file_path, map_location=device, weights_only=True)
    part.load_state_dict(state_dict)
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
    slopes = torch.arange(1, num_heads + 1, dtype=torch.float32, device=device)
    alibi = slopes.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, seq_len)
    return alibi
