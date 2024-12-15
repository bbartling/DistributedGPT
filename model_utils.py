import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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


def calculate_chunks(num_layers, model_size_gb, available_memory_gb):
    """Calculate number of chunks based on available memory."""
    memory_per_layer_gb = model_size_gb / num_layers
    layers_per_chunk = max(1, int(available_memory_gb / memory_per_layer_gb))
    chunks = max(3, (num_layers + layers_per_chunk - 1) // layers_per_chunk)  # Ensure at least 3 chunks
    return chunks


def split_and_save_model(model, num_chunks, directory="./model_parts"):
    print("Spliting the model into parts and save....")

    import os
    import torch
    
    os.makedirs(directory, exist_ok=True)

    # Generalized layer selection logic
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise AttributeError("Cannot determine the layers for splitting the model.")

    num_layers = len(layers)
    print("num_layers: ",num_layers)

    layers_per_chunk = (num_layers + num_chunks - 1) // num_chunks  # Round up
    print("layers_per_chunk: ",layers_per_chunk)

    # Split and save chunks
    for i in range(0, num_layers, layers_per_chunk):
        part = torch.nn.ModuleList(layers[i:i + layers_per_chunk])
        torch.save(part.state_dict(), os.path.join(directory, f"rank{i // layers_per_chunk}.pt"))
        print(f"Saved part {i // layers_per_chunk}")

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
