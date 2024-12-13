import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_tokenizer_from_cache(directory):
    """Load the tokenizer from the local cache directory."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")
    
    expected_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']
    for file in expected_files:
        if not os.path.exists(os.path.join(directory, file)):
            raise FileNotFoundError(f"Expected file '{file}' not found in directory '{directory}'.")
    
    return AutoTokenizer.from_pretrained(directory)


def load_model_from_cache(directory):
    """Load the model from the local cache directory."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")
    
    expected_files = ['config.json', 'model.safetensors', 'generation_config.json']
    for file in expected_files:
        if not os.path.exists(os.path.join(directory, file)):
            raise FileNotFoundError(f"Expected file '{file}' not found in directory '{directory}'.")
    
    return AutoModelForCausalLM.from_pretrained(directory, low_cpu_mem_usage=True)


def load_model_part(file_path, model_part, device):
    """Load a specific model part with `weights_only=True`."""
    state_dict = torch.load(file_path, map_location=device, weights_only=True)
    part = torch.nn.ModuleList(model_part)
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
