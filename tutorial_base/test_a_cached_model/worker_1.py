import os
import requests
import torch
from transformers import AutoTokenizer

MODEL_SERVER_URL = "http://127.0.0.1:8000"
MODEL_PART_ID = 1  # Change this per worker
TOKENIZER_DIR = "./worker_tokenizer"

def fetch_tokenizer():
    response = requests.get(f"{MODEL_SERVER_URL}/get_tokenizer")
    response.raise_for_status()
    tokenizer_data = response.json()
    tokenizer_path = tokenizer_data["path"]
    
    # Debugging: Check the contents of the tokenizer directory
    print("Checking directory contents for:", tokenizer_path)
    print(os.listdir(tokenizer_path))  # This should list tokenizer files
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def fetch_model_part(part_id):
    response = requests.get(f"{MODEL_SERVER_URL}/get_model_part/{part_id}")
    response.raise_for_status()
    state_dict = response.json()["model_part"]
    part = torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(d_model=768, nhead=12)])  # Update layer config as needed
    part.load_state_dict(state_dict)
    print(f"Model part {part_id} fetched and loaded.")
    return part

if __name__ == "__main__":
    tokenizer = fetch_tokenizer()
    model_part = fetch_model_part(MODEL_PART_ID)
