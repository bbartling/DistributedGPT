import os
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from model_utils import (
    load_tokenizer_from_cache,
    load_model_from_cache,
    load_model_part,
    generate_text_modular
)

# Web app setup
app = FastAPI()

# Constants
CACHE_DIRECTORY = r"C:\Users\ben\.cache\huggingface\hub\models--distilgpt2\snapshots\2290a62682d06624634c1f46a6ad5be0f47f38aa"
MODEL_DIR = "./model_parts"
TOKENIZER_DIR = CACHE_DIRECTORY

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.9

# Load tokenizer and model from cache
tokenizer = load_tokenizer_from_cache(CACHE_DIRECTORY)
print("Tokenizer loaded successfully!")

model = load_model_from_cache(CACHE_DIRECTORY)
print("Model loaded successfully!")

# Save model parts to disk
os.makedirs(MODEL_DIR, exist_ok=True)
part_0 = torch.nn.ModuleList(model.transformer.h[:2])
part_1 = torch.nn.ModuleList(model.transformer.h[2:4])
part_2 = torch.nn.ModuleList(model.transformer.h[4:6])

torch.save(part_0.state_dict(), os.path.join(MODEL_DIR, "rank0.pt"))
torch.save(part_1.state_dict(), os.path.join(MODEL_DIR, "rank1.pt"))
torch.save(part_2.state_dict(), os.path.join(MODEL_DIR, "rank2.pt"))

print("Model parts saved successfully!")

# FastAPI endpoints
@app.get("/get_tokenizer")
async def get_tokenizer():
    """Return the tokenizer directory path."""
    if not Path(TOKENIZER_DIR).exists():
        raise HTTPException(status_code=404, detail="Tokenizer not found!")
    return {"path": TOKENIZER_DIR}

@app.get("/get_model_part/{part_id}")
async def get_model_part(part_id: int):
    """Return a specific model part."""
    part_file = Path(MODEL_DIR) / f"rank{part_id}.pt"
    if not part_file.exists():
        raise HTTPException(status_code=404, detail=f"Model part {part_id} not found!")
    state_dict = torch.load(part_file, map_location="cpu", weights_only=True)  # Explicitly set weights_only=True
    return {"model_part": state_dict}


@app.post("/generate_text")
async def generate_text(input_text: str):
    """Generate text using modular model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model parts dynamically
    loaded_part_0 = load_model_part(os.path.join(MODEL_DIR, "rank0.pt"), model.transformer.h[:2], device)
    loaded_part_1 = load_model_part(os.path.join(MODEL_DIR, "rank1.pt"), model.transformer.h[2:4], device)
    loaded_part_2 = load_model_part(os.path.join(MODEL_DIR, "rank2.pt"), model.transformer.h[4:6], device)
    parts = [loaded_part_0, loaded_part_1, loaded_part_2]

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    generated_output_modular = generate_text_modular(input_ids, parts, model, device, MAX_NEW_TOKENS, TEMPERATURE)
    generated_text_modular = tokenizer.decode(generated_output_modular[0], skip_special_tokens=True)

    return {"generated_text": generated_text_modular}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
