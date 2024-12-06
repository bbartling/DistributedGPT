from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mpi4py import MPI
from transformers import AutoTokenizer
import torch

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initialize FastAPI
app = FastAPI()

# Load tokenizer (assume all nodes have the model parts already loaded)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Rank-specific operations
if rank == 0:
    # Define the API input structure
    class InputPrompt(BaseModel):
        prompt: str

    # REST API endpoint to handle requests
    @app.post("/generate/")
    async def generate_text(input_prompt: InputPrompt):
        prompt = input_prompt.prompt

        # Convert prompt to embeddings
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        embeddings = AutoModelForCausalLM.from_pretrained("distilgpt2").transformer.wte(input_ids)
        embeddings = embeddings.to(torch.float32)

        # Forward through Rank 0's layers
        intermediate_activations = forward_through_layers(model_part, embeddings)
        comm.send(intermediate_activations, dest=1, tag=11)

        # Receive final activations from Rank 2
        final_activations = comm.recv(source=2, tag=13)

        # Generate text
        generated_output = tokenizer.decode(
            final_activations, skip_special_tokens=True
        )
        return {"generated_text": generated_output}

    # Launch FastAPI (only on Boss Pi)
    if __name__ == "__main__":
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    # Worker nodes logic remains unchanged
    while True:
        if rank == 1:
            activations = comm.recv(source=0, tag=11)
            intermediate_activations = forward_through_layers(model_part, activations)
            comm.send(intermediate_activations, dest=2, tag=12)

        elif rank == 2:
            activations = comm.recv(source=1, tag=12)
            final_activations = forward_through_layers(model_part, activations)
            comm.send(final_activations, dest=0, tag=13)

