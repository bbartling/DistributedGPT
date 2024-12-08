from fastapi import FastAPI
from pydantic import BaseModel
from mpi4py import MPI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

"""
Usage:
mpirun -np 3 --hostfile ~/mpi_hostfile /home/ben/mpi_env/bin/python3 /home/ben/main.py
"""

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initialize FastAPI
app = FastAPI()

# Load tokenizer and define model parts for each rank
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

if rank == 0:
    model_full = AutoModelForCausalLM.from_pretrained("distilgpt2", low_cpu_mem_usage=True)
    embedding_layer = model_full.transformer.wte
    model_part = model_full.transformer.h[:2]
elif rank == 1:
    model_full = AutoModelForCausalLM.from_pretrained("distilgpt2", low_cpu_mem_usage=True)
    model_part = model_full.transformer.h[2:4]
elif rank == 2:
    model_full = AutoModelForCausalLM.from_pretrained("distilgpt2", low_cpu_mem_usage=True)
    model_part = model_full.transformer.h[4:6]

comm.barrier()  # Synchronize nodes

# Define FastAPI input schema for Rank 0
if rank == 0:
    class InputPrompt(BaseModel):
        prompt: str

    @app.post("/generate/")
    async def generate_text(input_prompt: InputPrompt):
        prompt = input_prompt.prompt

        # Convert prompt to embeddings
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        embeddings = embedding_layer(input_ids).to(torch.float32)

        # Forward through Rank 0's layers
        intermediate_activations = forward_through_layers(model_part, embeddings)
        comm.send(intermediate_activations, dest=1, tag=11)

        # Receive final activations from Rank 2
        final_activations = comm.recv(source=2, tag=13)

        # Generate text
        generated_output = model_full.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=150,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True,
        )

        return {"generated_text": tokenizer.decode(generated_output[0], skip_special_tokens=True)}

    if __name__ == "__main__":
        import uvicorn
        print("Starting FastAPI server...")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
else:
    # Worker nodes process activations
    def forward_through_layers(layers, input_tensor):
        """
        Manually pass the input tensor through a list of layers.
        This function simulates distributed forward propagation
        for a model split across nodes.
        """
        output_tensor = input_tensor
        for layer in layers:
            output_tensor = layer(output_tensor)[0]  # Pass through each layer
        return output_tensor

    while True:
        if rank == 1:
            activations = comm.recv(source=0, tag=11)
            intermediate_activations = forward_through_layers(model_part, activations)
            comm.send(intermediate_activations, dest=2, tag=12)

        elif rank == 2:
            activations = comm.recv(source=1, tag=12)
            final_activations = forward_through_layers(model_part, activations)
            comm.send(final_activations, dest=0, tag=13)
