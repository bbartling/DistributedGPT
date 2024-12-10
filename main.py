import os
import torch
from mpi4py import MPI
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

"""
$ mpirun -np 3 --hostfile ~/mpi_hostfile /home/ben/mpi_env/bin/python3 /home/ben/main.py
"""

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# FastAPI setup
app = FastAPI()

# Shared variables
model_name = "distilgpt2"
model_dir = "/home/ben/model_parts"

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model parts
if rank == 0:
    print(f"[Rank {rank}] Loading embedding layer and model part.")
    model_full = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    embedding_layer = model_full.transformer.wte
    model_part = torch.load(os.path.join(model_dir, "rank0.pt"))

elif rank == 1:
    model_file = os.path.join(model_dir, "rank1.pt")
    print(f"[Rank {rank}] Loading model part from {model_file}.")
    model_part = torch.load(model_file)

elif rank == 2:
    model_file = os.path.join(model_dir, "rank2.pt")
    print(f"[Rank {rank}] Loading model part from {model_file}.")
    model_part = torch.load(model_file)

comm.barrier()
print(f"[Rank {rank}] Model parts synchronized across all processes.")

# Forward function
def forward_through_layers(layers, input_tensor):
    """
    Pass input tensor through model layers with gradient calculation disabled.
    """
    try:
        with torch.no_grad():
            output_tensor = input_tensor
            for layer in layers:
                output_tensor = layer(output_tensor)[0]  # Handle tuple output
            return output_tensor
    except Exception as e:
        print(f"[Rank {rank}] Error during forward propagation: {e}")
        raise

# FastAPI Endpoint for Rank 0
if rank == 0:
    class InputPrompt(BaseModel):
        prompt: str

    @app.post("/generate/")
    async def generate_text(input_prompt: InputPrompt):
        prompt = input_prompt.prompt
        print(f"\n[Rank {rank}] Received prompt: '{prompt}'")

        try:
            with torch.no_grad():
                print(f"[Rank {rank}] Tokenizing the prompt and converting to embeddings.")
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                print(f"[Rank {rank}] Tokenized input IDs: {input_ids.tolist()}")

                embeddings = embedding_layer(input_ids).to(torch.float32)
                print(f"[Rank {rank}] Generated embeddings. Shape: {embeddings.shape}")

                # Forward through Rank 0's layers
                print(f"[Rank {rank}] Forwarding embeddings through Rank 0's model part.")
                activations = forward_through_layers(model_part, embeddings)
                print(f"[Rank {rank}] Completed forward pass. Activations shape: {activations.shape}")

            print(f"[Rank {rank}] Sending activations to Rank 1.")
            comm.send(activations, dest=1, tag=11)
            print(f"[Rank {rank}] Activations sent to Rank 1.")

            print(f"[Rank {rank}] Sending input_ids to Rank 2.")
            comm.send(input_ids, dest=2, tag=12)
            print(f"[Rank {rank}] input_ids sent to Rank 2.")

            # Receive generated text from Rank 2
            print(f"[Rank {rank}] Waiting to receive generated text from Rank 2.")
            generated_text = comm.recv(source=2, tag=13)
            print(f"[Rank {rank}] Received generated text from Rank 2: {generated_text}")

            return {"generated_text": generated_text}

        except Exception as e:
            print(f"[Rank {rank}] Error: {e}")
            raise HTTPException(status_code=500, detail="Text generation failed.")

    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# Rank 1 Processing
elif rank == 1:
    while True:
        try:
            print(f"[Rank {rank}] Waiting to receive activations from Rank 0.")
            activations = comm.recv(source=0, tag=11)
            print(f"[Rank {rank}] Received activations from Rank 0.")

            with torch.no_grad():
                print(f"[Rank {rank}] Forwarding activations through Rank 1's model part.")
                intermediate = forward_through_layers(model_part, activations)
                print(f"[Rank {rank}] Completed forward pass. Intermediate shape: {intermediate.shape}")

            print(f"[Rank {rank}] Sending intermediate activations to Rank 2.")
            comm.send(intermediate, dest=2, tag=12)
            print(f"[Rank {rank}] Intermediate activations sent to Rank 2.")

        except Exception as e:
            print(f"[Rank {rank}] Error during processing: {e}")

# Rank 2 Processing
elif rank == 2:
    while True:
        try:
            print(f"[Rank {rank}] Waiting to receive intermediate activations from Rank 1.")
            activations = comm.recv(source=1, tag=12)
            print(f"[Rank {rank}] Received intermediate activations from Rank 1.")

            print(f"[Rank {rank}] Waiting to receive input_ids from Rank 0.")
            input_ids = comm.recv(source=0, tag=12)
            print(f"[Rank {rank}] Received input_ids from Rank 0.")

            with torch.no_grad():
                print(f"[Rank {rank}] Forwarding activations through Rank 2's model part.")
                final_activations = forward_through_layers(model_part, activations)
                print(f"[Rank {rank}] Completed forward pass. Final activations shape: {final_activations.shape}")

                print(f"[Rank {rank}] Generating text using final activations.")
                generated_output = model_full.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=150,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True,
                )
                generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
                print(f"[Rank {rank}] Generated text: {generated_text}")

            print(f"[Rank {rank}] Sending generated text to Rank 0.")
            comm.send(generated_text, dest=0, tag=13)
            print(f"[Rank {rank}] Generated text sent to Rank 0.")

        except Exception as e:
            print(f"[Rank {rank}] Error during text generation: {e}")
