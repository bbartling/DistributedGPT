from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mpi4py import MPI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# FastAPI setup
app = FastAPI()

# Load tokenizer on all ranks
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model parts
if rank == 0:
    print(f"Rank {rank}: Loading first part of the model.")
    model_full = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    embedding_layer = model_full.transformer.wte  # Token embedding layer
    print(f"Model length: {len(model_full.transformer.h)} layers")
    model_part = model_full.transformer.h[:2]
elif rank == 1:
    print(f"Rank {rank}: Loading second part of the model.")
    model_full = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    model_part = model_full.transformer.h[2:4]
elif rank == 2:
    print(f"Rank {rank}: Loading third part of the model.")
    model_full = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    model_part = model_full.transformer.h[4:6]

comm.barrier()
print(f"Process {rank}: Synchronized.")

# Function to forward activations
def forward_through_layers(layers, input_tensor):
    try:
        output_tensor = input_tensor
        for layer in layers:
            output_tensor = layer(output_tensor)[0]  # Handle tuple output
        return output_tensor
    except Exception as e:
        print(f"Rank {rank}: Error during forward propagation - {e}")
        raise

# FastAPI endpoint for Rank 0
if rank == 0:
    class InputPrompt(BaseModel):
        prompt: str

    @app.post("/generate/")
    async def generate_text(input_prompt: InputPrompt):
        prompt = input_prompt.prompt
        print(f"Rank {rank}: Received prompt: {prompt}")

        try:
            # Convert prompt to embeddings
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            embeddings = embedding_layer(input_ids).to(torch.float32)

            # Forward through Rank 0's layers
            activations = forward_through_layers(model_part, embeddings)
            comm.send(activations, dest=1, tag=11)
            comm.send(input_ids, dest=2, tag=12)  # Send input IDs directly to Rank 2

            print(f"Rank {rank}: Waiting for generated text from Rank 2.")
            generated_text = comm.recv(source=2, tag=13)
            return {"generated_text": generated_text}
        except Exception as e:
            print(f"Rank {rank}: Error - {e}")
            raise HTTPException(status_code=500, detail="Text generation failed.")

    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# Rank 1 processing
elif rank == 1:
    while True:
        try:
            activations = comm.recv(source=0, tag=11)
            activations = forward_through_layers(model_part, activations)
            comm.send(activations, dest=2, tag=12)
        except Exception as e:
            print(f"Rank {rank}: Error during processing - {e}")

# Rank 2 processing
elif rank == 2:
    while True:
        try:
            activations = comm.recv(source=1, tag=12)
            input_ids = comm.recv(source=0, tag=12)

            # Forward through layers
            final_activations = forward_through_layers(model_part, activations)

            # Generate text using final activations
            with torch.no_grad():
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
            comm.send(generated_text, dest=0, tag=13)
        except Exception as e:
            print(f"Rank {rank}: Error during text generation - {e}")
