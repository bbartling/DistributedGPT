from mpi4py import MPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Ensure that we have exactly 3 processes
if size != 3:
    if rank == 0:
        print("This script requires exactly 3 MPI processes.")
    exit()

# Load tokenizer on all nodes
model_name = "quantized_llama_3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if rank == 0:
    # Rank 0: Load the first part of the quantized model
    print(f"Rank {rank}: Loading the first part of the quantized model.")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True,  # Enable 8-bit quantization
    )
    embedding_layer = model.transformer.wte  # Token embedding layer
    model_part = model.transformer.h[:2]  # First 2 layers
elif rank == 1:
    # Rank 1: Load the second part of the quantized model
    print(f"Rank {rank}: Loading the second part of the quantized model.")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True,
    )
    model_part = model.transformer.h[2:4]  # Next 2 layers
elif rank == 2:
    # Rank 2: Load the third part of the quantized model
    print(f"Rank {rank}: Loading the third part of the quantized model.")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True,
    )
    model_part = model.transformer.h[4:6]  # Last 2 layers

# Synchronize nodes
comm.barrier()

def forward_through_layers(layers, input_tensor):
    """Manually pass the input tensor through a list of layers."""
    output_tensor = input_tensor
    for layer in layers:
        output_tensor = layer(output_tensor)[0]  # Pass through each layer
    return output_tensor

# Main processing loop for Rank 0
if rank == 0:
    while True:
        # Prompt the user for input
        prompt = input("\nEnter your prompt (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            print("Exiting...")
            break

        # Convert prompt to embeddings
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # Token IDs
        attention_mask = torch.ones_like(input_ids)  # Create attention mask
        print(f"Rank {rank}: Generating embeddings.")
        embeddings = embedding_layer(input_ids)  # Convert token IDs to embeddings
        embeddings = embeddings.to(torch.float32)  # Convert embeddings to float32
        print(f"Rank {rank}: Forwarding embeddings through first part of the model.")
        intermediate_activations = forward_through_layers(model_part, embeddings)
        print(f"Rank {rank}: Sending activations to Rank 1.")
        comm.send(intermediate_activations, dest=1, tag=11)

        # Receive final activations from Rank 2
        print(f"Rank {rank}: Receiving final activations from Rank 2.")
        final_activations = comm.recv(source=2, tag=13)
        final_activations = final_activations.to(torch.float32)  # Convert to float32

        # Generate text using the final activations
        print(f"Rank {rank}: Generating text from final activations.")
        generated_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  # Pass attention mask explicitly
            max_new_tokens=150,  # Maximum length of the generated text
            temperature=0.8,  # Sampling temperature
            top_k=50,  # Top-k sampling
            top_p=0.9,  # Top-p nucleus sampling
            do_sample=True,  # Enable sampling
        )
        print(f"\nGenerated Text: {tokenizer.decode(generated_output[0], skip_special_tokens=True)}")

# Process input on other ranks
if rank == 1:
    while True:
        print(f"Rank {rank}: Receiving activations from Rank 0.")
        activations = comm.recv(source=0, tag=11)
        activations = activations.to(torch.float32)
        print(f"Rank {rank}: Forwarding activations through second part of the model.")
        intermediate_activations = forward_through_layers(model_part, activations)
        print(f"Rank {rank}: Sending activations to Rank 2.")
        comm.send(intermediate_activations, dest=2, tag=12)

if rank == 2:
    while True:
        print(f"Rank {rank}: Receiving activations from Rank 1.")
        activations = comm.recv(source=1, tag=12)
        activations = activations.to(torch.float32)
        print(f"Rank {rank}: Forwarding activations through third part of the model.")
        final_activations = forward_through_layers(model_part, activations)
        print(f"Rank {rank}: Sending final activations back to Rank 0.")
        comm.send(final_activations, dest=0, tag=13)
