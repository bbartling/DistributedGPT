import time
import torch
from model_utils import load_tokenizer_from_cache, load_model_from_cache, calculate_max_new_tokens

# Define prompt details
SYSTEM_MESSAGE = "You are a helpful assistant with expertise in HVAC systems, building automation, smart building IoT, and optimization."
QUESTION = (
    "I have a variable air volume (VAV) air handling unit (AHU) in a commercial building with 10 VAV boxes. "
    "Can you provide a pseudo-code algorithm to optimize the AHU leaving duct static pressure setpoint? "
    "The algorithm should use VAV box damper positions and implement a trim-and-respond strategy to achieve the lowest possible static pressure setpoint "
    "while maintaining an average damper position between 60% and 80% across all VAV boxes."
)


INSTRUCTION_TEMPLATE = (
    ">>CONTEXT<<\n{context}\n\n>>QUESTION<< {question}\n>>ANSWER<< "
)

INPUT_TEXT = INSTRUCTION_TEMPLATE.format(context=f"{SYSTEM_MESSAGE}", question=QUESTION)

# Parameters
DEFAULT_MAX_NEW_TOKENS = 300
TEMPERATURE = 0.6
TOP_P = 0.9
REPETITION_PENALTY = 2.0

# Paths for model and tokenizer
FALCON_1B_CACHE_DIRECTORY = (
    r"C:\\Users\\ben\\.cache\\huggingface\\hub\\models--ericzzz--falcon-rw-1b-instruct-openorca\\snapshots\\29cc70a0af3ac4826702ec46667931c0b0af340b"
)

# Load tokenizer and model
tokenizer = load_tokenizer_from_cache(FALCON_1B_CACHE_DIRECTORY)
model = load_model_from_cache(FALCON_1B_CACHE_DIRECTORY)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenize input
input_ids = tokenizer(INPUT_TEXT, return_tensors="pt").input_ids.to(device)

# Call the function to calculate MAX_NEW_TOKENS dynamically
MAX_NEW_TOKENS = calculate_max_new_tokens(
    input_text=INPUT_TEXT,
    tokenizer=tokenizer,
    model=model,
    max_new_tokens_default=DEFAULT_MAX_NEW_TOKENS,
    device=device
)

start_time = time.time()

# Generate text
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_length=MAX_NEW_TOKENS + input_ids.size(1),
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

total_time = time.time() - start_time

# Print the generated output
print("\nGenerated Output:", output_text)
print(f"Total Inference Time: {round(total_time,3)}s")