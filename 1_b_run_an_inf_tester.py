import time
import torch
from model_utils import load_tokenizer_from_cache, load_model_from_cache, calculate_max_new_tokens
import os
import re

# Define prompt details
SYSTEM_MESSAGE = "You are a Python coding LLM fine tuned to make optimization scripts to save energy in HVAC systems."
QUESTION = (
    "Please make one pseudo code in Python of an algorithm to optimize variable supply fans in a large commercial building. "
    "The pseudo code needs to display all math and data structures. " 
    "The algorithm should adjust the duct static pressure setpoint based on air damper positions in the duct system to maintain the dampers approximately 70% open."
)

INSTRUCTION_TEMPLATE = (
    ">>CONTEXT<<\n{context}\n\n>>QUESTION<< {question}\n>>ANSWER<< "
)

INPUT_TEXT = INSTRUCTION_TEMPLATE.format(context=f"{SYSTEM_MESSAGE}", question=QUESTION)

# Parameters
DEFAULT_MAX_NEW_TOKENS = 300
TEMPERATURE = 1.0
TOP_P = 2.0
REPETITION_PENALTY = 1.1

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
print("Starting inference timer...!")

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

print("Inference completed.")
total_time = time.time() - start_time

# Print the generated output
print("Generated Output:", output_text)
print(f"Total Inference Time: {round(total_time,3)}s")
print("Temperature Setting:", TEMPERATURE)
print("Top Probability Setting:", TOP_P)
print("Repeition Penalty Setting:", REPETITION_PENALTY)


filename = f"falcon1b_fan_controls.py"

# Extract the Python code block from the output text
match = re.search(r"```python\n(.*?)\n```", output_text, re.DOTALL)
if match:
    python_code = match.group(1)
    # Save the extracted Python code to a file
    with open(filename, "w") as file:
        file.write("# Generated pseudo-code for optimizing variable supply fan controls\n")
        file.write("# Model context: Optimizing HVAC energy efficiency\n\n")
        file.write(python_code)
    print(f"Extracted Python code saved to file: {filename}")
else:
    print("No Python code block found in the output text.")
