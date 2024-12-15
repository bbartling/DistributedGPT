# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from huggingface_hub import login

"""Try these???
OPT (Open Pretrained Transformer):

Developed by Meta as an open-source alternative to GPT-3.
Example: facebook/opt-6.7b
Falcon:

Competitive with GPT-3 for many tasks and highly efficient.
Example: tiiuae/falcon-7b
BLOOM:

A multilingual model developed by BigScience.
Example: bigscience/bloom
Mistral:

Focused on efficient, smaller-scale models with strong performance.
Example: mistralai/Mistral-7B

"""

hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
if not hf_api_key:
    print("Error: HUGGINGFACE_API_KEY environment variable is not set.")
    exit(1)

try:
    login(hf_api_key)
    print("Successfully authenticated!")
except Exception as e:
    print(f"Error authenticating with Hugging Face Hub: {e}")


#tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b") 
#model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct")
