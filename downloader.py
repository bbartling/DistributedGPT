# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from huggingface_hub import login

#os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
if not hf_api_key:
    print("Error: HUGGINGFACE_API_KEY environment variable is not set.")
    exit(1)

try:
    login(hf_api_key)
    print("Successfully authenticated!")
except Exception as e:
    print(f"Error authenticating with Hugging Face Hub: {e}")


#tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-Python-hf")
#model = AutoModelForCausalLM.from_pretrained("meta-llama/CodeLlama-7b-Python-hf")


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")