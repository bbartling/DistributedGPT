# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from huggingface_hub import login


hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
if not hf_api_key:
    print("Error: HUGGINGFACE_API_KEY environment variable is not set.")
    exit(1)

try:
    login(hf_api_key)
    print("Successfully authenticated!")
except Exception as e:
    print(f"Error authenticating with Hugging Face Hub: {e}")


#tokenizer = AutoTokenizer.from_pretrained("ericzzz/falcon-rw-1b-instruct-openorca") 
#model = AutoModelForCausalLM.from_pretrained("ericzzz/falcon-rw-1b-instruct-openorca") 

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct")
