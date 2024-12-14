# DistributedGPT

This is a hobby project aimed at learning and experimentation. The primary goal is to explore running a large language model (LLM) entirely on `localhost` by breaking the model into smaller portions, loading them sequentially into memory for inference, and unloading them to optimize resource usage.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/distributedgpt.git
   cd distributedgpt
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv myenv
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\myenv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install torch transformers psutil
   ```

## Notes

- Check your system memory:
   ```powershell
   Get-CIMInstance Win32_OperatingSystem | Select-Object -Property *memory*
   ```

- To use a Hugging Face model, ensure you set your API key securely:
   ```powershell
   $env:HUGGINGFACE_API_KEY = "your_huggingface_api_key"
   ```

> **Important:** Do not include your Hugging Face API key directly in your scripts or repository.

- Activate your virtual environment before running any scripts:
   ```powershell
   .\myenv\Scripts\activate
   ```

- [x] Test on GPT2
- [x] Test on CodeLlama-7b-Instruct-hf on Py file creation abilities
- [ ] Test with a GPU that supports CUDA! 
- [ ] Test on Llama-3.2-1B
- [ ] Test on Llama-3.3-70B-Instruct with a [chat template like](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct#:~:text=Here%20is%20a%20quick%20example%20showing%20a%20single%20simple%20tool)
- [ ] Experiment with multi-agent(s) doing tasks for the human...
- [ ] Anything else? 🤔

## TODO 

## License

MIT