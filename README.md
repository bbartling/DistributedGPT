# DistributedGPT

This is a hobby project aimed at learning and experimentation. The primary goal is to explore running a large language model (LLM) entirely on `localhost` by breaking the model into smaller portions, loading them sequentially into memory for inference, and unloading them to optimize resource usage.

## Installation

1. Clone the repository:

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install torch transformers psutil
   ```

## Notes

- Check your system memory on Windows in PowerShell:
   ```powershell
   Get-CIMInstance Win32_OperatingSystem | Select-Object -Property *memory*
   ```

- On Windows in PowerShell set your API key as a temporary operating system variable:
   ```powershell
   $env:HUGGINGFACE_API_KEY = "your_huggingface_api_key"
   ```
## Download models from Hugging Face

Use the `downloader.py` and modify as necessary to grab a model from Huggingface. Models on my PC are cached in this location `C:\Users\ben\.cache\huggingface\hub` and I can use PowerShell to see how big they are in gigabytes.
This data is used as a setting another Py file that divides up the model into different parts. Do a change directory or a `Set-Location` as shown below:

```powershell
# Change to the Hugging Face cache directory
Set-Location -Path 'C:\Users\ben\.cache\huggingface\hub'
```

Loop through each folder and print out model sizes which also maybe available somewhere on Huggingface as well for each model:

```powershell
# Loop through directories and calculate their size
Get-ChildItem -Directory | ForEach-Object {
    $sizeBytes = (Get-ChildItem -Path $_.FullName -Recurse | Where-Object { $_.PSIsContainer -eq $false } | Measure-Object -Property Length -Sum).Sum
    $sizeGB = [math]::Round($sizeBytes / 1GB, 2)
    Write-Output "Model: $($_.Name), Size: $sizeGB GB"
}
```

This prints out as:

```powershell
Model: models--distilgpt2, Size: 0.33 GB
Model: models--meta-llama--CodeLlama-7b-Python-hf, Size: 12.55 GB
Model: models--meta-llama--Llama-3.1-8B, Size: 14.97 GB
Model: models--meta-llama--Llama-3.2-1B, Size: 2.31 GB
```

## Remove a model with PowerShell

```powershell
Remove-Item -Recurse -Force "models--meta-llama--Llama-3.2-1B"
```

## Splitting up the model 

With a text editor in `split_up_model.py` hard code in your model cache location and size:
```python
# Define cache directory
CACHE_DIRECTORY = r"C:\Users\ben\.cache\huggingface\hub\models--meta-llama--Llama-3.2-1B\snapshots\4e20de362430cd3b72f300e6b0f18e50e7166e08"
MODEL_SIZE_GIGS = 2.31  # Model size in GB (from PowerShell output)
```

Then run `split_up_model.py`:

```powershell
> python .\split_up_model.py
Tokenizer loaded successfully!
Model loaded successfully!

Model Configuration:
- Number of transformer layers: 16
- Hidden size: 2048
- Vocabulary size: 128256

Available Memory: 21.05 GB

Model will be split into 3 chunks.
Saved part 0
Saved part 1
Saved part 2

Model parts saved successfully!
```

The script is designed to efficiently handle large models by providing detailed information and optimizing their usage in constrained environments. It begins by printing the model's configuration details, including the number of transformer layers, hidden size, and vocabulary size, to give a clear understanding of the model's architecture. Next, it calculates the available memory in gigabytes using the `psutil` library, enabling the script to estimate how much of the model can fit into memory at any given time.

To manage large models, the `calculate_chunks` function divides the model into at least three parts, with more parts created if the model size exceeds available memory. This calculation takes into account the total number of transformer layers (`num_layers`), the model size in gigabytes (`MODEL_SIZE_GIGS`), and the available system memory (`available_memory_gb`). Each chunk contains a subset of the model's layers, with the chunk sizes optimized for memory efficiency.

Finally, the `split_and_save_model` function splits the model layers into these calculated chunks and saves each chunk as a separate file in a specified directory (default: `./model_parts`). This process ensures that the model can be loaded incrementally or distributed across systems with limited memory, facilitating efficient inference or fine-tuning workflows.

## Run an inference test

Now that the model is split up into parts locally in the `model_parts_ directory` then we can try running the `run_an_inf_tester.py`:

```powershell
> python .\run_an_inf_tester.py
Tokenizer loaded successfully!
Model loaded successfully!
```

## Lesson Learned So Far...

- [x] **Test on GPT2**  
  *Works, but it’s GPT2!*
  
- [x] **Test on Llama-3.2-1B**  
  *Errors trying to use Llama models.*
  
- [x] **Test on Llama-3.1-8B**  
  *Errors trying to use Llama models.*
  
- [x] **Test on `tiiuae/falcon-7b`**  
  *Works, but inference results are odd and the computer is very laggy. Planning to try the instruct version for better memory handling.*
  *Try the fine tuned instruct version and research a prompt template to use to be built into the `run_an_inf_tester.py`.*

- [ ] **Test on `tiiuae/falcon-7b-instruct`**

- [ ] **Test with a GPU that supports CUDA!**

- [ ] **Experiment with multi-agent(s) performing tasks for the human.**

- [ ] **Brainstorm further ideas or improvements.** 🤔


## License

MIT