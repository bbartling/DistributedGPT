# DistributedGPT

This is a hobby project aimed at learning and experimentation. The primary goal is to explore running a large language model (LLM) entirely on `localhost` by breaking the model into smaller portions, loading them sequentially into memory for inference, and unloading them to optimize resource usage. At moment we are expermenting with seeing if the LLM can come up with good pseudo code for HVAC systems optimizations.

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

## Download a model from Huggingface

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

Remove a model with PowerShell:

```powershell
Remove-Item -Recurse -Force "models--meta-llama--Llama-3.2-1B"
```

## Splitting up the model 

This Python file splits up the model into a hard coded defined number of files. With a text editor in `split_up_model.py` hard code in your model cache location and size:
```python
FALCON_7B_CACHE_DIRECTORY = r"C:\Users\ben\.cache\huggingface\hub\models--tiiuae--falcon-7b-instruct\snapshots\8782b5c5d8c9290412416618f36a133653e85285"
FALCON_7B_MODEL_PARTS_DIR = "./7_b_model_parts"
FALCON_7B_HARDCODED_NUM_CHUNKS = 8  # Example hard-coded value
```

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
  
- [x] **Test on Llama-3.1-8B and Llama-3.2-1B**  
  *Errors trying to use Llama models.*
  * Try again in the future! When splitting up the model I need research more about the deep learning architecture especially internal workings of the Python transformer library in how deep learning models are defined.

- [x] **Test on on a fine tuned model - ericzzz--falcon-rw-1b-instruct-openorca**
  * https://huggingface.co/ericzzz/falcon-rw-1b-instruct-openorca

- [x] **Test on on a fine tuned model - tiiuae/falcon-7b-instruct**
  * https://huggingface.co/tiiuae/falcon-7b-instruct
  * Notice its a different prompt template as compared to the 1b Falcon. This was found out by trial and error ☹️😒 but we got it working! 👍😊👌💪

- [ ] **Test on largest Falcon model???**

- [ ] **Test with a GPU that supports CUDA!**

- [ ] **Experiment with multi-agent(s) performing tasks for the human.**

- [ ] **Brainstorm further ideas or improvements.** 🤔

## TODO
Implement a conversation history and prompt input function in Py to push model to complete creations.
* https://youtu.be/v-dO88wU-jM?si=AErw3jewRs-oQ_zr
* https://langchain-ai.github.io/langgraph/concepts/memory/

Experiment with a coding LLM to get better pseudo code results.

## License

MIT