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
  
- [x] **Test on on a fine tuned model - `ericzzz--falcon-rw-1b-instruct-openorca`**
  * https://huggingface.co/ericzzz/falcon-rw-1b-instruct-openorca

  ```python
  # Define system message and structured prompt
  SYSTEM_MESSAGE = "You are a helpful assistant with expertise in HVAC systems."
  INSTRUCTION = "Explain in detail what an air handling unit does in large commercial buildings..."
  INPUT_TEXT = f"<SYS> {SYSTEM_MESSAGE} <INST> {INSTRUCTION} <RESP> "
  ```
  * Total Inference Time: 80.889s

  Details per chunk:

  ```powershell
  --- Metrics ---
  Chunk 0 - Time: 1.162s, Memory Used: 1670.56MB
      Layer 0 Time: 0.021s
      Layer 1 Time: 0.017s
      Layer 2 Time: 0.017s
      Layer 3 Time: 0.018s
      Layer 4 Time: 0.016s
      Layer 5 Time: 0.017s
      Layer 6 Time: 0.017s
      Layer 7 Time: 0.018s
  Chunk 1 - Time: 0.665s, Memory Used: -29.34MB
      Layer 0 Time: 0.020s
      Layer 1 Time: 0.018s
      Layer 2 Time: 0.018s
      Layer 3 Time: 0.016s
      Layer 4 Time: 0.016s
      Layer 5 Time: 0.014s
      Layer 6 Time: 0.015s
      Layer 7 Time: 0.014s
  Chunk 2 - Time: 0.796s, Memory Used: 445.60MB
  ```
  Inference results:
  ```powershell
  1. Air circulation: The air handling unit's primary function is to distribute and circulate air throughout the building. This is achieved by controlling the airflow, which ensures that the air is evenly distributed throughout the space.

  2. Temperature and humidity control: The air handling unit also controls the temperature and humidity levels within the space. By regulating the temperature, you can maintain a comfortable indoor temperature that is suitable for the comfort of occupants. Similarly, by controlling the humidity levels, you can prevent the growth of mold and other contaminants, which can be detrimental to the health of building occupants.

  3. Energy efficiency: Many air handling units are designed to be energy-efficient, which means they consume less energy than traditional systems. This helps to reduce the building's overall operating costs and contributes to a more sustainable environment.

  4. Occupant comfort: The air handling unit's primary purpose is to maintain a comfortable indoor environment for the building's occupants. This includes ensuring that the temperature and humidity levels are consistent, providing a consistent level of comfort for everyone in the building.

  In summary, the air handling unit in large commercial buildings is an essential component that plays a crucial role in maintaining the well-being of building occupants while ensuring efficient and effective air circulation and temperature control.
  ```
  * Conclusion: Hey that is not that bad! 👍😊👌💪 80 seconds is a little slow but good enough for not even a GPU and no lag on my computer while I do other tasks. 🎯🙌😎

- [ ] **Test on on a fine tuned model - `tiiuae/falcon-7b-instruct`**
  * https://huggingface.co/tiiuae/falcon-7b-instruct

- [ ] **Test on largest Falcon model???**

- [ ] **Test with a GPU that supports CUDA!**

- [ ] **Experiment with multi-agent(s) performing tasks for the human.**

- [ ] **Brainstorm further ideas or improvements.** 🤔


## License

MIT