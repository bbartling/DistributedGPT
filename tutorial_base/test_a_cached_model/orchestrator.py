import subprocess
import os

# Path to the Python executable in the virtual environment
VENV_PYTHON = os.path.join(os.getcwd(), "myenv", "Scripts", "python.exe")

# Define the workers and their respective model parts
WORKERS = [
    {"script": "worker_0.py", "model_part_id": 0},
    {"script": "worker_1.py", "model_part_id": 1},
    {"script": "worker_2.py", "model_part_id": 2},
]

def run_worker(worker):
    try:
        # Run the worker script with the virtual environment's Python executable
        result = subprocess.run(
            [VENV_PYTHON, worker["script"]],
            check=True,  # Raise an error if the subprocess fails
            capture_output=True,  # Capture standard output and errors
            text=True,  # Decode outputs as text
        )
        print(f"Worker {worker['script']} completed successfully!")
        print(f"Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Worker {worker['script']} failed!")
        print(f"Error Output:\n{e.stderr}")

if __name__ == "__main__":
    # Ensure the worker scripts exist in the current directory
    for worker in WORKERS:
        if not os.path.exists(worker["script"]):
            print(f"Worker script {worker['script']} not found!")
            continue
        print(f"Running {worker['script']} with virtual environment...")
        run_worker(worker)
