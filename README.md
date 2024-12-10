# DistributedGPT

This is a hobby project aimed at learning and experimentation. The primary goal is to explore whether cluster computing can be used to run a large language model (LLM) across multiple computers, ultimately finding a cost-effective way to operate an LLM on my own hardware. For this experiment, I’m using three Raspberry Pi devices, which were repurposed from previous projects and cost nothing additional. 

Currently, I’m running a distilled version of GPT-2, with each Raspberry Pi loading a portion of the model. While the setup is functional, it’s extremely slow, and the model outputs are practically useless—but it works! This sparks the question: why couldn’t this approach be scaled up using three affordable, used gaming PCs equipped with GPUs, such as NVIDIA RTX 3060s? This could potentially make cluster computing a viable and low-cost way to run an LLM entirely on local hardware.

## MPI

I've been reading *The Art of HPC* by Victor Eijkhout, an excellent resource for learning about cluster computing, parallel computing, and high-performance computing concepts.
* https://theartofhpc.com/

HTML online free version for a book about the science of computing, `The Art of HPC`, volume 1 by Victor Eijkhout:
* https://theartofhpc.com/istc/index.html

## Architecture
Cluster computing relies on networking to enable communication between computers. In my current setup, each Raspberry Pi uses a dynamic IP address assigned via DHCP by a router. For now, this is sufficient, as the project is more of a science experiment focused on rapid prototyping to explore what can be achieved. 

Future iterations will likely include static IP assignments and enhanced security measures. Ideally, the "Boss Pi" would be a computer with two NICs (network interface cards). One NIC would facilitate communication with the worker Pis on a dedicated subnet, while the other NIC would handle external access, limited only to the Boss Pi. This setup would allow the Boss Pi to manage tasks like running Ansible and hosting a web app for API interactions, ensuring both functionality and isolation.

![architecture](images/arch.jpg)

## Setup Notes

On all 3 Pi's setup virtual env and install packages:
```bash
python3 -m venv ~/mpi_env
source ~/mpi_env/bin/activate
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
pip3 install transformers bitsandbytes
pip3 install accelerate
pip install fastapi uvicorn
```

For mpi4py on Ubuntu Linux and Debian Linux systems, binary packages are available for installation using the system package manager: (dont use pip)
* https://mpi4py.readthedocs.io/en/stable/install.html#linux
```bash
sudo apt install python3-mpi4py
```

Update system level packages:
```bash
source ~/mpi_env/bin/activate
python3 -m venv ~/mpi_env --system-site-packages
source ~/mpi_env/bin/activate
```

Make sure Bosspi has Passwordless SSH Access to the worker Pi's:
```bash
ssh-keygen -t rsa
ssh-copy-id ben@192.168.1.x  # workerpi1
ssh-copy-id ben@192.168.1.x  # workerpi2
```

## Setup With Ansible
Notes automating tasks with Ansible.

- **Ansible Inventory**: Defines the cluster nodes and their connection details.
- **MPI Hostfile**: Specifies the nodes and slots for MPI execution.
- **Ansible Playbook**: Automates cluster health checks and MPI program execution.

### Ansible Commands

- **RUN ALL**: 
  Executes the entire playbook:
  ```bash
  ansible-playbook -i /home/ben/ansible_hosts /home/ben/update_and_reboot_pis_workflow.yml -vv
  ```

- **Update All Pis Without Rebooting**: 
  Skips the reboot task:
  ```bash
  ansible-playbook -i /home/ben/ansible_hosts /home/ben/update_and_reboot_pis_workflow.yml --skip-tags "reboot" -vv
  ```

- **Reboot Only Worker Pis**: 
  Executes only the reboot task:
  ```bash
  ansible-playbook -i /home/ben/ansible_hosts /home/ben/update_and_reboot_pis_workflow.yml --tags "reboot" -vv
  ```

- **Check Installed Python Packages**: 
  Filters for the package check tasks:
  ```bash
  ansible-playbook -i /home/ben/ansible_hosts /home/ben/update_and_reboot_pis_workflow.yml --tags "check_packages" -vv
  ```

- **Install Project Dependencies**: 
  Runs only the dependency installation tasks:
  ```bash
  ansible-playbook -i /home/ben/ansible_hosts /home/ben/update_and_reboot_pis_workflow.yml --tags "install_dependencies" -vv
  ```

- **Copy Files Over to Worker Pis**: 
  Executes only the file copy task of `main.py` from boss pi to worker pis:
  ```bash
  ansible-playbook -i /home/ben/ansible_hosts /home/ben/update_and_reboot_pis_workflow.yml --tags "copy_mainpy_to_workers" -vv
  ```

- **Ensure Model Directory Exists on All Pis**:  
  Ensures the `/home/ben/model_parts` directory exists on all Pis:
  ```bash
  ansible-playbook -i /home/ben/ansible_hosts /home/ben/update_and_reboot_pis_workflow.yml --tags "ensure_model_dir" -vv
  ```

- **Download and Serialize Model on Boss Pi**:  
  Downloads the `distilgpt2` model on the Boss Pi, serializes it into layers, and saves the parts in `/home/ben/model_parts`:
  ```bash
  ansible-playbook -i /home/ben/ansible_hosts /home/ben/update_and_reboot_pis_workflow.yml --tags "download_new_model_on_boss" -vv
  ```

- **Distribute Model Parts to Worker Pis**:  
  Copies the serialized model parts from the Boss Pi to all Worker Pis:
  ```bash
  ansible-playbook -i /home/ben/ansible_hosts /home/ben/update_and_reboot_pis_workflow.yml --tags "distribute_new_model_to_workers" -vv
  ```

## **Run `main.py` with MPI** 
Executes the `main.py` script using MPI across the Boss Pi and Worker Pis:
```bash
mpirun -np 3 --hostfile ~/mpi_hostfile /home/ben/mpi_env/bin/python3 /home/ben/main.py
```
- **Explanation**:
  - `mpirun`: Runs the MPI program.
  - `-np 3`: Specifies 3 processes to run (1 for each Pi: Boss and 2 Workers).
  - `--hostfile ~/mpi_hostfile`: Specifies the hostfile containing the IPs or hostnames of the participating Pis.
  - `/home/ben/mpi_env/bin/python3 /home/ben/main.py`: Executes the Python script `main.py` using the virtual environment `mpi_env`.


## Inference notes across multiple devices

### **Rank 0 (Boss Pi/Web Server):**
- **Receives prompt via FastAPI**: The `generate_text` method handles the incoming HTTP request. This corresponds to the diagram's "Receive prompt" step.
- **Tokenizes prompt and generates embeddings**: The code tokenizes the prompt and converts it into embeddings using `embedding_layer`. This is the "Tokenize prompt" and "Generate embeddings" step.
- **Processes embeddings through its model layers**: The embeddings are forwarded through Rank 0's portion of the model via `forward_through_layers`. This is the "Forward embeddings" step.
- **Sends data to Rank 1 and Rank 2**: Activations are sent to Rank 1, and input IDs are sent to Rank 2 using `comm.send`. This corresponds to "Send activations to Rank 1" and "Send input IDs to Rank 2".
- **Receives generated text from Rank 2**: Rank 0 waits for and receives the final generated text from Rank 2. This is the "Receive generated text from Rank 2" step.
- **Returns the response to the client**: The generated text is returned as the HTTP response. This completes the process.

### **Rank 1 (Worker Pi 1):**
- **Receives activations from Rank 0**: The `comm.recv` method waits for activations from Rank 0. This matches the "Receive activations from Rank 0" step.
- **Processes activations through its model layers**: The activations are forwarded through Rank 1's portion of the model. This is the "Forward activations" step.
- **Sends intermediate activations to Rank 2**: The intermediate activations are sent to Rank 2. This is the "Send activations to Rank 2" step.

### **Rank 2 (Worker Pi 2):**
- **Receives activations from Rank 1 and input IDs from Rank 0**: The `comm.recv` calls wait for activations from Rank 1 and input IDs from Rank 0. This corresponds to "Receive intermediate activations from Rank 1" and "Receive input IDs from Rank 0".
- **Processes activations through its model layers**: The activations are processed through Rank 2's portion of the model. This is the "Forward activations" step.
- **Generates text using the final activations**: The text is generated using the `generate` method of the `model_full`. This matches the "Generate text using final activations" step.
- **Sends the generated text to Rank 0**: The generated text is sent back to Rank 0. This is the "Send generated text to Rank 0" step.

