# DistributedGPT

This is a hobby project primarily for learning and experimentation using three old Raspberry Pi Model B units that were gathering dust. 
The goal is to explore the feasibility of distributing a GPT-2 model across multiple devices and leveraging parallel computing techniques to enable efficient operation of a large language model. 


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
  ansible-playbook -i /home/ben/ansible_hosts /home/ben/update_and_reboot_pis_workflow.yml --tags "copy_files" -vv
  ```
