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
```

This installs mpi4py with apt:
```bash
sudo apt install python3-mpi4py
```

Update system level packages:
```bash
source ~/mpi_env/bin/activate
python3 -m venv ~/mpi_env --system-site-packages
source ~/mpi_env/bin/activate
```

Then test on all pys:
```bash
python3 -c "from mpi4py import MPI; print('mpi4py is installed and working.')"
```

On Boss PI this is ran to generate ssh keys and send them to workerpi's for configuring Passwordless SSH Access:
```bash
ssh-keygen -t rsa
ssh-copy-id ben@192.168.1.x  # workerpi1
ssh-copy-id ben@192.168.1.x  # workerpi2
```


On Boss PI this is ran:
```bash
nano ~/mpi_hosts
192.168.1.x slots=1  # bosspi
192.168.1.x slots=1  # workerpi1
192.168.1.x slots=1  # workerpi2
```

On all py's run virtual env
```bash
source ~/mpi_env/bin/activate
which python3
```

Copy the py script to the worker pys
```bash
scp ~/distilgpt2_distributed.py ben@192.168.1.x:/home/ben/  # workerpi1
scp ~/distilgpt2_distributed.py ben@192.168.1.x:/home/ben/  # workerpi2
```

On the bosspi run the script with MPI
```bash
mpirun -np 3 --hostfile ~/mpi_hosts /home/ben/mpi_env/bin/python3 ~/distilgpt2_distributed.py
```

## Setup With Ansible
Notes to running Pi clusters with Ansible.

- **Ansible Inventory**: Defines the cluster nodes and their connection details.
- **MPI Hostfile**: Specifies the nodes and slots for MPI execution.
- **Ansible Playbook**: Automates cluster health checks and MPI program execution.

### **Files and Configuration**

#### **3.1 Ansible Inventory File (`~/ansible_hosts`)**
Defines the nodes for Ansible automation:
```ini
[all]
bosspi ansible_host=192.168.1.149 ansible_connection=local
workerpi1 ansible_host=192.168.1.181 ansible_user=ben
workerpi2 ansible_host=192.168.1.183 ansible_user=ben
```

#### **3.2 MPI Hostfile (`~/mpi_hostfile`)**
Specifies the nodes and slots for MPI execution:
```plaintext
192.168.1.149 slots=1 # bosspi
192.168.1.181 slots=1 # workerpi1
192.168.1.183 slots=1 # workerpi2
```

## **Run All With Ansible Playbook**

```bash
ansible-playbook -i ~/ansible_hosts ~/run_mpi_with_health.yml
```

## **Individual Ansible Commands**

Ping all Pi's:
```bash
ansible all -i ~/ansible_hosts -m ping
```

Check memory on all Pi's:
```bash
ansible all -i ~/ansible_hosts -m command -a "free -m"
```

Distribute new Py file to all Pi's:
```bash
ansible all -i ~/ansible_hosts -m copy -a "src=/home/ben/mpi_hello_world.py dest=/home/ben/mpi_hello_world.py owner=ben mode=0755"
```

Run MPI program on the Boss Pi:
```bash
ansible bosspi -i ~/ansible_hosts -m shell -a "mpirun -np 3 --hostfile /home/ben/mpi_hostfile /home/ben/mpi_env/bin/python3 /home/ben/mpi_hello_world.py"
```
