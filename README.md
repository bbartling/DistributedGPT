# DistributedGPT

This is a hobby project primarily for learning and experimentation using three old Raspberry Pi Model B units that were gathering dust. 
The goal is to explore the feasibility of distributing a GPT-2 model across multiple devices and leveraging parallel computing techniques to enable efficient operation of a large language model. 


## Notes

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

