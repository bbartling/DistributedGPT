from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Each process generates a random number
local_value = np.random.random()
print(f"Process {rank} on {MPI.Get_processor_name()} has value: {local_value}")

# Find the maximum using MPI_Reduce
global_max = comm.reduce(local_value, op=MPI.MAX, root=0)

if rank == 0:
    print(f"Global maximum is: {global_max}")


global_max = comm.bcast(global_max, root=0)
scaled_value = local_value / global_max
print(f"Process {rank} on {MPI.Get_processor_name()} scaled value: {scaled_value}")

local_value = np.random.random()
global_sum = comm.allreduce(local_value, op=MPI.SUM)
print(f"Process {rank} on {MPI.Get_processor_name()} sees global sum: {global_sum}")

scaled_value = local_value / global_sum
scaled_sum = comm.allreduce(scaled_value, op=MPI.SUM)
print(f"Process {rank} on {MPI.Get_processor_name()}: Scaled sum is {scaled_sum}")

data = np.array(rank, dtype=int)
gathered_data = None
if rank == 0:
    gathered_data = np.empty(comm.size, dtype=int)

comm.Gather(data, gathered_data, root=0)
if rank == 0:
    print(f"Gathered data at root: {gathered_data}")

scatter_data = None
if rank == 0:
    scatter_data = np.arange(comm.size, dtype=int)

recv_data = np.empty(1, dtype=int)
comm.Scatter(scatter_data, recv_data, root=0)
print(f"Process {rank} on {MPI.Get_processor_name()} received {recv_data[0]}")

send_data = np.array([rank * comm.size + i for i in range(comm.size)], dtype=int)
recv_data = np.empty(comm.size, dtype=int)
comm.Alltoall(send_data, recv_data)
print(f"Process {rank} on {MPI.Get_processor_name()} sent {send_data} and received {recv_data}")

print(f"Process {rank} on {MPI.Get_processor_name()} before barrier")
comm.Barrier()
print(f"Process {rank} on {MPI.Get_processor_name()} after barrier")
