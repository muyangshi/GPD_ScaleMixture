# mpirun --oversubscribe -n 100 python3 test_mpi.py

# %%
# Imports and Set Parameters
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Print hello message from each process
print(f"Hello from process {rank} out of {size}")

if rank == 0:
    print("This is the master process!")

from RW_inte import *

print(f"pRW_standard_Pareto_nugget_vec(20, 0.5, 0.5, 10) = {pRW_standard_Pareto_nugget_vec(20, 0.5, 0.5, 10)} from rank {rank}")

# from RW_inte import *
# print(rank, pRW_transformed_cpp(20, 0.5, 0.5, 10))

# if rank == 0:
#     Nt = size
#     k = 8
#     # R_log_cov = np.array([t*np.eye(k) for t in range(Nt)])
#     R_log_cov_to_scat = [[0.5*i] * k for i in range(Nt)] # shape(Nt, k) scatter along axis 0
# else:
#     R_log_cov_to_scat = None
# scattered_R_log_cov = comm.scatter(R_log_cov_to_scat, root = 0)
# print('rank:',rank,scattered_R_log_cov,type(scattered_R_log_cov[0]))
# %%
