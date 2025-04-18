# name: 
- gpd

# channels:
- defaults

# dependencies:
  - elevation==1.1.3
  - gstools==1.6.0
  - numba==0.60.0
  - matplotlib==3.10.0
  - numpy==2.0.2
  - mpmath==1.3.0
  - scipy==1.14.1
  - mpi4py==3.1.4
  - python==3.11.11
  - rpy2==3.5.11
  - gmpy2==2.1.5
  - geopandas==1.0.1 (available on conda; misspiggy installed through pip)
  - (misspiggy) ipykernel
  - (misspiggy) jupyter_client[version='<8']
  - (misspiggy) pyzmq[version='<25']
  - (misspiggy) notebook
  - (misspiggy) openmpi=4.1.5
  - (misspiggy) openmpi-mpicxx=4.1.5
  - (misspiggy) htop==3.2.2

# pip:
  - (misspiggy) geopandas==1.0.1
  - (misspiggy) nvidia-cuda...
  - (misspiggy) tensorflow==2.18.0
    - (misspiggy) keras==3.7.0
  - (misspiggy) pandas==2.2.3


