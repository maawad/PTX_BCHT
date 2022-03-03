# PTX BCHT
Bucketed Cuckoo hash table written in PTX and JIT-compiled.

## About
This repository aims to experiment with JIT compilations of hand-written PTX CUDA files (we also love assembly). We implemented [insertion](./ptx/bcht_insert_kernel.ptx) and [query](./ptx/bcht_find_kernel.ptx)  kernels for a bucketed cuckoo hash set entirely in NVIDIA's PTX.


Cuckoo hashing is a probing scheme that achieves very low number of probes at very high load factors. This implementation is probably is the fastest GPU hash set implementation, but if you are interested in the state-of-the-art cuckoo hashing implementation, check out our bucketed cuckoo hashing [implementation](https://github.com/owensgroup/BGHT) and [paper](https://arxiv.org/abs/2108.07232).

## Build
```bash
git clone https://github.com/maawad/PTX_BCHT.git
mkdir build && cd build
cmake ..
make
```

## Run
Requirement: NVIDIA GPU with Volta or later microarchitecture and CUDA.

Usage
```bash
./ptx_cuckoo_hashtable
--num-keys                    Number of keys
--load-factor                 Load factor of the hash set
--exist-ratio                 Ratio of queries that exist in the hash set
--device_id                   GPU device ID
```

Example:
```bash
# from the build directory:
./ptx_cuckoo_hashtable --num-keys=1'000'000 --load-factor=0.99 --exist-ratio=1.0
```
Output:
```bash
num_keys: 1000000
load_factor: 0.99
device_id: 0
Device[0]: NVIDIA TITAN V
--------------------------------------------------------
CUDA Link Completed in 0.000000 ms.
Linker Output:
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'bcht_insert' for 'sm_70'
ptxas info    : Function properties for bcht_insert
ptxas         .     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 28 registers, 412 bytes cmem[0]
info    : 0 bytes gmem
info    : Function properties for 'bcht_insert':
info    : used 28 registers, 0 stack, 0 bytes smem, 412 bytes cmem[0], 0 bytes lmem
--------------------------------------------------------
CUDA kernel: bcht_insert launched
--------------------------------------------------------
CUDA Link Completed in 0.000000 ms.
Linker Output:
ptxas info    : 98 bytes gmem
ptxas info    : Compiling entry function 'bcht_find' for 'sm_70'
ptxas info    : Function properties for bcht_find
ptxas         .     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 28 registers, 424 bytes cmem[0]
info    : 98 bytes gmem
info    : Function properties for 'bcht_find':
info    : used 28 registers, 0 stack, 0 bytes smem, 424 bytes cmem[0], 0 bytes lmem
--------------------------------------------------------
CUDA kernel: bcht_find launched
Success!
Find rate:      3160.4 million keys/s
Insert rate:    1650.99 million keys/s
Find ratio was: 100%
```

## Performance
Results on a Titan V GPU for building a hash table with a given number of keys then performing queries (same number as the number of keys, 50% of the queries exist in the hash table).
```
   Millions    |      %      |        Million keys/s
Number of keys | Load factor | Insertion rate | Find rate
      50       |     60      |     1461.81    |  3800.74
      50       |     65      |     1462.44    |  3779.85
      50       |     70      |     1462.18    |  3736.75
      50       |     75      |     1458.97    |  3660.28
      50       |     80      |     1451.99    |  3538.27
      50       |     82      |     1446.38    |  3454.41
      50       |     84      |     1440.71    |  3378.41
      50       |     86      |     1431.60    |  3262.82
      50       |     88      |     1419.21    |  3264.78
      50       |     90      |     1404.71    |  2987.34
      50       |     91      |     1395.24    |  2917.55
      50       |     92      |     1383.33    |  2822.76
      50       |     93      |     1371.44    |  2740.23
      50       |     94      |     1356.04    |  2724.78
      50       |     95      |     1337.41    |  2601.53
      50       |     96      |     1314.53    |  2431.92
      50       |     97      |     1285.49    |  2316.00
      50       |     98      |     1243.96    |  2188.13
      50       |     99      |     1178.17    |  2045.16


```

## Authors:
[Muhammad Awad](https://github.com/maawad) and [Serban Porumbescu](https://github.com/porumbes).
