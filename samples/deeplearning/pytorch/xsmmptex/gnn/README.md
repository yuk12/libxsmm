The codebase for the paper titled "DistGNN: Scalable Distributed Training for Large-Scale Graph
Neural Networks" is available at:  
https://github.com/sanchit-misra/dgl/tree/xeon-optimizations

We used GCC 8.3.0 and CentOS 7.6/8.0 for compilation and runs.  


Required libraries/frameworks:
1. Pytorch - https://pytorch.org/
2. OneCCL - https://github.com/ddkalamk/torch-ccl/tree/working_1.7 

A. For single socket experiments: 
please refer to the README file in 'dgl/examples/pytorch/graphsage' and 'dgl/examples/pytorch/rgcn'


Example commands to run expeirments on benchmark datasets:
Assuming a dual socket system with 28 cores per socket, we use the following commands.

We use the following ENV variable settings:  
export OMP_NUM_THREADS=28   
export KMP_AFFINITY=compact,granularity=fine,1,0     
export DGL_CPU_INTEL_KERNEL_ENABLED=1  


numactl -N 0 -m 0 python train_full.py --n-epochs 200 --dataset reddit   
numactl -N 0 -m 0 python train_full_ogbn-products.py --n-epochs 300 --dataset ogbn-products  
numactl -N 0 -m 0 python train_full_ogbn-papers.py --n-epochs 200 --dataset ogbn-papers100M  
numactl -N 0 -m 0 python train_full_proteins.py --n-epochs 200 --dataset proteins  



B. For distributed-memory experiments,
please refer to the README file in 'dgl/examples/pytorch/graphsage/experimental'

Algorithm cd-0:   
sh run_dist.sh -n <#sockets> -ppn 2 python train_dist_sym.py --dataset reddit --n-epochs 200 --nr 1  
Algorithm cd-5:   
sh run_dist.sh -n <#sockets> -ppn 2 python train_dist_sym.py --dataset reddit --n-epochs 200 --nr 5  
Algorithm 0c :   
sh run_dist.sh -n <#sockets> -ppn 2 python train_dist_sym.py --dataset reddit --n-epochs 200 --nr -1  



