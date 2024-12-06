# csi596-project

```
g++ -o transformer transformer.cpp -std=c++11
```

```
./transformer
```
## Overview
This project provides a highly optimized implementation of Multi-Head Self-Attention, a key component of Transformer models, for GPU architectures. Using CUDA, the project explores and evaluates two parallelization strategies: horizontal parallelization using native C++ threads and vertical parallelization using CUDA streams and events. The goal is to achieve substantial speedups for Transformer workloads, which are widely used in Natural Language Processing (NLP) and Computer Vision (CV) tasks.
## Background

Transformer models, and their multi-head attention mechanisms, have emerged as a powerful class of architectures for tasks in NLP and beyond. While these models demonstrate remarkable performance and flexibility, their computational demands can be substantial. This project focuses on the core computational kernel of the Transformer—the multi-head self-attention operation—and examines how different parallelization strategies affect performance. We begin with a naive CPU implementation that serves as a baseline, illustrating the computational steps of the attention kernel in a straightforward, single-threaded manner. We then introduce an MPI + OpenMP version that leverages distributed memory parallelism across multiple processes as well as shared-memory parallelism within each process. Finally, we develop a CUDA-accelerated GPU version that uses efficient GPU kernels and streams to exploit the massive parallelism of modern graphics processors. By comparing these three approaches, this project highlights the trade-offs in complexity, scalability, and speedup that emerge when moving from a simple CPU-only solution to more advanced parallelization strategies, ultimately guiding practitioners in selecting the most suitable optimization approach for their specific system and workload.

![Figure1](./README_IMAGE/Figure4.png)

## Overview

1. **Baseline Version (pure C++)**  
   A single-core, non-optimized reference implementation,Use pure C++ to implement a transformer.

2. **MPI + OpenMP parallelization:**

   - **MPI:** Distribute the workload (e.g., different input batches or heads) across multiple compute nodes/processes.
   - **OpenMP:** Further parallelize computations within each MPI process across multiple CPU threads.

3. **CUDA GPU acceleration:**  
   Implement and optimize the key operations on GPUs to exploit massive parallelism, and use CUDA streams to parallelize computations across multiple heads.

This comparison will be accomplished by time performance data monitored by python scripts and the generated diagrams.

## Methods

A single head attention kernel can be described as:

$$
Softmax\left(\frac{Q \times K^T}{\sqrt{d_k}}\right) \times V
$$

The Multi Head Attention Kernel includes three core operations:

- General Matrix Multiplication (GEMM)
- Softmax
- Transpose

The core of a multi head attention kernel is the single head attention kernel. The chronological order of a single head attention kernel can be described as:

- $X * W_Q \rightarrow Q(Q$ calculation $)$
- $X * W_K \rightarrow K$ (K calculation)
- $X * W_V \rightarrow V(V$ calculation $)$
- $Q * K^T\left(K^T\right.$ calculation $)$
- Softmax $\left(\frac{Q * K^T}{\sqrt{d_k}}\right)$
- Softmax $\left(\frac{Q * K^T}{\sqrt{d_k}}\right) \times V( Softmax * Vcalculation )$

## Summary of Results

### Baseline Performance

![Performance Comparison](./baseline/baseline.png)
This image shows the baseline performance of the system without optimization.

### OpenMP Performance

![Performance Comparison](./MPI_transformer/OpenMP.png)
This image demonstrates the performance improvement using OpenMP parallelization.

### OpenMP + MPI Performance

![Performance Comparison](./MPI_transformer/OpenMP+MPI.png)
This image shows the performance with combined OpenMP and MPI optimizations.

### Speedup Comparison

![Speedup Comparison](./speedup_comparison.png)
This image shows the speedup comparison between the different implementations.

## Reference
+ [Parallelizing Multi-Head Attention on GPUs (CS259)](https://hd10.dev/posts/my-interests-2/cs259.pdf)  
[]()
+ [NVIDIA cuBLAS Documentation: cuBLAS API Reference](https://docs.nvidia.com/cuda/cublas/index.html)
+ [MPI (MPICH / Open MPI)](https://www.mpich.org/),[Open MPI](https://www.open-mpi.org/)
+ [OpenMP Official Documentation](https://www.openmp.org/)
+ [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
