# **Quantum High-Level Interface (QHLI)**

*A GPU-accelerated quantum simulation toolkit for Pawsey supercomputing nodes*

---

## Overview

**QHLI** (Quantum High-Level Interface) is a modular C++/CUDA framework designed for **GPU-accelerated quantum algorithm simulation**.
Unlike hardware emulators, QHLI focuses on **numerical algorithmic exploration**, enabling researchers to test, validate, and benchmark dense and sparse quantum operations directly on GPU hardware.

QHLI is optimized for **NVIDIA Grace-Hopper (GH200)** and **ELLA GPU nodes** at the **Pawsey Supercomputing Centre**, leveraging NVIDIA’s quantum SDK and HPC libraries for scalable performance.

---

## ⚙️ Key Features

* **GPU-accelerated quantum statevector simulation**
  Supports both dense and sparse unitary operations.

* **Sparse matrix operators (CSR)**
  Built with cuSPARSE for Hamiltonian evolution, Chebyshev, and Taylor exponential methods.

* **Hybrid dense/sparse routines**
  Optimized switching between cuBLAS and cuSPARSE backends.

* **Custom quantum gates and high-level operators**
  Apply arbitrary sparse unitaries, controlled gates, and multi-qubit exponentials.

* **Benchmark and profiling tools**
  Integrated timers and support for NVIDIA Nsight Systems profiling.

---

## Dependencies

QHLI builds upon **CUDA and NVIDIA’s HPC libraries**, requiring the following:

| Library                                     | Purpose                                       | Notes                                        |
| ------------------------------------------- | --------------------------------------------- | -------------------------------------------- |
| **CUDA Toolkit**                   | GPU runtime and compiler                      | `nvcc`, `cuda_runtime.h`                     |
| **cuSPARSE**                                | Sparse matrix-vector multiplication (CSR/CSC) | Used in sparse gates & Chebyshev expansion   |
| **cuBLAS**                                  | Dense matrix operations                       | For full-matrix exponentials and rotations   |
| **cuQuantum SDK (cuStateVec, cuTensorNet)** | Quantum state simulation backend              | Optional acceleration for large statevectors |
| **GoogleTest**                              | Unit testing                                  | Linked via CMake `FetchContent`              |
| **CMake**                          | Build system                                  | Supports both `nvcc` and `nvc++`             |
| **OpenMP / MPI**                            | Optional                                      | Multi-threaded or distributed execution      |

---

## System Requirements (Pawsey ELLA / Setonix)

| Resource            | Requirement                                    |
| ------------------- | ---------------------------------------------- |
| **GPU**             | NVIDIA A100, H100, or GH200                    |
| **CPU**             | Grace-Hopper CPU (ARMv9) or x86_64             |
| **Compiler**        | `nvc++` (HPC SDK ≥24.3) or `nvcc` (CUDA ≥12.4) |
| **Modules to Load** |                                                |

---

## Build Instructions

```bash
[git clone https://github.com/username/qhli.git](https://github.com/Tuan-D-N/Quantum-High-Level-Interface.git)
cd qhli
mkdir build && cd build
cmake ..
make 
```
