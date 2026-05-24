# Bridging the Vendor Gap: Enabling AMD GPU Support for Awkward Array via ROCm/HIP for the HL-LHC Era 

The computational demands of the High-Luminosity LHC (HL-LHC) necessitate a transition toward heterogeneous computing environments. While the Scikit-HEP ecosystem has historically leveraged NVIDIA GPUs through CUDA, the increasing deployment of AMD-based supercomputers requires a vendor-neutral approach to performance portability.

This contribution details the design and implementation of the initial ROCm/HIP backend for Awkward Array, the foundation for Pythonic HEP analysis. By implementing the library’s kernel infrastructure in HIP, we enable the execution of complex, nested, and irregular array operations on AMD hardware without sacrificing the library's high-level user interface.

We present a performance evaluation conducted on the Princeton Della Cluster, utilizing AMD Instinct MI210/MI250 (CDNA 2) GPUs. Our benchmarks focus on kernel execution throughput and memory management efficiency for jagged data structures, comparing these results against traditional CUDA-based execution. Furthermore, we discuss the forward-compatibility of this infrastructure with the latest CDNA 4 architecture (AMD Instinct MI350 series). We highlight how the massive 288GB HBM3E memory and advanced datatype support of the MI350 series can further alleviate the memory-bottlenecks typical of HEP data processing. This work represents a critical step in providing the HEP community with a truly hardware-agnostic analysis stack, ensuring high performance across the full spectrum of modern HPC resources.

### Performance Benchmarks

Click below to view the interactive Criterion profile reports for each GPU execution backend:

[![Metal Benchmarks](https://img.shields.io/badge/Benchmarks-Apple_Metal-black.svg?logo=apple)](https://raw.githack.com/ianna/CHEP2026-Bridging-the-Vendor-Gap-Enabling-AMD-GPU-Support-for-Awkward-Array/main/criterion/report/index.html)
[![AMD GPU Benchmarks](https://img.shields.io/badge/Benchmarks-AMD_ROCm-orange.svg?logo=amd)](https://raw.githack.com/ianna/CHEP2026-Bridging-the-Vendor-Gap-Enabling-AMD-GPU-Support-for-Awkward-Array/main/criterion/report/index.html)
