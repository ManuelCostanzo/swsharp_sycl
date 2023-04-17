# SW#SYCL

## Description
***SW#SYCL*** is a SYCL project based on [SW#](https://github.com/mkorpar/swsharp "SW#") to extend support for multiple architectures.

## Prerequisites
There is a `builder` folder with all the necessary tools to compile with [Intel LLVM](https://github.com/intel/llvm "Intel LLVM"), [hipSYCL](https://github.com/illuhad/hipSYCL "hipSYCL"), [DPCPP](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top.html "DPCPP") and [NVCC 11.7.0](https://developer.nvidia.com/cuda-downloads "NVCC 11.7.0"). In addition, it installs the [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#base-kit, "Intel oneAPI Base Toolkit").


## Build
- *Compilation for Intel LLVM*: `BACK=intel make`
- *Compilation for hipSYCL*: `BACK=hip make`
- *Compilation for DPC++*: `BACK=dpcpp make`
- *Compilation for FPGA*: `BACK=fpga make`
- *Compilation CUDA*: `make`

## Usage
- *Proteins*
	- `./bin/swsharpd .. (same SW# flags)`
- *DNA*
	- `./bin/swsharpn .. (same SW# flags)`


## Folders
- *Builder*: contains all the prerequisite software
- *CUDA*: SW# project with a few improvements
- *SYCL*: SW#SYCL project.
- *databases*: contains scripts to download database and sequences of both proteins and dna.
- *dpct_original_output*: contains the original output of the `oneAPI dpct tool` after migration.
- *scripts*: contains tests scripts

## References

Costanzo. M, Rucci. E, García. C, Naiouf. M, and Prieto-Matias M. 2022. Migrating CUDA to oneAPI: A Smith-Waterman Case Study. In Bioinformatics and Biomedical Engineering: 9th International Work-Conference, IWBBIO 2022, Maspalomas, Gran Canaria, Spain, June 27–30, 2022, Proceedings, Part II. Springer-Verlag, Berlin, Heidelberg, 103–116. https://doi.org/10.1007/978-3-031-07802-6_9

Costanzo, M., Rucci, E., Sánchez, C. G., Naiouf, M., & Prieto-Matías, M. (2022). Assessing Opportunities of SYCL and Intel oneAPI for Biological Sequence Alignment. arXiv preprint arXiv:2211.10769. https://arxiv.org/abs/2211.10769

## Contact
If you have any question or suggestion, please contact Manuel Costanzo (mcostanzo@lidi.info.unlp.edu.ar)
