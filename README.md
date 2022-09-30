# SW#SYCL

## Description
SW#SYCL is a SYCL project based on [SW#](http://https://github.com/mkorpar/swsharp/tree/master/swsharp "SW#") to extend support for multiple architectures.

## Prerequisites
SW#SYCL contains a `builder` folder with all the necessary tools to compile with [Intel LLVM](https://github.com/intel/llvm "Intel LLVM"), [hipSYCL](https://github.com/illuhad/hipSYCL "hipSYCL"), [DPCPP](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top.html "DPCPP") and [NVCC 11.7.0](https://developer.nvidia.com/cuda-downloads "NVCC 11.7.0"). In addition, it installs the [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#base-kit, "Intel oneAPI Base Toolkit").


## Build
- Compilation for Intel LLVM `BACK=intel make`
- Compilation for hipSYCL `BACK=hip make`
- Compilation for DPC++ `BACK=dpcpp make`
- *Compilation CUDA*: make

## Usage
- *Proteins*
	- `./bin/swsharpd .. (same SW# flags)`
- *DNA*
	- `./bin/swsharpn .. (same SW# flags)`


## FOLDERS
- *Builder*: contains all the prerequisite software
- *CUDA*: SW# project with some modifications
- *SYCL*: SW#SYCL project.
- *databases*: contains scripts to download database and sequences of both proteins and dna.
- *dpct_original_output*: contains the original output of the `oneAPI dpct tool` after migration.
- *scripts*: contains tests scripts
