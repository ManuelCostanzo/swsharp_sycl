#!/bin/bash
mkdir -p /opt/build/intel
cd /opt/build/intel
git clone https://github.com/intel/llvm.git -b sycl
cd /opt/build/intel/llvm
# CUDA_LIB_PATH=/usr/local/cuda/lib64/stubs CC=gcc CXX=g++ python3 ./buildbot/configure.py --enable-all-llvm-targets --shared-libs --cuda --hip --cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda" --cmake-opt="-DSYCL_BUILD_PI_HIP_ROCM_DIR=/opt/rocm" -t release
CUDA_LIB_PATH=/usr/local/cuda/lib64/stubs CC=gcc CXX=g++ python3 ./buildbot/configure.py --enable-all-llvm-targets --shared-libs --hip --cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda" --cmake-opt="-DSYCL_BUILD_PI_HIP_ROCM_DIR=/opt/rocm" -t release
CUDA_LIB_PATH=/usr/local/cuda/lib64/stubs CC=gcc CXX=g++ python3 ./buildbot/compile.py
