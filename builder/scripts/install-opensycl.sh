#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/llvm/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/hip/lib/
export PATH=/opt/llvm/bin:$PATH
export PATH=/opt/rocm/bin:$PATH
export OPENSYCL_INSTALL_PREFIX=/opt/openSYCL

set -e
OPENSYCL_BUILD_DIR=/tmp/openSYCL-installer
OPENSYCL_LLVM_DIR=/opt/llvm/lib/cmake/llvm/


rm -rf $OPENSYCL_BUILD_DIR
git clone --recurse-submodules -b develop https://github.com/OpenSYCL/OpenSYCL $OPENSYCL_BUILD_DIR

mkdir -p $OPENSYCL_BUILD_DIR/build
cd $OPENSYCL_BUILD_DIR/build

cmake \
-DWITH_ACCELERATED_CPU=ON \
-DWITH_SSCP_COMPILER=ON \
-DWITH_CPU_BACKEND=ON \
-DWITH_CUDA_BACKEND=ON \
-DWITH_ROCM_BACKEND=ON \
-DLLVM_DIR=$OPENSYCL_LLVM_DIR \
-DROCM_PATH=/opt/rocm \
-DCMAKE_INSTALL_PREFIX=$OPENSYCL_INSTALL_PREFIX \
..

make -j `nproc` install
ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so.1
