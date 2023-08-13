#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/llvm-17/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/hip/lib/
export PATH=/opt/llvm/bin:$PATH
export PATH=/opt/rocm/bin:$PATH
export OPENSYCL_INSTALL_PREFIX=/opt/openSYCL

set -e
OPENSYCL_BUILD_DIR=/tmp/openSYCL-installer
OPENSYCL_LLVM_DIR=/opt/llvm/lib/cmake/llvm/


ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so.1
rm -rf $OPENSYCL_BUILD_DIR
git clone --recurse-submodules -b develop https://github.com/OpenSYCL/OpenSYCL $OPENSYCL_BUILD_DIR

mkdir -p $OPENSYCL_BUILD_DIR/build
cd $OPENSYCL_BUILD_DIR/build

#-DWITH_SSCP_COMPILER=OFF \
#-DWITH_CUDA_BACKEND=ON \
#-DWITH_ROCM_BACKEND=ON \

cmake \
-DLLVM_DIR=/usr/lib/llvm-17/cmake \
-DWITH_ACCELERATED_CPU=ON \
-DWITH_CPU_BACKEND=ON \
-DROCM_PATH=/opt/rocm \
-DCMAKE_INSTALL_PREFIX=$OPENSYCL_INSTALL_PREFIX \
..

make -j `nproc` install
