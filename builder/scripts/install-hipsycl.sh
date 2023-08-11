#!/bin/bash
export LD_LIBRARY_PATH=/opt/hipSYCL/llvm/lib/:$LD_LIBRARY_PATH
export PATH=/opt/hipSYCL/llvm/bin:$PATH
export HIPSYCL_INSTALL_PREFIX=/opt/hipSYCL

set -e
HIPSYCL_BUILD_DIR=/tmp/hipSYCL-installer
HIPSYCL_LLVM_DIR=/opt/hipSYCL/llvm/lib/


echo "Cloning hipSYCL"
git clone --recurse-submodules -b develop https://github.com/illuhad/hipSYCL $HIPSYCL_BUILD_DIR

mkdir -p $HIPSYCL_BUILD_DIR/build
cd $HIPSYCL_BUILD_DIR/build

cmake \
-DDWITH_SSCP_COMPILER=ON \
-DWITH_CPU_BACKEND=ON \
-DWITH_CUDA_BACKEND=ON \
-DWITH_ROCM_BACKEND=ON \
-DLLVM_DIR=$HIPSYCL_LLVM_DIR \
-DROCM_PATH=/opt/rocm \
-DCMAKE_INSTALL_PREFIX=$HIPSYCL_INSTALL_PREFIX \
..

make -j `nproc` install
