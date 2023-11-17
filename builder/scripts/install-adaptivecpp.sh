#!/bin/bash
export DEBIAN_FRONTEND=noninteractive

### BEGIN - Install AdaptiveCPP
export LLVM_PATH=/usr/lib/llvm-15

export CPLUS_INCLUDE_PATH="/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:$CPLUS_INCLUDE_PATH"
export C_INCLUDE_PATH="/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:$C_INCLUDE_PATH"
export LD_LIBRARY_PATH="$LLVM_PATH/lib:$LD_LIBRARY_PATH"

export ACPP_INSTALL_PREFIX=/opt/AdaptiveCPP
export ACPP_BUILD_DIR=/tmp/AdaptiveCPP-installer

rm -rf $ACPP_BUILD_DIR
git clone --recurse-submodules -b develop https://github.com/AdaptiveCpp/AdaptiveCpp $ACPP_BUILD_DIR

mkdir -p $ACPP_BUILD_DIR/build
cd $ACPP_BUILD_DIR/build

cmake \
-DCLANG_EXECUTABLE_PATH=$LLVM_PATH/bin/clang++ \
-DCMAKE_C_COMPILER=$LLVM_PATH/bin/clang \
-DCMAKE_CXX_COMPILER=$LLVM_PATH/bin/clang++ \
-DLLVM_DIR=$LLVM_PATH/cmake \
-DWITH_ACCELERATED_CPU=ON \
-DWITH_CPU_BACKEND=ON \
-DWITH_SSCP_COMPILER=ON \
-DWITH_CUDA_BACKEND=ON \
-DWITH_ROCM_BACKEND=ON \
-DWITH_OPENCL_BACKEND=ON \
-DWITH_LEVEL_ZERO_BACKEND=ON \
-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
-DROCM_PATH=/opt/rocm \
-DOpenCL_LIBRARY=/usr/lib/x86_64-linux-gnu/libOpenCL.so \
-DCMAKE_INSTALL_PREFIX=$ACPP_INSTALL_PREFIX \
..
# -DNVCXX_COMPILER=/path/to/nvc++`

make -j `nproc` install

### END - Install AdaptiveCPP