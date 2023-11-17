#!/bin/bash
export DEBIAN_FRONTEND=noninteractive

### BEGIN - Install Intel LLVM
export CPLUS_INCLUDE_PATH="/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:$CPLUS_INCLUDE_PATH"
export C_INCLUDE_PATH="/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:$C_INCLUDE_PATH"
export LD_LIBRARY_PATH="$LLVM_PATH/lib:$LD_LIBRARY_PATH"


export INTEL_LLVM_PATH=/opt/intel/llvm
mkdir -p $INTEL_LLVM_PATH
cd $INTEL_LLVM_PATH
git clone https://github.com/intel/llvm.git -b sycl
cd $INTEL_LLVM_PATH/llvm
CUDA_LIB_PATH=/usr/local/cuda/lib64/stubs CC=gcc CXX=g++ python3 ./buildbot/configure.py --shared-libs --cuda --hip --enable-all-llvm-targets --cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda" --cmake-opt="-DSYCL_BUILD_PI_HIP_ROCM_DIR=/opt/rocm" -t release
CUDA_LIB_PATH=/usr/local/cuda/lib64/stubs CC=gcc CXX=g++ python3 ./buildbot/compile.py
### BEGIN - Install Intel LLVM

# export LLVM_PATH=/home/mcostanzo/libs/llvm
# export LLVM_BUILD_DIR=/home/mcostanzo/builder/llvm

# CMAKE_OPTIONS="-DLLVM_ENABLE_PROJECTS=clang;compiler-rt;lld;openmp \
# -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
# -DCMAKE_C_COMPILER=gcc \
# -DCMAKE_CXX_COMPILER=g++ \
# -DCMAKE_BUILD_TYPE=Release \
# -DCMAKE_INSTALL_PREFIX=$LLVM_PATH \
# -DLLVM_ENABLE_ASSERTIONS=OFF \
# -DLLVM_TARGETS_TO_BUILD=all \
# -DCLANG_ANALYZER_ENABLE_Z3_SOLVER=0 \
# -DLLVM_INCLUDE_BENCHMARKS=0 \
# -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
# -DCMAKE_INSTALL_RPATH=$LLVM_PATH/lib \
# -DLLVM_ENABLE_OCAMLDOC=OFF \
# -DLLVM_ENABLE_BINDINGS=OFF \
# -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=OFF \
# -DLLVM_ENABLE_DUMP=OFF"

# mkdir -p $LLVM_BUILD_DIR/build
# cd $LLVM_BUILD_DIR/build
# cmake $CMAKE_OPTIONS $LLVM_BUILD_DIR/llvm
# make -j `nproc`
# make install
# cp -p $LLVM_BUILD_DIR/build/bin/llvm-lit   $LLVM_PATH/bin/llvm-lit
# cp -p $LLVM_BUILD_DIR/build/bin/FileCheck  $LLVM_PATH/bin/FileCheck
# cp -p $LLVM_BUILD_DIR/build/bin/count      $LLVM_PATH/bin/count
# cp -p $LLVM_BUILD_DIR/build/bin/not        $LLVM_PATH/bin/not
# cp -p $LLVM_BUILD_DIR/build/bin/yaml-bench $LLVM_PATH/yaml-bench