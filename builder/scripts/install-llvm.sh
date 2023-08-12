#!/bin/bash
LLVM_BUILD_DIR=/opt/build/llvm
LLVM_INSTALL_PREFIX=/opt/llvm

set -e

mkdir -p $LLVM_BUILD_DIR
wget https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-16.0.4.tar.gz -P $LLVM_BUILD_DIR
tar -xzvf $LLVM_BUILD_DIR/llvmorg-16.0.4.tar.gz  --directory $LLVM_BUILD_DIR

LLVM_BUILD_DIR=$LLVM_BUILD_DIR/llvm-project-llvmorg-16.0.4

CC=cc
CXX=c++
BUILD_TYPE=Release
TARGETS_TO_BUILD="AMDGPU;NVPTX;X86"
NUMTHREADS=`nproc`

CMAKE_OPTIONS="-DLLVM_ENABLE_PROJECTS=clang;compiler-rt;lld;openmp \
-DOPENMP_ENABLE_LIBOMPTARGET=OFF \
-DCMAKE_C_COMPILER=$CC \
-DCMAKE_CXX_COMPILER=$CXX \
-DCMAKE_BUILD_TYPE=$BUILD_TYPE \
-DCMAKE_INSTALL_PREFIX=$LLVM_INSTALL_PREFIX \
-DLLVM_ENABLE_RUNTIMES=all \
-DLLVM_ENABLE_ASSERTIONS=OFF \
-DLLVM_TARGETS_TO_BUILD=$TARGETS_TO_BUILD \
-DCLANG_ANALYZER_ENABLE_Z3_SOLVER=0 \
-DLLVM_INCLUDE_BENCHMARKS=0 \
-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
-DCMAKE_INSTALL_RPATH=$LLVM_INSTALL_PREFIX/lib \
-DLLVM_ENABLE_OCAMLDOC=OFF \
-DLLVM_LINK_LLVM_DYLIB=ON \
-DLLVM_ENABLE_BINDINGS=ON \
-DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=OFF \
-DLLVM_BUILD_LLVM_DYLIB=ON \
-DWITH_ROCM_BACKEND=ON \
-DWITH_LEVEL_ZERO_BACKEND=ON \
-DLLVM_ENABLE_DUMP=OFF" \

mkdir -p $LLVM_BUILD_DIR/build
cd $LLVM_BUILD_DIR/build
cmake $CMAKE_OPTIONS $LLVM_BUILD_DIR/llvm
make -j $NUMTHREADS
make install
cp -p $LLVM_BUILD_DIR/build/bin/llvm-lit   $LLVM_INSTALL_PREFIX/bin/llvm-lit
cp -p $LLVM_BUILD_DIR/build/bin/FileCheck  $LLVM_INSTALL_PREFIX/bin/FileCheck
cp -p $LLVM_BUILD_DIR/build/bin/count      $LLVM_INSTALL_PREFIX/bin/count
cp -p $LLVM_BUILD_DIR/build/bin/not        $LLVM_INSTALL_PREFIX/bin/not
cp -p $LLVM_BUILD_DIR/build/bin/yaml-bench $LLVM_INSTALL_PREFIX/yaml-bench
