#!/bin/bash

export LD_PRELOAD=/opt/rocm/lib/libamdhip64.so:$LD_PRELOAD
export INTEL_LLVM_PATH=/opt/intel/llvm
export ONEAPI_PATH=/opt/intel/oneapi
export LD_LIBRARY_PATH=$INTEL_LLVM_PATH/llvm/build/lib:$LD_LIBRARY_PATH
export PATH=$INTEL_LLVM_PATH/llvm/build/bin:$PATH
source $ONEAPI_PATH/setvars.sh

