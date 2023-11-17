#!/bin/bash

export LD_PRELOAD=/opt/rocm/lib/libamdhip64.so:$LD_PRELOAD
export LLVM_PATH=/usr/lib/llvm-15
export ACPP_PATH=/opt/AdaptiveCPP
export LD_LIBRARY_PATH=$LLVM_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ACPP_PATH/lib:$LD_LIBRARY_PATH
export PATH=$LLVM_PATH/bin:$PATH
export PATH=$ACPP_PATH/bin:$PATH
