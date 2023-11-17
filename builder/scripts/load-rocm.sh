#!/bin/bash

export LD_PRELOAD=/opt/rocm/lib/libamdhip64.so:$LD_PRELOAD
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc