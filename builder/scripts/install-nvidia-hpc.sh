#!/bin/bash
export DEBIAN_FRONTEND=noninteractive

### BEGIN - NVHPC (26 GB SIZE!)
export NVHPC_MAJOR_VERSION="23"
export NVHPC_MINOR_VERSION="9"
curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | tee /etc/apt/sources.list.d/nvhpc.list
apt update -y && apt install -y nvhpc-$NVHPC_MAJOR_VERSION-$NVHPC_MINOR_VERSION-cuda-multi
### END - NVHPC
