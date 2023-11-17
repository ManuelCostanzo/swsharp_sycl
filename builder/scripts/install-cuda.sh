#!/bin/bash
export DEBIAN_FRONTEND=noninteractive

### BEGIN - Install CUDA
export CUDA_VERSION="12-3"
export CUDA_PATH=/usr/local/cuda
apt update -y
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" -y
apt update -y

apt install -y cuda-$CUDA_VERSION
apt install -y cuda-toolkit-$CUDA_VERSION
apt install -y cuda-nsight-systems-$CUDA_VERSION
apt install -y cuda-nsight-compute-$CUDA_VERSION

echo "export CUDA_PATH=$CUDA_PATH" >> ~/.bashrc
echo 'export CUDA_HOME=$CUDA_PATH' >> ~/.bashrc
echo 'export CUDADIR=$CUDA_PATH' >> ~/.bashrc
echo 'export PATH=$CUDA_PATH/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

ln -s $CUDA_PATH/lib64/stubs/libcuda.so /usr/lib/libcuda.so.1
echo kernel.perf_event_paranoid=2 > /etc/sysctl.d/local.conf
### END - Install CUDA


