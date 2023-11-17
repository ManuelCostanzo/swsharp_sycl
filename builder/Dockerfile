FROM ubuntu:22.04 as builder

ENV DEBIAN_FRONTEND noninteractive

RUN apt update -y && apt upgrade -y && apt install -y build-essential python3-dev git ninja-build python3-pip nano wget curl software-properties-common libboost-all-dev zip

RUN mkdir -p /scripts

### BEGIN - Install CMAKE
COPY /builder/scripts/install-cmake.sh /scripts/install-cmake.sh
RUN sh /scripts/install-cmake.sh
### END - Install CMAKE

### BEGIN - Install CUDA
COPY /builder/scripts/install-cuda.sh /scripts/install-cuda.sh
RUN sh /scripts/install-cuda.sh
### END - Install CUDA

### BEGIN - Install ROCM
COPY /builder/scripts/install-rocm.sh /scripts/install-rocm.sh
RUN sh /scripts/install-rocm.sh
### END - Install ROCM

### BEGIN - Install oneAPI
COPY /builder/scripts/install-oneapi.sh /scripts/install-oneapi.sh
RUN sh /scripts/install-oneapi.sh
### END - Install oneAPI

### BEGIN - Install Intel Runtime
COPY /builder/scripts/install-intel-runtime.sh /scripts/install-intel-runtime.sh
RUN sh /scripts/install-intel-runtime.sh
### END - Install Intel Runtime

### BEGIN - Install OpenCL
COPY /builder/scripts/install-opencl.sh /scripts/install-opencl.sh
RUN sh /scripts/install-opencl.sh
### END - Install OpenCL

### BEGIN Install LLVM
COPY /builder/scripts/install-llvm.sh /scripts/install-llvm.sh
RUN sh /scripts/install-llvm.sh
### END Install LLVM

### BEGIN Install AdaptiveCPP
COPY /builder/scripts/install-adaptivecpp.sh /scripts/install-adaptivecpp.sh
RUN sh /scripts/install-adaptivecpp.sh
### END Install AdaptiveCPP

## BEGIN Install Intel LLVM
COPY /builder/scripts/install-intel-llvm.sh /scripts/install-intel-llvm.sh
RUN sh /scripts/install-intel-llvm.sh
## END Install Intel LLVM


WORKDIR /app
