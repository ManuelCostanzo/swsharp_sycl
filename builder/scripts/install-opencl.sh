#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
### BEGIN - Install OpenCL
apt update -y && apt install -y clinfo ocl-icd-libopencl1 ocl-icd-opencl-dev
### END - Install OpenCL