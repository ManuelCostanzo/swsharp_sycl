FROM ubuntu:20.04 as builder

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y build-essential python3-dev git ninja-build python3-pip nano wget curl software-properties-common libboost-all-dev zip

#Copy installation scripts
COPY /builder/scripts /scripts

#### BEGIN - Install last cmake
RUN apt purge --auto-remove cmake -y
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - |  tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'     
RUN apt update -y && apt install cmake -y
#### END - Install last cmake

#### BEGIN - Install CUDA 11.7.00
COPY --from=nvidia/cuda:11.7.0-devel-ubuntu20.04 /usr/local/cuda /usr/local/cuda
RUN printf "export CUDA_PATH=/usr/local/cuda\nexport PATH=\$CUDA_PATH/bin:\$PATH\nexport LD_LIBRARY_PATH=\$CUDA_PATH/lib64:\$LD_LIBRARY_PATH\n" >> ~/.bashrc
RUN rm -rf /opt/nvidia/nsight-compute /opt/nvidia/nsight-systems
ADD https://developer.download.nvidia.com/devtools/repos/ubuntu1804/amd64/nsight-compute-2022.4.0_2022.4.0.15-1_amd64.deb /scripts
ADD https://developer.download.nvidia.com/devtools/repos/ubuntu1804/amd64/nsight-systems-2022.5.1_2022.5.1.82-1_amd64.deb /scripts
RUN apt install -y .//scripts/nsight-compute-2022.4.0_2022.4.0.15-1_amd64.deb
RUN apt install -y .//scripts/nsight-systems-2022.5.1_2022.5.1.82-1_amd64.deb

RUN sh -c 'echo kernel.perf_event_paranoid=2 > /etc/sysctl.d/local.conf'
#### END - Install CUDA 11.7.0

#### BEGIN - Install oneAPI 2022.2.0
RUN wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18673/l_BaseKit_p_2022.2.0.262.sh -O /tmp/oneapi_install.sh
RUN sh /tmp/oneapi_install.sh -a --silent --eula accept
RUN printf "source /opt/intel/oneapi/setvars.sh\n" >> ~/.bashrc
#### END - Install oneAPI 2022.2.0


#### BEGIN - Install INTEL-LLVM
RUN sh /scripts/install-intel-llvm.sh

#Install Intel OpenCL and oneAPI TBB
ENV INTEL_OCLCPUEXP=2022.14.8.0.04_rel
ENV INTEL_OCLCPU_FOLDER=/opt/intel/oclcpuexp_$INTEL_OCLCPUEXP
ENV ONEAPI_TBB=oneapi-tbb-2021.6.0-lin
ENV INTEL_ONEAPI_TBB_FOLDER=/opt/intel/$ONEAPI_TBB
RUN wget https://github.com/intel/llvm/releases/download/2022-WW33/oclcpuexp-$INTEL_OCLCPUEXP.tar.gz -O /tmp/oclcpuexp-$INTEL_OCLCPUEXP.tar.gz
RUN wget https://github.com/oneapi-src/oneTBB/releases/download/v2021.6.0/$ONEAPI_TBB.tgz -O /tmp/ONEAPI_TBB.tgz
RUN mkdir -p $INTEL_OCLCPU_FOLDER
RUN tar -zxvf /tmp/oclcpuexp-$INTEL_OCLCPUEXP.tar.gz -C $INTEL_OCLCPU_FOLDER
RUN echo echo $INTEL_OCLCPU_FOLDER/x64/libintelocl.so > /etc/OpenCL/vendors/intel_expcpu.icd
RUN tar -zxvf /tmp/ONEAPI_TBB.tgz -C $INTEL_OCLCPU_FOLDER
RUN ln -s $INTEL_ONEAPI_TBB_FOLDER/lib/intel64/gcc4.8/libtbb.so $INTEL_OCLCPU_FOLDER/x64
RUN ln -s $INTEL_ONEAPI_TBB_FOLDER/lib/intel64/gcc4.8/libtbbmalloc.so $INTEL_OCLCPU_FOLDER/x64
RUN ln -s $INTEL_ONEAPI_TBB_FOLDER/lib/intel64/gcc4.8/libtbb.so.12 $INTEL_OCLCPU_FOLDER/x64
RUN ln -s $INTEL_ONEAPI_TBB_FOLDER/lib/intel64/gcc4.8/libtbbmalloc.so.2 $INTEL_OCLCPU_FOLDER/x64
RUN echo $INTEL_OCLCPU_FOLDER/x64 >> /etc/ld.so.conf.d/libintelopenclexp.conf
RUN ldconfig -f /etc/ld.so.conf.d/libintelopenclexp.conf
RUN printf "export DPCPP_HOME=/opt/build/intel\nexport PATH=\$DPCPP_HOME/llvm/build/bin:\$PATH\nexport LD_LIBRARY_PATH=\$DPCPP_HOME/llvm/build/lib:\$LD_LIBRARY_PATH\n" >> ~/.bashrc
#### END - Install INTEL-LLVM

#### BEGIN Install HIPSYCL & LLVM
# RUN sh /scripts/install-hipsycl-llvm.sh
# RUN sh /scripts/install-hipsycl.sh
# RUN printf "export DPCPP_HOME=/opt/hipSYCL\nexport PATH=\$DPCPP_HOME/llvm/bin:\$PATH\nexport LD_LIBRARY_PATH=\$DPCPP_HOME/llvm/lib:\$LD_LIBRARY_PATH\n" >> ~/.bashrc
#### END Install HIPSYCL & LLVM

WORKDIR /app
