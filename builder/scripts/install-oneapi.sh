#!/bin/bash
export DEBIAN_FRONTEND=noninteractive

### BEGIN - Install oneAPI 2023.2.0
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/992857b9-624c-45de-9701-f6445d845359/l_BaseKit_p_2023.2.0.49397.sh -O /tmp/oneapi_install.sh
sh /tmp/oneapi_install.sh -a --silent --eula accept

export INTEL_OCLCPUEXP="2023.16.6.0.28_rel"
export ONEAPI_TBB_VERSION="2021.10.0"
export INTEL_OCLCPU_FOLDER=/opt/intel/oclcpuexp_$INTEL_OCLCPUEXP
export ONEAPI_TBB_PATH=oneapi-tbb-$ONEAPI_TBB_VERSION-lin
export INTEL_ONEAPI_TBB_FOLDER=/opt/intel/$ONEAPI_TBB_PATH
wget https://github.com/intel/llvm/releases/download/2023-WW13/oclcpuexp-$INTEL_OCLCPUEXP.tar.gz -O /tmp/oclcpuexp-$INTEL_OCLCPUEXP.tar.gz
wget https://github.com/oneapi-src/oneTBB/releases/download/v$ONEAPI_TBB_VERSION/$ONEAPI_TBB_PATH.tgz -O /tmp/$ONEAPI_TBB_PATH.tgz
mkdir -p $INTEL_OCLCPU_FOLDER
tar -zxvf /tmp/oclcpuexp-$INTEL_OCLCPUEXP.tar.gz -C $INTEL_OCLCPU_FOLDER
echo $INTEL_OCLCPU_FOLDER/x64/libintelocl.so > /etc/OpenCL/vendors/intel_expcpu.icd
tar -zxvf /tmp/ONEAPI_TBB.tgz -C $INTEL_OCLCPU_FOLDER
ln -s $INTEL_ONEAPI_TBB_FOLDER/lib/intel64/gcc4.8/libtbb.so $INTEL_OCLCPU_FOLDER/x64
ln -s $INTEL_ONEAPI_TBB_FOLDER/lib/intel64/gcc4.8/libtbbmalloc.so $INTEL_OCLCPU_FOLDER/x64
ln -s $INTEL_ONEAPI_TBB_FOLDER/lib/intel64/gcc4.8/libtbb.so.12 $INTEL_OCLCPU_FOLDER/x64
ln -s $INTEL_ONEAPI_TBB_FOLDER/lib/intel64/gcc4.8/libtbbmalloc.so.2 $INTEL_OCLCPU_FOLDER/x64
echo $INTEL_OCLCPU_FOLDER/x64 >> /etc/ld.so.conf.d/libintelopenclexp.conf
ldconfig -f /etc/ld.so.conf.d/libintelopenclexp.conf
### END - Install oneAPI