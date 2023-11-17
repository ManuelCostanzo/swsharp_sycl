#!/bin/bash
export DEBIAN_FRONTEND=noninteractive

#Reference => https://intel.github.io/llvm-docs/GetStartedGuide.html#use-dpc-toolchain

export INSTALL_PATH=/tmp/compilers

mkdir -p $INSTALL_PATH
cd $INSTALL_PATH
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.14828.8/intel-igc-core_1.0.14828.8_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.14828.8/intel-igc-media_1.0.14828.8_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.14828.8/intel-igc-opencl_1.0.14828.8_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.14828.8/intel-igc-opencl-devel_1.0.14828.8_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-level-zero-gpu-dbgsym_1.3.26918.9_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-level-zero-gpu_1.3.26918.9_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-opencl-icd-dbgsym_23.30.26918.9_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-opencl-icd_23.30.26918.9_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/libigdgmm12_22.3.0_amd64.deb

wget https://github.com/intel/cm-compiler/releases/download/cmclang-1.0.144/intel-igc-cm_1.0.144+i75_u20.04_amd64.deb

wget https://github.com/oneapi-src/level-zero/releases/download/v1.14.0/level-zero-devel_1.14.0+u22.04_amd64.deb
wget https://github.com/oneapi-src/level-zero/releases/download/v1.14.0/level-zero_1.14.0+u22.04_amd64.deb
wget https://github.com/oneapi-src/level-zero/releases/download/v1.14.0/level-zero-sdk_1.14.0.zip -O $INSTALL_PATH/level-zero-sdk.zip
unzip $INSTALL_PATH/level-zero-sdk.zip -d $INSTALL_PATH/level-zero-sdk
cp -r $INSTALL_PATH/level-zero-sdk/include/* /usr/local/include/
cp -r $INSTALL_PATH/level-zero-sdk/lib/* /usr/local/lib/

wget https://storage.googleapis.com/spirv-tools/artifacts/prod/graphics_shader_compiler/spirv-tools/linux-clang-release/continuous/2216/20231102-103029/install.tgz -O $INSTALL_PATH/spirtv_tools.tgz
mkdir $INSTALL_PATH/spirtv_tools
tar -xzvf $INSTALL_PATH/spirtv_tools.tgz -C $INSTALL_PATH/spirtv_tools
cp -r $INSTALL_PATH/spirtv_tools/install/bin/* /usr/local/bin/
cp -r $INSTALL_PATH/spirtv_tools/install/include/* /usr/local/include/
cp -r $INSTALL_PATH/spirtv_tools/install/lib/* /usr/local/lib/

dpkg -i *deb
