#!/bin/bash
export DEBIAN_FRONTEND=noninteractive

### BEGIN - Install LLVM 16
export LLVM_VERSION=15

ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so
apt update -y && apt install -y libc++-dev libc++abi-dev libstdc++-11 libstdc++-11-dev
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh $LLVM_VERSION
apt install -y libclang-$LLVM_VERSION-dev clang-tools-$LLVM_VERSION libomp-$LLVM_VERSION-dev llvm-$LLVM_VERSION-dev lld-$LLVM_VERSION

echo "export LLVM_PATH=/usr/lib/llvm-$LLVM_VERSION" >> ~/.bashrc
echo 'export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:$CPLUS_INCLUDE_PATH' >> ~/.bashrc
echo 'export C_INCLUDE_PATH=/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:$C_INCLUDE_PATH' >> ~/.bashrc
### END - Install LLVM 16