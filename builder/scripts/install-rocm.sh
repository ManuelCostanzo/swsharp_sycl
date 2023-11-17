#!/bin/bash
export DEBIAN_FRONTEND=noninteractive

### BEGIN - Install ROCM
export ROCM_VERSION=5.4

wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} jammy main" | tee /etc/apt/sources.list.d/rocm.list
printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | tee /etc/apt/preferences.d/rocm-pin-600
apt update -y && apt install -y rocm-dev hip-base hip-dev hip-runtime-amd
### END - Install ROCM