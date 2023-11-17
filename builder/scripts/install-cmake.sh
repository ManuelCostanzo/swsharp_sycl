#!/bin/bash
export DEBIAN_FRONTEND=noninteractive

### BEGIN - Install CMAKE
apt purge --auto-remove cmake -y
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - |  tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
apt-add-repository 'deb https://apt.kitware.com/ubuntu/ jammy main'
apt update -y && apt install cmake -y
### END - Install CMAKE