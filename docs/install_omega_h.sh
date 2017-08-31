#!/bin/sh

git clone https://github.com/LoicMarechal/libMeshb.git
cd libMeshb
cmake -DCMAKE_INSTALL_PREFIX:PATH=$PWD/install
make install
cd ..
git clone https://github.com/ibaned/omega_h.git
cd omega_h
cmake -DCMAKE_INSTALL_PREFIX:PATH=$PWD/install \
 -DOmega_h_USE_libMeshb:BOOL=ON \
 -DlibMeshb_PREFIX:PATH=$PWD/../libMeshb/install
make install -j 4
cd ..
