#!/bin/bash

## Initialization
cd $VOLTRIX_PATH/third-party/DTC-SpMM
source init_dtc.sh
cd third_party/

## Build Glog

cd ${DTC_HOME}/third_party/glog  && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${DTC_HOME}/third_party/glog/build 
make -j VERBOSE=1
make install 


GlogPath="${DTC_HOME}/third_party/glog"
if [ -z "$GlogPath" ]
then
  echo "Defining the GLOG path is necessary, but it has not been defined."
else
  export GLOG_PATH=$GlogPath
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GLOG_PATH/build/lib
  export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$GLOG_PATH/build/include
  export LIBRARY_PATH=$LD_LIBRARY_PATH:$GLOG_PATH/build/lib
fi


## Build Sputnik

cd ${DTC_HOME}/third_party/sputnik  && mkdir build && cd build
cmake .. -DGLOG_INCLUDE_DIR=$GLOG_PATH/build/include -DGLOG_LIBRARY=$GLOG_PATH/build/lib/libglog.so -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=OFF -DBUILD_BENCHMARK=OFF -DCUDA_ARCHS="90;89;86" -DCMAKE_CUDA_FLAGS="-Xcompiler=-I/usr/include,-I/usr/include/x86_64-linux-gnu"
make VERBOSE=1

SputnikPath="${DTC_HOME}/third_party/sputnik"
if [ -z "$SputnikPath" ]
then
  echo "Defining the Sputnik path is necessary, but it has not been defined."
else
  export SPUTNIK_PATH=$SputnikPath
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SPUTNIK_PATH/build/sputnik
fi

### Build DTC-SpMM
cd ${DTC_HOME}/DTC-SpMM
TORCH_CUDA_ARCH_LIST="9.0a 9.0" python setup.py install


