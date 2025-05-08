#!/bin/bash


cd $VOLTRIX_PATH/third-party/DTC-SpMM
source init_dtc.sh

rm ${DTC_HOME}/third_party/glog/build -r
rm ${DTC_HOME}/third_party/sputnik/build -r

