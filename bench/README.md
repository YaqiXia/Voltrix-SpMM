# Steps of reproduction

```bash
export VOLTRIX_PATH=`pwd`
```

## 1. Build third-party libraries


```bash
git submodule update --init --recursive
```

- 1. TCGNN-PyTorch

```bash
cd $VOLTRIX_PATH/third-party/TC-GNN/TCGNN_conv
TORCH_CUDA_ARCH_LIST="9.0a;9.0" python setup.py install
```

- 2. DTC-SpMM

```bash
cd $VOLTRIX_PATH/third-party/
bash ./build_dtc.sh
```

- 3. RoDe

```bash
cd $VOLTRIX_PATH/third-party/RoDe
git apply ../../patches/rode_fix.patch


mkdir build && cd build
GLOG_PATH=$VOLTRIX_PATH/third-party/DTC-SpMM/third_party/glog cmake .. -DCMAKE_CUDA_FLAGS="--use_fast_math -Xcompiler=-fopenmp -gencode arch=compute_90,code=sm_90 -Xcompiler=-I/usr/include,-I/usr/include/x86_64-linux-gnu" -DCMAKE_CUDA_ARCHITECTURES="" -DGLOG_INCLUDE_DIR=$GLOG_PATH/build/include -DGLOG_LIBRARY=$GLOG_PATH/build/lib/libglog.so 
make -j
```

## 2. Build scripts

```bash
cd $VOLTRIX_PATH/bench/scripts
bash build.sh
```

## 3. download datasets
```bash
cd $VOLTRIX_PATH
wget https://drive.google.com/file/d/1MuMsRr_Swi66isVHZYNkCWTnPblROT2g/view?usp=sharing
unzip datasets.zip
...

```

## 4. Set environment variables

```bash
export TCGNN_PATH=$VOLTRIX_PATH/third-party/TC-GNN
export DATASET_PATH=$VOLTRIX_PATH/datasets
export RODE_HOME=$VOLTRIX_PATH/third-party/RoDe
export LD_LIBRARY_PATH=$VOLTRIX_PATH/third-party/DTC-SpMM/third_party/sputnik/build/sputnik:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$VOLTRIX_PATH/third-party/DTC-SpMM/third_party/glog/build:$LD_LIBRARY_PATH
```


## 5. Run the benchmark

```bash
cd bench
python bench_all.py

```

