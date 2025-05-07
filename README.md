# Voltrix-SpMM

Artifact Evaluation for the paper: 'Revitalizing Sparse Matrix-Matrix Multiplication on Tensor Cores with Asynchronous and Balanced Kernel Optimization'.

## 📦 Clone the Repository

```bash
git clone git@github.com:YaqiXia/Voltrix-SpMM.git
cd Voltrix-SpMM
```


## ⚙️ Environment Setup 

We recommend using `conda` to manage the environment.

### Step 1: Create and activate the environment 



```bash
conda create -n voltrix python=3.12
conda activate voltrix
```


### Step 2: Install dependencies 



```bash
pip install torch torchvision torchaudio
pip install scipy
```


### Step 3: Install Voltrix-SpMM in editable mode 



```bash
pip install -e .
```


### Step 4: Test JIT compilation 



```bash
python tests/test_jit.py
```


## 🧪 Run SpMM Test 



```bash
python tests/test_spmm.py
```


## 🛠️ Environment Variables 

Voltrix-SpMM supports several environment variables for debugging and optimization. These can be found in `project/const.py`. Key variables include:
| Variable | Description | 
| --- | --- | 
| VOLTRIX_JIT_DEBUG | Enables detailed debug output for JIT compilation | 
| VOLTRIX_NVCC_COMPILER | Specifies the path to a custom NVCC compiler | 
| VOLTRIX_CACHE_DIR | Sets the cache directory for compiled binaries (default: ~/.voltrix-spmm/) | 
| VOLTRIX_PTXAS_VERBOSE | Enables verbose output from the PTX compiler | 
| VOLTRIX_JIT_PRINT_NVCC_COMMAND | Prints the full NVCC command used during JIT compilation | 
| VOLTRIX_PRINT_AUTO_TUNE | Prints detailed logs during auto-tuning of kernels | 


You can set these in your shell session like this:



```bash
export VOLTRIX_JIT_DEBUG=1
export VOLTRIX_NVCC_COMPILER=/usr/local/cuda/bin/nvcc
export VOLTRIX_CACHE_DIR=~/.voltrix-spmm/
export VOLTRIX_PTXAS_VERBOSE=1
export VOLTRIX_JIT_PRINT_NVCC_COMMAND=1
export VOLTRIX_PRINT_AUTO_TUNE=1
```


## 📄 License 


This project is released under the MIT License.

