
nvcc -arch=sm_90a -gencode=arch=compute_90a,code=sm_90a -lcuda -std=c++17 -lcublas -lcublasLt --ptxas-options=-v --expt-relaxed-constexpr --expt-extended-lambda --ptxas-options=--register-usage-level=10 gespmm.cu -o gespmm
mv gespmm ..
nvcc -arch=sm_90a -gencode=arch=compute_90a,code=sm_90a -lcuda -std=c++17 -lcublas -lcublasLt --ptxas-options=-v --expt-relaxed-constexpr --expt-extended-lambda --ptxas-options=--register-usage-level=10 tcgnn.cu -o tcgnn
mv tcgnn ..



