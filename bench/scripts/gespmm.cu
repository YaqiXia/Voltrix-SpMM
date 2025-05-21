#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <csetjmp>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

// Primary header is compatible with pre-C++11, collective algorithm headers
// require C++11
#include <cooperative_groups.h>
// Optionally include for memcpy_async() collective
#include <cooperative_groups/memcpy_async.h>
// Optionally include for reduce() collective
#include <algorithm>
#include <cuda/barrier>
#include <tuple>

void load_from_file(std::vector<float>& data, const std::string& filename) {
  std::ifstream in_file(filename, std::ios::binary);
  in_file.seekg(0, std::ios::end);
  size_t file_size = in_file.tellg();
  in_file.seekg(0, std::ios::beg);
  data.resize(file_size / sizeof(float));
  in_file.read(reinterpret_cast<char*>(data.data()), file_size);
  in_file.close();
}

bool are_floats_equal(float a, float b, float epsilon = 1e-6) {
  return std::fabs(a - b) < epsilon;
}

void find_differences(const std::vector<float>& v1,
                      const std::vector<float>& v2, float epsilon = 1e-6,
                      std::string name = "") {
  if (v1.size() != v2.size()) {
    std::cout << "Vectors are of different sizes!" << std::endl;
    return;
  }

  int error_num = 0;

  bool found_difference = false;
  for (size_t i = 0; i < v1.size(); ++i) {
    if (!are_floats_equal(v1[i], v2[i], epsilon)) {
      std::cout << "Difference at index " << i << ": " << "v1[" << i
                << "] = " << v1[i] << ", " << "v2[" << i << "] = " << v2[i]
                << std::endl;
      error_num++;
      if (error_num > 5) {
        std::string error_msg = std::string("[") + name + "] Too many errors!";
        throw std::runtime_error(error_msg);
      }
      found_difference = true;
    }
  }

  if (!found_difference) {
    std::cout << "The vectors are equal!" << std::endl;
  }
}

struct Timer {
  virtual void tick() = 0;
  virtual double report_last_ms() = 0;
};

struct CPUTimer : public Timer {
  void tick() final {
    trace_[cur_] = std::chrono::high_resolution_clock::now();
    cur_ = 1 - cur_;
  }

  double report_last_ms() final {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        trace_[1 - cur_] - trace_[cur_]);

    return duration.count() / 1e3;
  }

 private:
  decltype(std::chrono::high_resolution_clock::now()) trace_[2];
  int cur_ = 0;
};

struct GPUTimer : public Timer {
  GPUTimer() {
    cudaEventCreate(&events_[0]);
    cudaEventCreate(&events_[1]);
  }

  ~GPUTimer() {
    cudaEventDestroy(events_[0]);
    cudaEventDestroy(events_[1]);
  }

  void tick() final {
    cudaEventRecord(events_[cur_]);
    cur_ = 1 - cur_;
  }

  double report_last_ms() final {
    float ms;
    cudaEventElapsedTime(&ms, events_[cur_], events_[1 - cur_]);
    return ms;
  }

  void sync_all() { cudaDeviceSynchronize(); }

 private:
  cudaEvent_t events_[2];
  int cur_ = 0;
};

void read_csv(const std::string& filename, std::vector<float>& data) {
  std::ifstream file(filename);
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string value;
    while (std::getline(ss, value, ',')) {
      data.push_back(std::stof(value));
    }
  }
}

void read_csv(const std::string& filename, std::vector<int>& data) {
  std::ifstream file(filename);
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string value;
    while (std::getline(ss, value, ',')) {
      data.push_back(std::stoi(value));
    }
  }
}

void checkCudaError(const char* msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Error after " << msg << ": " << cudaGetErrorString(err)
              << std::endl;
  }
}

__device__ __forceinline__ float sum_reduce(float acc, float x) {
  return acc + x;
}

__device__ __forceinline__ float4 sum_reduce(float4 acc, float4 x) {
  float4 out;

  out.x = acc.x + x.x;
  out.y = acc.y + x.y;
  out.z = acc.z + x.z;
  out.w = acc.w + x.w;

  return out;
}

__device__ __forceinline__ float2 sum_reduce(float2 acc, float2 x) {
  float2 out;

  out.x = acc.x + x.x;
  out.y = acc.y + x.y;
  return out;
}

__device__ __forceinline__ float sum_init() { return 0; }

__global__ void topoCacheCoarsenSPMMKernelE1(int m, int k, const int* A_indptr,
                                             const int* A_indices,
                                             const float* B, float* C) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y << 5);  // 32*threadIdx.y
  int thread_idx = sm_offset + threadIdx.x;

  int rid = blockDim.y * blockIdx.x + threadIdx.y;
  if (rid < m) {
    int cid = (blockIdx.y << 5) + threadIdx.x;
    int lb = A_indptr[rid];
    int hb = A_indptr[rid + 1];
    int ptr = lb + threadIdx.x;
    int offset;
    float acc1 = sum_init();
    // float acc2 = sum_init();
    if (blockIdx.y != gridDim.y - 1) {
      for (int jj = lb; jj < hb; jj += 32) {
        if (ptr < hb) {
          sh[thread_idx] = A_indices[ptr] * k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
          offset = sh[(sm_offset + kk)] + cid;
          acc1 = sum_reduce(acc1, B[offset]);
          // acc2 = sum_reduce(acc2, B[(offset + 32)]);
          // acc1 = sum_reduce(acc1, __ldg(B+offset));
          // acc2 = sum_reduce(acc2, __ldg(B+offset+32));
        }
        __syncwarp();
      }
      offset = rid * k + cid;
      C[offset] = acc1;
      // C[offset + 32] = acc2;
    } else {  // threadIdx.y==blockDim.y-1
      int nout = (k - cid + 31) / 32;
      for (int jj = lb; jj < hb; jj += 32) {
        if (ptr < hb) {
          sh[thread_idx] = A_indices[ptr] * k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
          offset = sh[(sm_offset + kk)] + cid;
          if (nout > 0) {
            acc1 = sum_reduce(acc1, B[offset]);
          }
          // acc1 = sum_reduce(acc1, __ldg(B+offset)); }
          if (nout > 1) {
            // acc2 = sum_reduce(acc2, B[(offset + 32)]);
          }
          // acc2 = sum_reduce(acc2, __ldg(B+offset+32));}
        }
        __syncwarp();
      }
      offset = rid * k + cid;
      if (nout > 0) {
        C[offset] = acc1;
      }
      if (nout > 1) {
        // C[offset + 32] = acc2;
      }
    }
  }
}

__global__ void topoCacheCoarsenSPMMKernel(int m, int k, const int* A_indptr,
                                           const int* A_indices, const float* B,
                                           float* C) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y << 5);  // 32*threadIdx.y
  int thread_idx = sm_offset + threadIdx.x;

  int rid = blockDim.y * blockIdx.x + threadIdx.y;
  if (rid < m) {
    int cid = (blockIdx.y << 6) + threadIdx.x;
    int lb = A_indptr[rid];
    int hb = A_indptr[rid + 1];
    int ptr = lb + threadIdx.x;
    int offset;
    float acc1 = sum_init();
    float acc2 = sum_init();
    if (blockIdx.y != gridDim.y - 1) {
      for (int jj = lb; jj < hb; jj += 32) {
        if (ptr < hb) {
          sh[thread_idx] = A_indices[ptr] * k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
          offset = sh[(sm_offset + kk)] + cid;
          acc1 = sum_reduce(acc1, B[offset]);
          acc2 = sum_reduce(acc2, B[(offset + 32)]);
          // acc1 = sum_reduce(acc1, __ldg(B+offset));
          // acc2 = sum_reduce(acc2, __ldg(B+offset+32));
        }
        __syncwarp();
      }
      offset = rid * k + cid;
      C[offset] = acc1;
      C[offset + 32] = acc2;
    } else {  // threadIdx.y==blockDim.y-1
      int nout = (k - cid + 31) / 32;
      for (int jj = lb; jj < hb; jj += 32) {
        if (ptr < hb) {
          sh[thread_idx] = A_indices[ptr] * k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
          offset = sh[(sm_offset + kk)] + cid;
          if (nout > 0) {
            acc1 = sum_reduce(acc1, B[offset]);
          }
          // acc1 = sum_reduce(acc1, __ldg(B+offset)); }
          if (nout > 1) {
            acc2 = sum_reduce(acc2, B[(offset + 32)]);
          }
          // acc2 = sum_reduce(acc2, __ldg(B+offset+32));}
        }
        __syncwarp();
      }
      offset = rid * k + cid;
      if (nout > 0) {
        C[offset] = acc1;
      }
      if (nout > 1) {
        C[offset + 32] = acc2;
      }
    }
  }
}

#define INT2(ptr) (*(reinterpret_cast<const int2*>(ptr)))
#define FLOAT4(ptr) (*(reinterpret_cast<float4*>(ptr)))
#define C_FLOAT4(ptr) (*(reinterpret_cast<const float4*>(ptr)))

#define FLOAT2(ptr) (*(reinterpret_cast<float2*>(ptr)))
#define C_FLOAT2(ptr) (*(reinterpret_cast<const float2*>(ptr)))

__global__ void topoCacheCoarsenSPMMKernelFloat2(int m, int k,
                                                 const int* A_indptr,
                                                 const int* A_indices,
                                                 const float* B, float* C) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y << 5);  // 32*threadIdx.y
  int thread_idx = sm_offset + threadIdx.x;

  int rid = blockDim.y * blockIdx.x + threadIdx.y;
  if (rid < m) {
    int cid = (blockIdx.y << 6) + threadIdx.x * 2;
    int lb = A_indptr[rid];
    int hb = A_indptr[rid + 1];
    int ptr = lb + threadIdx.x;
    int offset;
    // float acc1 = sum_init();
    // float acc2 = sum_init();
    float2 accs = {0, 0};

    for (int jj = lb; jj < hb; jj += 32) {
      if (ptr < hb) {
        sh[thread_idx] = A_indices[ptr] * k;
        // sh[thread_idx] = __ldg(A_indices+ptr)*k;
      }
      __syncwarp();
      ptr += 32;
      for (int kk = 0; kk < 32 && jj + kk < hb; kk++) {
        offset = sh[(sm_offset + kk)] + cid;

        accs = sum_reduce(accs, C_FLOAT2(&B[offset]));

        // acc1 = sum_reduce(acc1, B[offset]);
        // acc2 = sum_reduce(acc2, B[(offset + 32)]);
      }
      __syncwarp();
    }
    offset = rid * k + cid;
    // C[offset] = acc1;
    // C[offset + 32] = acc2;
    FLOAT2(&C[offset]) = accs;
  }
}

__global__ void topoCacheCoarsenSPMMKernelFloat4(int m, int k,
                                                 const int* A_indptr,
                                                 const int* A_indices,
                                                 const float* B, float* C) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y << 5);  // 32*threadIdx.y
  int thread_idx = sm_offset + threadIdx.x;

  int rid = blockDim.y * blockIdx.x + threadIdx.y;
  if (rid < m) {
    int cid = (blockIdx.y << 7) + threadIdx.x * 4;  // 6 -> 7, *4

    int lb = A_indptr[rid];
    int hb = A_indptr[rid + 1];
    int ptr = lb + threadIdx.x;
    int offset;
    // float acc1 = sum_init();
    // float acc2 = sum_init();
    float4 accs = {0, 0, 0, 0};

    for (int jj = lb; jj < hb; jj += 32) {
      if (ptr < hb) {
        sh[thread_idx] = A_indices[ptr] * k;
        // sh[thread_idx] = __ldg(A_indices+ptr)*k;
      }
      __syncwarp();
      ptr += 32;
      for (int kk = 0; kk < 32 && jj + kk < hb;
           kk++) {  // accumulate 128 elements per thread.
        offset = sh[(sm_offset + kk)] + cid;

        accs = sum_reduce(accs, C_FLOAT4(&B[offset]));

        // acc1 = sum_reduce(acc1, B[offset]);
        // acc2 = sum_reduce(acc2, B[(offset + 32)]);
      }
      __syncwarp();
    }
    offset = rid * k + cid;

    FLOAT4(&C[offset]) = accs;

    // C[offset] = acc1;
    // C[offset + 32] = acc2;
  }
}

__global__ void topoCacheCoarsenSPMMKernelE4(int m, int k,
                                                   const int* A_indptr,
                                                   const int* A_indices,
                                                   const float* B, float* C) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y << 5);  // 32*threadIdx.y
  int thread_idx = sm_offset + threadIdx.x;

  int rid = blockDim.y * blockIdx.x + threadIdx.y;
  if (rid < m) {
    int cid = (blockIdx.y << 7) + threadIdx.x;  // 6 -> 7

    int lb = A_indptr[rid];
    int hb = A_indptr[rid + 1];
    int ptr = lb + threadIdx.x;
    int offset;

    float acc1 = sum_init();
    float acc2 = sum_init();
    float acc3 = sum_init();
    float acc4 = sum_init();
    // float4 accs = {0, 0, 0, 0};

    for (int jj = lb; jj < hb; jj += 32) {
      if (ptr < hb) {
        sh[thread_idx] = A_indices[ptr] * k;
        // sh[thread_idx] = __ldg(A_indices+ptr)*k;
      }
      __syncwarp();
      ptr += 32;
      for (int kk = 0; kk < 32 && jj + kk < hb;
           kk++) {  // accumulate 128 elements per thread.
        offset = sh[(sm_offset + kk)] + cid;

        // accs = sum_reduce(accs, C_FLOAT4(&B[offset]));

        acc1 = sum_reduce(acc1, B[offset]);
        acc2 = sum_reduce(acc2, B[(offset + 32 * 1)]);
        acc3 = sum_reduce(acc3, B[(offset + 32 * 2)]);
        acc4 = sum_reduce(acc4, B[(offset + 32 * 3)]);
      }
      __syncwarp();
    }
    offset = rid * k + cid;

    // FLOAT4(&C[offset]) = accs;

    C[offset] = acc1;
    C[offset + 32 * 1] = acc2;
    C[offset + 32 * 2] = acc3;
    C[offset + 32 * 3] = acc4;
  }
}

int main() {
  // Read files and initialize CSR matrix data structures
  std::vector<float> feat;
  std::vector<int> indices, offsets;

  // read_csv("feat.csv", feat);
  load_from_file(feat, "feat.csv");
  read_csv("indices.csv", indices);
  read_csv("indptr.csv", offsets);

  size_t batch_size = offsets.size() - 1;
  size_t num_indices = indices.size();
  size_t embedding_dim = feat.size() / batch_size;

  std::cout << "batch_size: " << batch_size << ", num_indices: " << num_indices
            << ", embedding_dim: " << embedding_dim << std::endl;

  int *d_indices, *d_offsets;
  float *d_feat, *d_output;

  int32_t* d_rowidx;

  cudaMalloc(&d_offsets, offsets.size() * sizeof(int));
  cudaMalloc(&d_indices, indices.size() * sizeof(int));
  cudaMalloc(&d_feat, feat.size() * sizeof(float));
  cudaMalloc(&d_output, batch_size * embedding_dim * sizeof(float));

  cudaMalloc(&d_rowidx, num_indices * sizeof(decltype(*d_rowidx)));

  // 将数据从 CPU 传输到 GPU
  cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, indices.data(), indices.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_feat, feat.data(), feat.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemset(d_output, 0, batch_size * embedding_dim * sizeof(float));

  GPUTimer gpu_timer;

  constexpr int iters = 100;
  auto k = embedding_dim;
  auto m = batch_size;
  checkCudaError("0");

/*
  {
    const int tile_k = (k + 31) / 32;
    const int n_block = (m + 8 - 1) / 8;
    printf("Launching kernel with grid(%d, %d, 1), block(32, 8, 1)\n", n_block,
           tile_k);
    topoCacheCoarsenSPMMKernelE1<<<dim3(n_block, tile_k, 1), dim3(32, 8, 1),
                                   8 * 32 * sizeof(int)>>>(
        m, k, d_offsets, d_indices, d_feat, d_output);

    checkCudaError("1");

    cudaStreamSynchronize(0);

    std::vector<float> output_ref;
    std::vector<float> output_vec(batch_size * embedding_dim);

    cudaMemcpy(output_vec.data(), d_output,
               batch_size * embedding_dim * sizeof(float),
               cudaMemcpyDeviceToHost);
    load_from_file(output_ref, "output_base.csv");

    std::cout << "Checking differences..." << std::endl;
    try {
      find_differences(output_vec, output_ref, 1e-1, "Embedding kernel");
      std::cout << "Test passed!" << std::endl;
    } catch (std::runtime_error& e) {
      std::cerr << e.what() << std::endl;
    }

    auto func = [&]() {
      topoCacheCoarsenSPMMKernelE1<<<dim3(n_block, tile_k, 1), dim3(32, 8, 1),
                                     8 * 32 * sizeof(int)>>>(
          m, k, d_offsets, d_indices, d_feat, d_output);
    };

    for (int i = 0; i < iters; ++i) func();

    gpu_timer.sync_all();
    gpu_timer.tick();
    for (int i = 0; i < iters; ++i) {
      func();
    }
    gpu_timer.tick();
    gpu_timer.sync_all();
    float latency = gpu_timer.report_last_ms() / float(iters);

    double bandwidth = (batch_size * embedding_dim * sizeof(float) * 2) /
                       (latency / 1000.0) / (1024 * 1024 * 1024);
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

    printf("[GE-SPMM-E1] Embedding time: %.4f ms\n", latency);
  }
*/

  {
    const int tile_k = (k + 63) / 64;
    const int n_block = (m + 8 - 1) / 8;
    printf("Launching kernel with grid(%d, %d, 1), block(32, 8, 1)\n", n_block,
           tile_k);
    topoCacheCoarsenSPMMKernel<<<dim3(n_block, tile_k, 1), dim3(32, 8, 1),
                                 8 * 32 * sizeof(int)>>>(
        m, k, d_offsets, d_indices, d_feat, d_output);

    checkCudaError("1");

    cudaStreamSynchronize(0);

    std::vector<float> output_ref;
    std::vector<float> output_vec(batch_size * embedding_dim);

    cudaMemcpy(output_vec.data(), d_output,
               batch_size * embedding_dim * sizeof(float),
               cudaMemcpyDeviceToHost);
    load_from_file(output_ref, "output_base.csv");

    std::cout << "Checking differences..." << std::endl;
    try {
      find_differences(output_vec, output_ref, 1e-1, "Embedding kernel");
      std::cout << "Test passed!" << std::endl;
    } catch (std::runtime_error& e) {
      std::cerr << e.what() << std::endl;
    }

    auto func = [&]() {
      topoCacheCoarsenSPMMKernel<<<dim3(n_block, tile_k, 1), dim3(32, 8, 1),
                                   8 * 32 * sizeof(int)>>>(
          m, k, d_offsets, d_indices, d_feat, d_output);
    };

    for (int i = 0; i < iters; ++i) func();

    gpu_timer.sync_all();
    gpu_timer.tick();
    for (int i = 0; i < iters; ++i) {
      func();
    }
    gpu_timer.tick();
    gpu_timer.sync_all();
    float latency = gpu_timer.report_last_ms() / float(iters);

    double bandwidth = (batch_size * embedding_dim * sizeof(float) * 2) /
                       (latency / 1000.0) / (1024 * 1024 * 1024);
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

    printf("[GE-SPMM] Embedding time: %.4f ms\n", latency);
  }
  {
    const int tile_k = (k + 63) / 64;
    const int n_block = (m + 8 - 1) / 8;
    printf("Launching kernel with grid(%d, %d, 1), block(32, 8, 1)\n", n_block,
           tile_k);
    topoCacheCoarsenSPMMKernelFloat2<<<dim3(n_block, tile_k, 1), dim3(32, 8, 1),
                                       8 * 32 * sizeof(int)>>>(
        m, k, d_offsets, d_indices, d_feat, d_output);

    checkCudaError("1");

    cudaStreamSynchronize(0);

    std::vector<float> output_ref;
    std::vector<float> output_vec(batch_size * embedding_dim);

    cudaMemcpy(output_vec.data(), d_output,
               batch_size * embedding_dim * sizeof(float),
               cudaMemcpyDeviceToHost);
    load_from_file(output_ref, "output_base.csv");

    std::cout << "Checking differences..." << std::endl;
    try {
      find_differences(output_vec, output_ref, 1e-1, "Embedding kernel");
      std::cout << "Test passed!" << std::endl;
    } catch (std::runtime_error& e) {
      std::cerr << e.what() << std::endl;
    }

    auto func = [&]() {
      topoCacheCoarsenSPMMKernelFloat2<<<
          dim3(n_block, tile_k, 1), dim3(32, 8, 1), 8 * 32 * sizeof(int)>>>(
          m, k, d_offsets, d_indices, d_feat, d_output);
    };

    for (int i = 0; i < iters; ++i) func();

    gpu_timer.sync_all();
    gpu_timer.tick();
    for (int i = 0; i < iters; ++i) {
      func();
    }
    gpu_timer.tick();
    gpu_timer.sync_all();
    float latency = gpu_timer.report_last_ms() / float(iters);

    double bandwidth = (batch_size * embedding_dim * sizeof(float) * 2) /
                       (latency / 1000.0) / (1024 * 1024 * 1024);
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

    printf("[GE-SPMM-Float2] Embedding time: %.4f ms\n", latency);
  }

  {
    if (k < 128 && k % 128 == 0) {
      throw std::runtime_error(
          "k must be larger than 128 for vectorized GE-SPMM implementation.");
    }
    const int tile_k = (k + 127) / 128;
    const int n_block = (m + 8 - 1) / 8;
    printf("Launching kernel with grid(%d, %d, 1), block(32, 8, 1)\n", n_block,
           tile_k);
    topoCacheCoarsenSPMMKernelFloat4<<<dim3(n_block, tile_k, 1), dim3(32, 8, 1),
                                       8 * 32 * sizeof(int)>>>(
        m, k, d_offsets, d_indices, d_feat, d_output);

    checkCudaError("1");

    cudaStreamSynchronize(0);

    std::vector<float> output_ref;
    std::vector<float> output_vec(batch_size * embedding_dim);

    cudaMemcpy(output_vec.data(), d_output,
               batch_size * embedding_dim * sizeof(float),
               cudaMemcpyDeviceToHost);
    load_from_file(output_ref, "output_base.csv");

    std::cout << "Checking differences..." << std::endl;
    try {
      find_differences(output_vec, output_ref, 1e-1, "Embedding kernel");
      std::cout << "Test passed!" << std::endl;
    } catch (std::runtime_error& e) {
      std::cerr << e.what() << std::endl;
    }

    auto func = [&]() {
      topoCacheCoarsenSPMMKernelFloat4<<<
          dim3(n_block, tile_k, 1), dim3(32, 8, 1), 8 * 32 * sizeof(int)>>>(
          m, k, d_offsets, d_indices, d_feat, d_output);
    };

    for (int i = 0; i < iters; ++i) func();

    gpu_timer.sync_all();
    gpu_timer.tick();
    for (int i = 0; i < iters; ++i) {
      func();
    }
    gpu_timer.tick();
    gpu_timer.sync_all();
    float latency = gpu_timer.report_last_ms() / float(iters);

    double bandwidth = (batch_size * embedding_dim * sizeof(float) * 2) /
                       (latency / 1000.0) / (1024 * 1024 * 1024);
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

    printf("[GE-SPMM-Float4] Embedding time: %.4f ms\n", latency);
  }

{
    if (k < 128 && k % 128 == 0) {
      throw std::runtime_error(
          "k must be larger than 128 for vectorized GE-SPMM implementation.");
    }
    const int tile_k = (k + 127) / 128;
    const int n_block = (m + 8 - 1) / 8;
    printf("Launching kernel with grid(%d, %d, 1), block(32, 8, 1)\n", n_block,
           tile_k);
    topoCacheCoarsenSPMMKernelE4<<<dim3(n_block, tile_k, 1), dim3(32, 8, 1),
                                       8 * 32 * sizeof(int)>>>(
        m, k, d_offsets, d_indices, d_feat, d_output);

    checkCudaError("1");

    cudaStreamSynchronize(0);

    std::vector<float> output_ref;
    std::vector<float> output_vec(batch_size * embedding_dim);

    cudaMemcpy(output_vec.data(), d_output,
               batch_size * embedding_dim * sizeof(float),
               cudaMemcpyDeviceToHost);
    load_from_file(output_ref, "output_base.csv");

    std::cout << "Checking differences..." << std::endl;
    try {
      find_differences(output_vec, output_ref, 1e-1, "Embedding kernel");
      std::cout << "Test passed!" << std::endl;
    } catch (std::runtime_error& e) {
      std::cerr << e.what() << std::endl;
    }

    auto func = [&]() {
      topoCacheCoarsenSPMMKernelE4<<<
          dim3(n_block, tile_k, 1), dim3(32, 8, 1), 8 * 32 * sizeof(int)>>>(
          m, k, d_offsets, d_indices, d_feat, d_output);
    };

    for (int i = 0; i < iters; ++i) func();

    gpu_timer.sync_all();
    gpu_timer.tick();
    for (int i = 0; i < iters; ++i) {
      func();
    }
    gpu_timer.tick();
    gpu_timer.sync_all();
    float latency = gpu_timer.report_last_ms() / float(iters);

    double bandwidth = (batch_size * embedding_dim * sizeof(float) * 2) /
                       (latency / 1000.0) / (1024 * 1024 * 1024);
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

    printf("[GE-SPMM-E4] Embedding time: %.4f ms\n", latency);
  }

  checkCudaError("Error");
}