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

#define BLK_H 16
#define BLK_W 8
#define WARP_SIZE 32
#define WPB 4
#define NUM_PIP 2
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

using namespace nvcuda;

inline __device__ float our_float_to_tf32(float in) {
  float ret;
  asm volatile(
      "{\n  .reg .b32 __$1;  // TAG2"
      "\n   cvt.rna.tf32.f32 __$1, %1;"
      "\n   mov.b32 %0, __$1;\n}\n"
      : "=f"(ret)
      : "f"(in));
  return ret;
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

  void sync_all() { cudaStreamSynchronize(0); }

 private:
  cudaEvent_t events_[2];
  int cur_ = 0;
};

void checkCudaError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Error after " << msg << ": " << cudaGetErrorString(err)
              << std::endl;
  }
}

void read_csv(const std::string &filename, std::vector<float> &data) {
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

void read_csv(const std::string &filename, std::vector<int> &data) {
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

void save_to_file(const std::vector<int> &data, const std::string &filename) {
  std::ofstream out_file(filename, std::ios::binary);
  out_file.write(reinterpret_cast<const char *>(data.data()),
                 data.size() * sizeof(int));
  out_file.close();
}

void save_to_file(const std::vector<float> &data, const std::string &filename) {
  std::ofstream out_file(filename, std::ios::binary);
  out_file.write(reinterpret_cast<const char *>(data.data()),
                 data.size() * sizeof(float));
  out_file.close();
}

// 比较浮点数是否相等，考虑误差
bool are_floats_equal(float a, float b, float epsilon = 1e-6) {
  return std::fabs(a - b) < epsilon;
}

// 查找不相等元素的索引和对应的值
void find_differences(const std::vector<float> &v1,
                      const std::vector<float> &v2, float epsilon = 1e-6,
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
    std::string msg = std::string("[") + name + "] Vectors are equal!";
    std::cout << msg << std::endl;
  }
}

// condense an sorted array with duplication: [1,2,2,3,4,5,5]
// after condense, it becomes: [1,2,3,4,5].
// Also, mapping the origin value to the corresponding new location in the new
// array. 1->[0], 2->[1], 3->[2], 4->[3], 5->[4].
std::map<unsigned, unsigned> inplace_deduplication(unsigned *array,
                                                   unsigned length) {
  int loc = 0, cur = 1;
  std::map<unsigned, unsigned> nb2col;
  nb2col[array[0]] = 0;
  while (cur < length) {
    if (array[cur] != array[cur - 1]) {
      loc++;
      array[loc] = array[cur];
      nb2col[array[cur]] = loc;  // mapping from eid to TC_block column index.[]
    }
    cur++;
  }
  return nb2col;
}

template <int DO_NOT_USE =
              0>  // Useless template parameter for reduced compilation time.
void printMap(const std::map<unsigned, unsigned> &clean_edges2col) {
  for (const auto &pair : clean_edges2col) {
    std::cout << "Key: " << pair.first << ", Value: " << pair.second
              << std::endl;
  }
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>,
           std::vector<int>>
preprocess(std::vector<int> &edgeList, std::vector<int> &nodePointer,
           int num_nodes, int blockSize_h, int blockSize_w,
           std::vector<int> &blockPartition, std::vector<int> &edgeToColumn,
           std::vector<int> &edgeToRow) {
  // // input tensors.
  // auto edgeList = edgeList_tensor.accessor<int, 1>();
  // auto nodePointer = nodePointer_tensor.accessor<int, 1>();

  // // output tensors.
  // auto blockPartition = blockPartition_tensor.accessor<int, 1>();
  // auto edgeToColumn = edgeToColumn_tensor.accessor<int, 1>();
  // auto edgeToRow = edgeToRow_tensor.accessor<int, 1>();

  unsigned block_counter = 0;

#pragma omp parallel for
  for (unsigned nid = 0; nid < num_nodes; nid++) {
    for (unsigned eid = nodePointer[nid]; eid < nodePointer[nid + 1]; eid++)
      edgeToRow[eid] = nid;
  }

#pragma omp parallel for reduction(+ : block_counter)
  for (unsigned iter = 0; iter < num_nodes; iter += blockSize_h) {
    unsigned windowId = iter / blockSize_h;
    unsigned block_start = nodePointer[iter];
    unsigned block_end = nodePointer[min(iter + blockSize_h, num_nodes)];
    unsigned num_window_edges = block_end - block_start;
    unsigned *neighbor_window =
        (unsigned *)malloc(num_window_edges * sizeof(unsigned));
    memcpy(neighbor_window, &edgeList[block_start],
           num_window_edges * sizeof(unsigned));

    // Step-1: Sort the neighbor id array of a row window.
    thrust::sort(neighbor_window, neighbor_window + num_window_edges);

    // Step-2: Deduplication of the edge id array.
    // printf("Before dedupblication: %d\n", num_window_edges);
    std::map<unsigned, unsigned> clean_edges2col =
        inplace_deduplication(neighbor_window, num_window_edges);
    // printMap(clean_edges2col);

    // generate blockPartition --> number of TC_blcok in each row window.
    blockPartition[windowId] =
        (clean_edges2col.size() + blockSize_w - 1) / blockSize_w;
    block_counter += blockPartition[windowId];

    // scan the array and generate edge to column mapping. --> edge_id to
    // compressed_column_id of TC_block.
    for (unsigned e_index = block_start; e_index < block_end; e_index++) {
      unsigned eid = edgeList[e_index];
      edgeToColumn[e_index] = clean_edges2col[eid];
    }
  }
  printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter,
         block_counter * 8 * 16);

  // Step-3: Calculate Pointer1 as prefix sum of blockPartition using thrust.
  std::vector<int> Pointer1(blockPartition.size() + 1);
  thrust::inclusive_scan(blockPartition.begin(), blockPartition.end(),
                         Pointer1.begin() + 1);
  Pointer1[0] = 0;

  std::vector<int> num_edges_onerow_oneblock((BLK_H)*block_counter);
  std::vector<int> Pointer2((BLK_H)*block_counter + 1);

#pragma omp parallel for
  for (unsigned iter = 0; iter < num_nodes; iter++) {
    unsigned windowId = iter / blockSize_h;
    unsigned windowInId = iter % blockSize_h;
    unsigned block_start = nodePointer[iter];
    unsigned block_end = nodePointer[iter + 1];
    unsigned num_TC_blocks = blockPartition[windowId];
    for (unsigned i = 0; i < num_TC_blocks; i++) {
      unsigned num_within_block = 0;
      for (unsigned j = block_start; j < block_end; j++) {
        unsigned col = edgeToColumn[j];
        if (i * BLK_W <= col && col < (i + 1) * BLK_W) {
          num_within_block += 1;
        }
      }
      unsigned block_id = Pointer1[windowId] * blockSize_h;
      unsigned num_id = block_id + i * blockSize_h + windowInId;
      num_edges_onerow_oneblock[num_id] = num_within_block;
    }
    // unsigned pointer1_start = Pointer1[windowId];
    // unsigned pointer1_end = Pointer1
  }

  // printf("num_edges_onerow_oneblock: \n");
  // for (size_t i = 0; i < num_edges_onerow_oneblock.size(); ++i) {
  //     printf("%d ", num_edges_onerow_oneblock[i]);
  // }
  // printf("\n");

  thrust::inclusive_scan(num_edges_onerow_oneblock.begin(),
                         num_edges_onerow_oneblock.end(), Pointer2.begin() + 1);
  Pointer2[0] = 0;

  int num_edges = edgeList.size();
  std::vector<int> global_map(num_edges);
  std::vector<int> local_map(num_edges);

#pragma omp parallel for
  for (unsigned iter = 0; iter < num_nodes; iter++) {
    unsigned windowId = iter / blockSize_h;
    unsigned windowInId = iter % blockSize_h;
    unsigned block_start = nodePointer[iter];
    unsigned block_end = nodePointer[iter + 1];
    unsigned num_TC_blocks = blockPartition[windowId];
    unsigned block_id = Pointer1[windowId] * blockSize_h;
    for (unsigned i = 0; i < num_TC_blocks; i++) {
      unsigned num_id = block_id + i * blockSize_h + windowInId;
      unsigned num_start = Pointer2[num_id];
      unsigned count = 0;
      for (unsigned j = block_start; j < block_end; j++) {
        unsigned col = edgeToColumn[j];
        if (i * BLK_W <= col && col < (i + 1) * BLK_W) {
          global_map[num_start + count] = edgeList[j];
          local_map[num_start + count] = edgeToColumn[j] - i * blockSize_w;
          count++;
          // printf("count:%d, num_start+count:%d, j:%d, col:%d, i:%d\n",count,
          // num_start+count, j, col, i);
        }
      }
    }
  }
  return std::make_tuple(Pointer1, Pointer2, local_map, global_map);
}

template <int DO_NOT_USE =
              0>  // Useless template parameter for reduced compilation time.
int cal_max_edge(std::vector<int> &nodePointer,
                 std::vector<int> &blockPartition) {
  std::vector<int> num_edges(
      blockPartition.size());  // num_edges of each windows
  int num_nodes = nodePointer.size() - 1;
#pragma omp for
  for (int iter = 0; iter < blockPartition.size(); iter++) {
    int win_start = iter * BLK_H;
    int win_end = min((iter + 1) * BLK_H, num_nodes);
    int edges = nodePointer[win_end] - nodePointer[win_start];
    num_edges[iter] = edges;
  }
  auto max_id = std::max_element(num_edges.begin(), num_edges.end());
  int max_edge = *max_id;
  return max_edge;
}

template <int DO_NOT_USE =
              0>  // Useless template parameter for reduced compilation time.
__global__ void hmat_cuda_kernel(
    const int *__restrict__ nodePointer,     // node pointer.
    const int *__restrict__ edgeList,        // edge list.
    const int *__restrict__ blockPartition,  // number of TC_blocks (16x8) in
                                             // each row_window.
    const int *__restrict__ edgeToColumn,  // eid -> col within each row_window.
    const int *__restrict__ edgeToRow,     // eid -> col within each row_window.
    const int *__restrict__ Pointer1, const int numNodes, const int numEdges,
    float *hspa, int *hind) {
  const unsigned bid = blockIdx.x;  // block_index == row_window_index
  const unsigned wid =
      threadIdx.y;  // warp_index handling multi-dimension > 16.
  const unsigned laneid = threadIdx.x;  // lanid of each warp.
  const unsigned tid =
      threadIdx.y * blockDim.x + laneid;  // threadid of each block.
  const unsigned warpSize = blockDim.x;   // number of threads per warp.
  const unsigned threadPerBlock =
      blockDim.x * blockDim.y;  // number of threads per block.

  const unsigned nIdx_start =
      bid * BLK_H;  // starting nodeIdx of current row_window.
  const unsigned nIdx_end = min(
      (bid + 1) * BLK_H, numNodes);  // ending nodeIdx of current row_window.

  const unsigned eIdx_start =
      nodePointer[nIdx_start];  // starting edgeIdx of current row_window.
  const unsigned eIdx_end =
      nodePointer[nIdx_end];  // ending edgeIdx of the current row_window.
  const unsigned num_TC_blocks =
      blockPartition[bid];  // number of TC_blocks of the current row_window.

  const unsigned spa_start =
      Pointer1[bid] * BLK_H *
      BLK_W;  // starting idx of current window (based on local indptr).
  const unsigned ind_start = Pointer1[bid] * BLK_W;

  // Processing TC_blocks along the column dimension of Sparse A.
  for (unsigned i = 0; i < num_TC_blocks; i++) {
    int offset_spa = i * BLK_H * BLK_W + spa_start;
    int offset_ind = i * BLK_W + ind_start;

    // Init A_colToX_row with dummy values.
    if (tid < BLK_W) {
      hind[tid + offset_ind] = 0;
    }

    __syncthreads();
// Init sparse_A with zero values.
#pragma unroll
    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
      hspa[idx + offset_spa] = 0;
    }
    __syncthreads();

#pragma unroll
    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end;
         eIdx += threadPerBlock) {
      unsigned col = edgeToColumn[eIdx];
      if (i * BLK_W <= col &&
          col < (i + 1) * BLK_W) {  // if the edge in the current TC_block frame
                                    // of column.
        unsigned row_local = edgeToRow[eIdx] % BLK_H;
        unsigned col_local = col % BLK_W;
        hspa[row_local * BLK_W + col_local + offset_spa] =
            1;  // set the edge of the sparse_A.
        hind[col_local + offset_ind] =
            edgeList[eIdx];  // record the mapping from sparse_A colId to rowId
                             // of dense_X.
      }
    }
    __syncthreads();
  }
}

template <int DO_NOT_USE = 0>  // Useless template parameter for reduced
                               // compilation time.
__global__ void hmat_convert_uint32_cuda_kernel(
    const int *__restrict__ Pointer1, const float *__restrict__ hspa,
    uint32_t *packed_hspa) {
  const unsigned bid = blockIdx.x;  // block_index == row_window_index
  const unsigned wid =
      threadIdx.y;  // warp_index handling multi-dimension > 16.
  const unsigned laneid = threadIdx.x;  // lanid of each warp.
  const unsigned tid =
      threadIdx.y * blockDim.x + laneid;  // threadid of each block.
  const unsigned warpSize = blockDim.x;   // number of threads per warp.
  const unsigned threadPerBlock =
      blockDim.x * blockDim.y;  // number of threads per block.
  const unsigned num_TC_blocks =
      Pointer1[bid + 1] -
      Pointer1[bid];  // number of TC_blocks of the current row_window.
  const unsigned spa_start = Pointer1[bid] * BLK_H * BLK_W;
  const unsigned spa_packed_start = Pointer1[bid] * BLK_H * BLK_W / 32;

  for (unsigned i = 0; i < num_TC_blocks; i++) {
    int offset_spa = i * BLK_H * BLK_W + spa_start;
    int offset_spa_packed = i * BLK_H * BLK_W / 32 + spa_packed_start;
// Init sparse_A with zero values.
#pragma unroll
    for (unsigned idx = tid; idx < BLK_W * BLK_H / 32; idx += threadPerBlock) {
      uint32_t byte = 0;
      for (int bit = 0; bit < 32; ++bit) {
        int element_idx = idx * 32 + bit;
        float val = hspa[element_idx + offset_spa];
        if (val != 0.0f) {
          byte |= (1 << bit);  // 设置对应的位
        }
      }
      packed_hspa[idx + offset_spa_packed] = byte;
    }
  }
}

template <int DO_NOT_USE =
              0>  // Useless template parameter for reduced compilation time.
__global__ void spmm_forward_cuda_kernel(
    const int *__restrict__ nodePointer,     // node pointer.
    const int *__restrict__ edgeList,        // edge list.
    const int *__restrict__ blockPartition,  // number of TC_blocks (16x8) in
                                             // each row_window.
    const int *__restrict__ edgeToColumn,  // eid -> col within each row_window.
    const int *__restrict__ edgeToRow,     // eid -> col within each row_window.
    const int numNodes, const int numEdges,
    const int embedding_dim,          // embedding dimension.
    const float *__restrict__ input,  // input feature matrix.
    float *output                     // aggreAGNNed output feature matrix.
) {
  const unsigned bid = blockIdx.x;  // block_index == row_window_index
  const unsigned wid =
      threadIdx.y;  // warp_index handling multi-dimension > 16.
  const unsigned laneid = threadIdx.x;  // lanid of each warp.  [0, 31]
  const unsigned tid =
      threadIdx.y * blockDim.x + laneid;  // threadid of each block.
  const unsigned warpSize = blockDim.x;   // number of threads per warp.  32
  const unsigned threadPerBlock =
      blockDim.x * blockDim.y;  // number of threads per block.

  const unsigned dimTileNum =
      embedding_dim / BLK_H;  // number of tiles along the dimension
  const unsigned nIdx_start =
      bid * BLK_H;  // starting nodeIdx of current row_window.
  const unsigned nIdx_end = min(
      (bid + 1) * BLK_H, numNodes);  // ending nodeIdx of current row_window.

  const unsigned eIdx_start =
      nodePointer[nIdx_start];  // starting edgeIdx of current row_window.
  const unsigned eIdx_end =
      nodePointer[nIdx_end];  // ending edgeIdx of the current row_window.
  const unsigned num_TC_blocks =
      blockPartition[bid];  // number of TC_blocks of the current row_window.
  const unsigned dense_bound = numNodes * embedding_dim;
  const unsigned offset_feat = (blockIdx.y << 6);

  __shared__ float
      sparse_A[BLK_H * BLK_W];  // row-major sparse matrix shared memory store.
  __shared__ int sparse_AToX_index[BLK_W];  // TC_block col to dense_tile row.
  // __shared__ float dense_X[dimTileNum * BLK_W * BLK_H];	// column-major
  // dense tile [dimTileNum, BLK_W, BLK_H]
  extern __shared__ float dense_X[];

  wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32,
                 wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32,
                 wmma::col_major>
      b_frag;
  wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  // Processing TC_blocks along the column dimension of Sparse A.
  for (unsigned i = 0; i < num_TC_blocks; i++) {
    // Init A_colToX_row with dummy values.
    if (tid < BLK_W) {
      sparse_AToX_index[tid] = numNodes + 1;
    }

    __syncthreads();

// Init sparse_A with zero values.
#pragma unroll
    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
      sparse_A[idx] = 0;
    }

// Init dense_X with zero values.
// #pragma unroll
//     for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H;
//          idx += threadPerBlock) {
//       dense_X[idx] = 0;
//     }

// Initialize sparse_A by using BLK_H (16) threads from the warp-0.
// currently fetch all neighbors of the current nodes.
// then to see whether it can fit into current TC_block frame of column.
#pragma unroll
    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end;
         eIdx += threadPerBlock) {
      unsigned col = edgeToColumn[eIdx];
      if (i * BLK_W <= col &&
          col < (i + 1) * BLK_W) {  // if the edge in the current TC_block frame
                                    // of column.
        unsigned row_local = edgeToRow[eIdx] % BLK_H;
        unsigned col_local = col % BLK_W;
        sparse_A[row_local * BLK_W + col_local] =
            1;  // set the edge of the sparse_A.
        sparse_AToX_index[col_local] =
            edgeList[eIdx];  // record the mapping from sparse_A colId to rowId
                             // of dense_X.
      }
    }
    __syncthreads();

    // Initialize dense_X by column-major store,
    // Threads of a warp for fetching a dense_X.
    // each warp identify by wid.
    if (wid < dimTileNum)
#pragma unroll
      for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += warpSize) {
        unsigned dense_rowIdx =
            sparse_AToX_index[idx % BLK_W];   // TC_block_col to dense_tile_row.
        unsigned dense_dimIdx = idx / BLK_W;  // dimIndex of the dense tile.
        unsigned source_idx =
            dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx +
            offset_feat;  // In each row of feats, each warp for BLK_H. In all
                          // rows, each warp for BLK_W feats.
        unsigned target_idx = wid * BLK_W * BLK_H + idx;
        // boundary test.
        if (source_idx >= dense_bound) {
          // printf("source_idx: %d, dense_bound: %d\n", source_idx,
          // dense_bound);
          dense_X[target_idx] = 0;
        } else
          dense_X[target_idx] =
              input[source_idx];  // target_idx sequence will be
                                  // n1f1,n2f1,n3f1,n4f1,,,,n1f2,n2f2,n3f2,,,,
      }

    __syncthreads();

    if (wid < dimTileNum) {
      wmma::load_matrix_sync(a_frag, sparse_A, BLK_W);
      wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);

#pragma unroll
      for (unsigned t = 0; t < a_frag.num_elements; t++) {
        a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
      }

#pragma unroll
      for (unsigned t = 0; t < b_frag.num_elements; t++) {
        b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
      }
      // Perform the matrix multiplication.
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // if (blockIdx.x == 35 && threadIdx.y == 12 && threadIdx.x == 0) {
    //   printf("===================================\n");
    //   printf("k: %d", i);
    //   printf("sparse_A: \n");
    //   for (int row = 0; row < 16; row ++ ) {
    //     for (int col = 0; col <8; col ++) {
    //       printf("%f ", sparse_A[row * 8 + col]);
    //     }
    //     printf("\n");
    //   }
    //   printf("\n dense_X: \n");

    //   // i, j = i * cols + j
    //   // i, j = j * rows + i

    //   for (int row = 0; row < 8; row ++ ) {
    //     for (int col = 0; col < 16; col ++) {
    //       printf("%f ", dense_X[wid * BLK_W * BLK_H + col * BLK_W + row]);
    //     }
    //     printf("\n");
    //   }

    //   printf("output offset: %u \n", bid * BLK_H * embedding_dim + wid *
    //   BLK_H); printf("===================================\n");
    // }
  }

  if (wid < dimTileNum)
    // Store the matrix to output matrix.
    // * Note * embeeding dimension should be padded divisible by BLK_H for
    // output correctness.
    wmma::store_matrix_sync(
        output + bid * BLK_H * embedding_dim + wid * BLK_H + offset_feat,
        acc_frag, embedding_dim, wmma::mem_row_major);
}

int get_sm_count() {
  int device_id;
  cudaError_t result = cudaGetDevice(&device_id);
  if (result != cudaSuccess) {
    std::cerr << "cudaGetDevice() returned error " << cudaGetErrorString(result)
              << std::endl;
    return 1;
  }
  int multiprocessor_count;
  result = cudaDeviceGetAttribute(&multiprocessor_count,
                                  cudaDevAttrMultiProcessorCount, device_id);
  if (result != cudaSuccess) {
    std::cerr << "cudaDeviceGetAttribute() returned error "
              << cudaGetErrorString(result) << std::endl;
    return 1;
  }
  return multiprocessor_count;
}

#define HOST __forceinline__ __host__
#define HOST_DEVICE __forceinline__ __host__ __device__
#define DEVICE __forceinline__ __device__

void load_from_file(std::vector<float> &data, const std::string &filename) {
  std::ifstream in_file(filename, std::ios::binary);
  in_file.seekg(0, std::ios::end);
  size_t file_size = in_file.tellg();
  in_file.seekg(0, std::ios::beg);
  data.resize(file_size / sizeof(float));
  in_file.read(reinterpret_cast<char *>(data.data()), file_size);
  in_file.close();
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
  printf("Max registers per block: %d\n", prop.regsPerBlock);
  printf("Max shared memory per block: %zu\n", prop.sharedMemPerBlock);
  printf("SM count: %d\n", prop.multiProcessorCount);
  printf("Max warp per SM: %d\n", prop.maxThreadsPerMultiProcessor / 32);

  // 读取文件并初始化CSR矩阵的数据结构
  std::vector<float> feat;
  std::vector<int> indices, indptr;

  std::vector<int32_t> block_offsets;

  // read_csv("feat.csv", feat);
  load_from_file(feat, "feat.csv");
  read_csv("indices.csv", indices);
  read_csv("indptr.csv", indptr);
  read_csv("block_offsets.csv", block_offsets);

  std::cout << "block_offsets size: " << block_offsets.size() << std::endl;
  // for (auto ofs : block_offsets) {
  //   std::cout << ofs << " ";
  // }

  size_t num_nodes = indptr.size() - 1;
  /*
  // 打印读取结果
  std::cout << "CSR Matrix:" << std::endl;

  std::cout << "Indices: ";
  for (int idx : indices) std::cout << idx << " ";
  std::cout << std::endl;

  std::cout << "Indptr: ";
  for (size_t i = 0; i < indptr.size(); ++i) std::cout << indptr[i] << "(" << i
  << ")" << " "; std::cout << std::endl;

  std::cout << "Feats: Dense" << feat.size() << " (" << num_nodes << " x " <<
  feat.size() / num_nodes << ")" <<  std::endl; std::cout << std::endl;

      std::cout << "num_nodes: " << num_nodes << std::endl;
  */

  size_t num_edges = indices.size();
  size_t num_row_windows = (num_nodes + BLK_H - 1) / BLK_H;
  size_t embedding_dim = feat.size() / num_nodes;

  std::cout << "num_nodes: " << num_nodes << std::endl;
  std::cout << "embedding_dim: " << embedding_dim << std::endl;

  // 创建一个与 indices 长度相同的全 0 vector
  std::vector<int> edgeToColumn(num_edges, 0);
  std::vector<int> edgeToRow(num_edges, 0);
  std::vector<int> blockPartition(num_row_windows, 0);

  // preprocess(indices, indptr, num_nodes, BLK_H, BLK_W, blockPartition,
  // edgeToColumn, edgeToRow);

  // HCSR test
  std::vector<int> Pointer1, Pointer2, local_map, global_map;
  std::tie(Pointer1, Pointer2, local_map, global_map) =
      preprocess(indices, indptr, num_nodes, BLK_H, BLK_W, blockPartition,
                 edgeToColumn, edgeToRow);

  int32_t avg_blocks =
      std::accumulate(blockPartition.begin(), blockPartition.end(), 0) /
      blockPartition.size();
  int32_t max_blocks =
      *std::max_element(blockPartition.begin(), blockPartition.end());
  int32_t min_blocks =
      *std::min_element(blockPartition.begin(), blockPartition.end());
  printf("TC Blocks: [max: %d, min: %d, avg: %d]\n", max_blocks, min_blocks,
         avg_blocks);

  // Print the result of Pointer1.

  /*
  printf("blockPartition:\n");
  for (size_t i = 0; i < blockPartition.size(); ++i) {
      printf("%d ", blockPartition[i]);
  }
  printf("\n");
  printf("Pointer1 (Prefix Sum of Block Partition):\n");
  for (size_t i = 0; i < Pointer1.size(); ++i) {
      printf("%d ", Pointer1[i]);
  }
  printf("\n");

  // Print the result of Pointer2.
  printf("Pointer2 (Inclusive Sum of num_edges_onerow_oneblock):\n");
  for (size_t i = 0; i < Pointer2.size(); ++i) {
      printf("%d ", Pointer2[i]);
  }
  printf("\n");

      // Print the result of edgeToColumn.
  std::cout << "edgeToColumn: ";
  for (int j = 0; j < edgeToColumn.size(); ++j) {
          std::cout << edgeToColumn[j] << " ";
      }
  std::cout << std::endl;

      // Print the result of edge.
  std::cout << "indices: ";
  for (int j = 0; j < indices.size(); ++j) {
          std::cout << indices[j] << " ";
      }
  std::cout << std::endl;


  // Print the result of local_map.
  printf("local_map:\n");
  for (size_t i = 0; i < local_map.size(); ++i) {
      printf("%d ", local_map[i]);
  }
  printf("\n");

  // Print the result of global_map.
  printf("global_map:\n");
  for (size_t i = 0; i < global_map.size(); ++i) {
      printf("%d ", global_map[i]);
  }
  printf("\n");
*/

  // 分配 GPU 内存
  int *d_nodePointer, *d_edgeList, *d_blockPartition, *d_edgeToColumn,
      *d_edgeToRow, *d_Pointer1, *d_Pointer2, *d_local_map, *d_global_map;
  float *d_feat, *d_output, *d_output_base, *d_output1;

  int32_t *d_block_offsets;

  cudaMalloc(&d_nodePointer, indptr.size() * sizeof(int));
  cudaMalloc(&d_edgeList, indices.size() * sizeof(int));
  cudaMalloc(&d_blockPartition, blockPartition.size() * sizeof(int));
  cudaMalloc(&d_edgeToColumn, edgeToColumn.size() * sizeof(int));
  cudaMalloc(&d_edgeToRow, edgeToRow.size() * sizeof(int));
  cudaMalloc(&d_Pointer1, Pointer1.size() * sizeof(int));
  cudaMalloc(&d_Pointer2, Pointer2.size() * sizeof(int));
  cudaMalloc(&d_local_map, local_map.size() * sizeof(int));
  cudaMalloc(&d_global_map, global_map.size() * sizeof(int));
  cudaMalloc(&d_feat, feat.size() * sizeof(float));
  cudaMalloc(&d_output, num_nodes * embedding_dim * sizeof(float));
  cudaMalloc(&d_output1, num_nodes * embedding_dim * sizeof(float));
  cudaMalloc(&d_output_base, num_nodes * embedding_dim * sizeof(float));

  cudaMalloc(&d_block_offsets, block_offsets.size() * sizeof(int));

  // 将数据从 CPU 传输到 GPU
  cudaMemcpy(d_nodePointer, indptr.data(), indptr.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_edgeList, indices.data(), indices.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_blockPartition, blockPartition.data(),
             blockPartition.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_edgeToColumn, edgeToColumn.data(),
             edgeToColumn.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_edgeToRow, edgeToRow.data(), edgeToRow.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_Pointer1, Pointer1.data(), Pointer1.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_Pointer2, Pointer2.data(), Pointer2.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_local_map, local_map.data(), local_map.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_global_map, global_map.data(), global_map.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_feat, feat.data(), feat.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_block_offsets, block_offsets.data(),
             block_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
  // cudaStreamSynchronize(0);
  // for (auto x : block_offsets) {
  //   printf("%d ", x);
  // }

  // 计算数组大小
  int size_d_hspa = Pointer1.back() * BLK_W * BLK_H;
  int size_d_hind = Pointer1.back() * BLK_W;
  int size_d_hspa_packed = size_d_hspa / 32;

  // 定义 GPU 上的指针
  float *d_hspa;
  int *d_hind;
  uint32_t *d_hspa_packed;

  // 分配 GPU 内存
  cudaMalloc((void **)&d_hspa, size_d_hspa * sizeof(float));
  cudaMalloc((void **)&d_hind, size_d_hind * sizeof(int));
  cudaMalloc((void **)&d_hspa_packed, size_d_hspa_packed * sizeof(uint32_t));

  // config for hmat gen
  // const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
  // const int WARPperBlock = (dimTileNum>WPB)? dimTileNum: WPB;

  // dim3 grid(num_row_windows, 1, 1);
  // dim3 block(WARP_SIZE, WARPperBlock, 1);

  // config for unpip
  const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
  const int dynamic_shared_size =
      dimTileNum * BLK_W * BLK_H * sizeof(float);  // dynamic shared memory.
  const int WARPperBlock = (dimTileNum > WPB) ? dimTileNum : WPB;
  const int dynamic_shared_size_split =
      4 * BLK_W * BLK_H * sizeof(float);  // dynamic shared memory.

  dim3 grid(num_row_windows, 1, 1);
  dim3 block(WARP_SIZE, WARPperBlock, 1);
  dim3 block_split(WARP_SIZE, 4, 1);
  dim3 grid_split(num_row_windows, WARPperBlock / 4, 1);

  dim3 grid_ws(num_row_windows, 1, 1);
  dim3 block_ws(WARP_SIZE, WARPperBlock + 1, 1);  // one more producer warp.
  const int dynamic_shared_size_ws = 2 * dynamic_shared_size;

  checkCudaError("Preprocess kernels...");

  std::cout << "dynamic_shared_size: " << dynamic_shared_size << std::endl;
  std::cout << "dynamic_shared_size_split: " << dynamic_shared_size_split
            << std::endl;
  std::cout << "dynamic_shared_size_ws: " << dynamic_shared_size_ws
            << std::endl;

  checkCudaError("Error at here");

  //////
  checkCudaError("Result Kernel");

  auto func = [&]() {
    spmm_forward_cuda_kernel<<<grid_split, block_split,
                               dynamic_shared_size_split>>>(
        d_nodePointer, d_edgeList, d_blockPartition, d_edgeToColumn,
        d_edgeToRow, num_nodes, num_edges, embedding_dim, d_feat, d_output);
  };

  GPUTimer gpu_timer;

#ifndef DEBUG
  constexpr int iters = 100;
  constexpr int warmups = 60;
  bool do_warmup = true;
#else
  constexpr int iters = 1;
  constexpr int warmups = 1;
  bool do_warmup = false;
  printf("[DEBUG] Runing in DEBUG model\n");
#endif

  if (do_warmup) {
    for (int i = 0; i < warmups; ++i) {
      func();
    }
  }

  gpu_timer.tick();
  for (int i = 0; i < iters; ++i) {
    func();
  }

  checkCudaError("Frist Kernel");
  gpu_timer.tick();
  gpu_timer.sync_all();
  float latency = gpu_timer.report_last_ms() / float(iters);
  printf("[TC-GNN] Kernel time: %.4f ms\n", latency);

  std::vector<float> output_ref;
  std::vector<float> output_vec(num_nodes * embedding_dim);
  cudaMemcpy(output_vec.data(), d_output,
             num_nodes * embedding_dim * sizeof(float), cudaMemcpyDeviceToHost);
  load_from_file(output_ref, "output_base.csv");

  // 打印部分输出
  std::cout << "Checking differences..." << std::endl;
  try {
    find_differences(output_vec, output_ref, 1e-1, "TC-GNN kernel");
  } catch (std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
  }

  cudaFree(d_nodePointer);
  cudaFree(d_edgeList);
  cudaFree(d_blockPartition);
  cudaFree(d_edgeToColumn);
  cudaFree(d_edgeToRow);
  cudaFree(d_Pointer1);
  cudaFree(d_Pointer2);
  cudaFree(d_local_map);
  cudaFree(d_global_map);
  cudaFree(d_feat);
  cudaFree(d_output);
  cudaFree(d_hspa);
  cudaFree(d_hind);
  cudaFree(d_hspa_packed);
  return 0;
}