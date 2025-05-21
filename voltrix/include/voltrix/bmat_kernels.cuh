#ifndef VOLTRIX_BMAT_KERNELS_CUH_
#define VOLTRIX_BMAT_KERNELS_CUH_

#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "voltrix/traits.h"

namespace voltrix {

////////////////////////////////////////////
//
// SPMM Foward Pass  (GCN, GraphSAGE)
//
////////////////////////////////////////////
__global__ void hmat_cuda_kernel(
    const int *__restrict__ nodePointer,    // node pointer.
    const int *__restrict__ edgeList,       // edge list.
    const int *__restrict__ blockPartition, // number of TC_blocks (16x8) in
                                            // each row_window.
    const int *__restrict__ edgeToColumn, // eid -> col within each row_window.
    const int *__restrict__ edgeToRow,    // eid -> col within each row_window.
    const int *__restrict__ Pointer1, const int numNodes, const int numEdges,
    float *hspa, int *hind) {
  const unsigned bid = blockIdx.x;  // block_index == row_window_index
  const unsigned wid = threadIdx.y; // warp_index handling multi-dimension > 16.
  const unsigned laneid = threadIdx.x; // lanid of each warp.
  const unsigned tid =
      threadIdx.y * blockDim.x + laneid; // threadid of each block.
  const unsigned warpSize = blockDim.x;  // number of threads per warp.
  const unsigned threadPerBlock =
      blockDim.x * blockDim.y; // number of threads per block.

  const unsigned nIdx_start =
      bid * BLK_H; // starting nodeIdx of current row_window.
  const unsigned nIdx_end =
      min((bid + 1) * BLK_H, numNodes); // ending nodeIdx of current row_window.

  const unsigned eIdx_start =
      nodePointer[nIdx_start]; // starting edgeIdx of current row_window.
  const unsigned eIdx_end =
      nodePointer[nIdx_end]; // ending edgeIdx of the current row_window.
  const unsigned num_TC_blocks =
      blockPartition[bid]; // number of TC_blocks of the current row_window.

  const unsigned spa_start =
      Pointer1[bid] * BLK_H *
      BLK_W; // starting idx of current window (based on local indptr).
  // const unsigned spa_end = Pointer1[bid+1] * BLK_H * BLK_W; // ending idx of
  // current window.
  const unsigned ind_start = Pointer1[bid] * BLK_W;
  // const unsigned ind_end = Pointer1[bid+1] * BLK_H;

  const int last_bid = gridDim.x;

  __shared__ float
      sparse_A[BLK_H * BLK_W]; // row-major sparse matrix shared memory store.
  __shared__ int sparse_AToX_index[BLK_W]; // TC_block col to dense_tile row.

  // Processing TC_blocks along the column dimension of Sparse A.
  for (unsigned i = 0; i < num_TC_blocks; i++) {
    int offset_spa = i * BLK_H * BLK_W + spa_start;
    int offset_ind = i * BLK_W + ind_start;

    // Init A_colToX_row with dummy values.
    if (tid < BLK_W) {
      hind[tid + offset_ind] = 0;
    }
    __syncthreads();
// if(bid == last_bid-1 && tid == BLK_W -1){
// 	printf("tid + offset_ind:%d, hind[idx + offset_ind]:%d\n", tid +
// offset_ind, hind[tid + offset_ind]);
// }
// Init sparse_A with zero values.
#pragma unroll
    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
      hspa[idx + offset_spa] = 0;
    }
    __syncthreads();

// if(bid == last_bid-1 && tid == BLK_W * BLK_H -1) {
// 	printf("tid+offset_spa:%d, hspa[idx + offset_spa]:%d\n", tid +
// offset_spa, hspa[tid + offset_spa]);
// }
// Initialize sparse_A by using BLK_H (16) threads from the warp-0.
// currently fetch all neighbors of the current nodes.
// then to see whether it can fit into current TC_block frame of column.
#pragma unroll
    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end;
         eIdx += threadPerBlock) {
      unsigned col = edgeToColumn[eIdx];
      if (i * BLK_W <= col &&
          col < (i + 1) * BLK_W) { // if the edge in the current TC_block frame
                                   // of column.
        unsigned row_local = edgeToRow[eIdx] % BLK_H;
        unsigned col_local = col % BLK_W;
        hspa[row_local * BLK_W + col_local + offset_spa] =
            1; // set the edge of the sparse_A.
        hind[col_local + offset_ind] =
            edgeList[eIdx]; // record the mapping from sparse_A colId to rowId
                            // of dense_X.
      }
    }
    __syncthreads();
  }
}

__global__ void
hmat_convert_packed_cuda_kernel(const int *__restrict__ Pointer1,
                                const float *__restrict__ hspa,
                                uint32_t *packed_hspa) {
  const unsigned bid = blockIdx.x;  // block_index == row_window_index
  const unsigned wid = threadIdx.y; // warp_index handling multi-dimension > 16.
  const unsigned laneid = threadIdx.x; // lanid of each warp.
  const unsigned tid =
      threadIdx.y * blockDim.x + laneid; // threadid of each block.
  const unsigned warpSize = blockDim.x;  // number of threads per warp.
  const unsigned threadPerBlock =
      blockDim.x * blockDim.y; // number of threads per block.
  const unsigned num_TC_blocks =
      Pointer1[bid + 1] -
      Pointer1[bid]; // number of TC_blocks of the current row_window.
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
          byte |= (1 << bit); 
        }
      }
      packed_hspa[idx + offset_spa_packed] = byte;
    }
  }
}

__global__ void
hmat_convert_uint32_swizzle_cuda_kernel(const int *__restrict__ Pointer1,
                                        const float *__restrict__ hspa,
                                        uint32_t *packed_hspa) {
  const unsigned bid = blockIdx.x;  // block_index == row_window_index
  const unsigned wid = threadIdx.y; // warp_index handling multi-dimension > 16.
  const unsigned laneid = threadIdx.x; // lanid of each warp.
  const unsigned tid =
      threadIdx.y * blockDim.x + laneid; // threadid of each block.
  const unsigned warpSize = blockDim.x;  // number of threads per warp.
  const unsigned threadPerBlock =
      blockDim.x * blockDim.y; // number of threads per block.
  const unsigned num_TC_blocks =
      Pointer1[bid + 1] -
      Pointer1[bid]; // number of TC_blocks of the current row_window.
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
        // int element_idx = idx * 32 + bit;
        // float val = hspa[element_idx + offset_spa];

        int group_id = bit >> 2;
        int threa_id_in_group = bit % 4;
        int row = group_id + 8 * (idx % 2);
        int col = threa_id_in_group + 4 * (idx / 2);
        float val = hspa[row * BLK_W + col + offset_spa];

        if (abs(val - 0.0f) > 1e-5) { // val != 0.0f
          byte |= (1 << bit);        
        }
      }
      packed_hspa[idx + offset_spa_packed] = byte;
    }
  }
}

void hmat_cuda(const int32_t *nodePointer, const int32_t *edgeList,
               const int32_t *blockPartition, const int32_t *edgeToColumn,
               const int32_t *edgeToRow, const int32_t *Pointer1,
               int32_t num_row_windows, int num_nodes, int num_edges,
               float *hspa, int *hind) {
  const int WARPperBlock = WPB;

  dim3 grid(num_row_windows, 1, 1);
  dim3 block(WARP_SIZE, WARPperBlock, 1);
  hmat_cuda_kernel<<<grid, block>>>(nodePointer, edgeList, blockPartition,
                                    edgeToColumn, edgeToRow, Pointer1,
                                    num_nodes, num_edges, hspa, hind);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error("CUDA error in hmat_cuda_kernel: " +
                             std::string(cudaGetErrorString(error)));
  }
}

void hmat_packed_cuda(int32_t num_row_windows, const int32_t *Pointer1,
                      const float *hspa, uint32_t *hspa_packed) {
  const int WARPperBlock = WPB;

  dim3 grid(num_row_windows, 1, 1);
  dim3 block(WARP_SIZE, WARPperBlock, 1);
  hmat_convert_packed_cuda_kernel<<<grid, block>>>(Pointer1, hspa, hspa_packed);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error("CUDA error in hmat_convert_packed_cuda_kernel: " +
                             std::string(cudaGetErrorString(error)));
  }
}

void hmat_packed_swizzle_cuda(int32_t num_row_windows, const int32_t *Pointer1,
                              const float *hspa, uint32_t *hspa_packed) {
  const int WARPperBlock = WPB;

  dim3 grid(num_row_windows, 1, 1);
  dim3 block(WARP_SIZE, WARPperBlock, 1);
  hmat_convert_uint32_swizzle_cuda_kernel<<<grid, block>>>(Pointer1, hspa,
                                                           hspa_packed);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(
        "CUDA error in hmat_convert_uint32_swizzle_cuda_kernel: " +
        std::string(cudaGetErrorString(error)));
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
      nb2col[array[cur]] = loc; // mapping from eid to TC_block column index.[]
    }
    cur++;
  }
  return nb2col;
}

void preprocess(const int32_t *edgeList, const int32_t *nodePointer,
                int num_nodes, int blockSize_h, int blockSize_w,
                int32_t *blockPartition, int32_t *edgeToColumn,
                int32_t *edgeToRow,
                int32_t *Pointer1) {

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

  const size_t num_row_windows = (num_nodes + BLK_H - 1) / BLK_H; 

  std::vector<int> blockPartition_vec(
      blockPartition, blockPartition + num_row_windows);
  // Step-3: Calculate Pointer1 as prefix sum of blockPartition using thrust.
  thrust::inclusive_scan(blockPartition_vec.begin(), blockPartition_vec.end(),
                         Pointer1 + 1);
  Pointer1[0] = 0;
}


} // namespace voltrix

#endif // VOLTRIX_BMAT_KERNELS_CUH_
