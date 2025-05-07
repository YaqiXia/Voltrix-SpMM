#ifndef VOLTRIX_SPMM_KERNELS_CUH_
#define VOLTRIX_SPMM_KERNELS_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <chrono>
#include <csetjmp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

// Primary header is compatible with pre-C++11, collective algorithm headers
// require C++11
// Optionally include for memcpy_async() collective
// Optionally include for reduce() collective
#include <cuda/barrier>

#include "voltrix/traits.h"

namespace voltrix {

inline __device__ float our_float_to_tf32(float in) {
  float ret;
  asm volatile("{\n  .reg .b32 __$1;  // TAG2"
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

void load_from_file(std::vector<float> &data, const std::string &filename) {
  std::ifstream in_file(filename, std::ios::binary);
  in_file.seekg(0, std::ios::end);
  size_t file_size = in_file.tellg();
  in_file.seekg(0, std::ios::beg);
  data.resize(file_size / sizeof(float));
  in_file.read(reinterpret_cast<char *>(data.data()), file_size);
  in_file.close();
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

DEVICE uint32_t cast_smem_ptr_to_uint(void const *const ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// struct Barrier {
//   uint64_t barrier_;
//   DEVICE Barrier() = delete;

//   DEVICE void init(uint32_t arrive_count) const {
//     uint64_t const *smem_ptr = &barrier_;
//     uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
//     asm volatile(
//         "{\n\t"
//         "mbarrier.init.shared.b64 [%1], %0; \n"
//         "}"
//         :
//         : "r"(arrive_count), "r"(smem_addr));
//   }

//   // local arrive
//   DEVICE void arrive() const {
//     uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
//     asm volatile(
//         "{\n\t"
//         "mbarrier.arrive.shared.b64 _, [%0];\n\t"
//         "}"
//         :
//         : "r"(smem_addr));
//   }

//   // remote arrive
//   DEVICE void arrive(uint32_t cta_id, uint32_t pred = true) const {
//     uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
//     asm volatile(
//         "{\n\t"
//         ".reg .pred p;\n\t"
//         ".reg .b32 remAddr32;\n\t"
//         "setp.eq.u32 p, %2, 1;\n\t"
//         "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
//         "@p mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
//         "}"
//         :
//         : "r"(smem_addr), "r"(cta_id), "r"(pred));
//   }

//   DEVICE void wait(uint32_t phase) {
//     uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
//     // Arbitrarily large timer value after which try-wait expires and
// re-tries. uint32_t ticks = 0x989680; asm volatile(
//         "{\n\t"
//         ".reg .pred       P1; \n\t"
//         "LAB_WAIT: \n\t"
//         "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1, %2; \n\t"
//         "@P1 bra.uni DONE; \n\t"
//         "bra.uni     LAB_WAIT; \n\t"
//         "DONE: \n\t"
//         "}"
//         :
//         : "r"(smem_addr), "r"(phase), "r"(ticks));
//   }

//   DEVICE void arrive_and_wait() {
//     arrive();
//     wait(0);
//   }

//   DEVICE uint32_t try_wait(uint32_t phase) {
//     uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
//     uint32_t waitComplete;

//     asm volatile(
//         "{\n\t"
//         ".reg .pred P1; \n\t"
//         "mbarrier.try_wait.parity.shared.b64 P1, [%1], %2; \n\t"
//         "selp.b32 %0, 1, 0, P1; \n\t"
//         "}"
//         : "=r"(waitComplete)
//         : "r"(smem_addr), "r"(phase));

//     return waitComplete;
//   }

//   DEVICE void invalidate() {
//     uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
//     asm volatile(
//         "{\n\t"
//         "mbarrier.ival.shared.b64 [%0]; \n\t"
//         "}"
//         :
//         : "r"(smem_addr));
//   }

//   // These are TMA related barrier methods.
//   // CULTASS implements it in another barrier.
//   // We put them together.
//   DEVICE void arrive_and_expect_tx(uint32_t transaction_bytes) {
//     uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
//     asm volatile(
//         "{\n\t"
//         "mbarrier.arrive.expect_tx.shared.b64 _, [%1], %0; \n\t"
//         "}"
//         :
//         : "r"(transaction_bytes), "r"(smem_addr));
//   }

//   DEVICE void arrive_and_expect_tx(uint32_t transaction_bytes, uint32_t
//   cta_id,
//                                    uint32_t pred) {
//     uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
//     asm volatile(
//         "{\n\t"
//         ".reg .pred p;\n\t"
//         ".reg .b32 remAddr32;\n\t"
//         "setp.eq.u32 p, %2, 1;\n\t"
//         "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
//         "@p mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], "
//         "%3;\n\t"
//         "}"
//         :
//         : "r"(smem_addr), "r"(cta_id), "r"(pred), "r"(transaction_bytes));
//   }

//   DEVICE void expect_transaction(uint32_t transaction_bytes) const {
//     uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
//     asm volatile(
//         "{\n\t"
//         "mbarrier.expect_tx.shared.b64 [%1], %0; \n\t"
//         "}"
//         :
//         : "r"(transaction_bytes), "r"(smem_addr));
//   }
// };

using barrier = cuda::barrier<cuda::thread_scope_block>;

struct alignas(16) Uint4 {
  uint32_t vs[4];

  __device__ __forceinline__ uint32_t &operator[](int i) { return vs[i]; }
};

#define PRINT_BT(BX, TX, TY, ...)                                              \
  {                                                                            \
    if (blockIdx.x == BX && threadIdx.x == TX && threadIdx.y == TY) {          \
      printf("[DEBUG] " __VA_ARGS__);                                          \
    }                                                                          \
  }

template <typename T> constexpr DEVICE __host__ T div_round_up(T a, T b) {
  return (a + b - 1) / b;
}

template <int32_t BLOCK_H, int32_t BLOCK_W, int32_t N, int32_t D,
          int32_t MAX_DIM>
struct Scheduler {
  static_assert(N % BLOCK_H == 0 && D % BLOCK_W == 0,
                "N should be divisible by BLOCK_H and D should be divisible by "
                "BLOCK_W");

  static constexpr int32_t NUM_BLOCKS_PER_ROW = N / BLOCK_H;
  static constexpr int32_t NUM_BLOCKS_PER_COL = D / BLOCK_W;
  static constexpr int32_t MAX_BLOCKS_PER_STAGE = MAX_DIM / BLOCK_H;
  static_assert(MAX_BLOCKS_PER_STAGE == 44,
                "For current implementation, "
                "MAX_BLOCKS_PER_STAGE should be 44");

  static constexpr int32_t NUM_STAGES_FULL_LINE =
      div_round_up(NUM_BLOCKS_PER_ROW, MAX_BLOCKS_PER_STAGE);

  Scheduler() = default;

  Scheduler(int32_t start, int32_t end) { init(start, end); }

  DEVICE void init(int32_t start, int32_t end) {
    auto num_cross_rows_ =
        end / NUM_BLOCKS_PER_ROW - (start - 1) / NUM_BLOCKS_PER_ROW + 1;
    num_stages_ = div_round_up(end - start, MAX_BLOCKS_PER_STAGE);
  }

  int32_t num_stages_; // stage performs max loading for computing a block row.
};

template <int32_t BLOCK_H, int32_t BLOCK_W, int32_t N, int32_t D,
          int32_t MAX_DIM>
struct ProducerScheduer : public Scheduler<BLOCK_H, BLOCK_W, N, D, MAX_DIM> {
  using Base = Scheduler<BLOCK_H, BLOCK_W, N, D, MAX_DIM>;

  DEVICE void plan(int32_t stage) { stage *Base::NUM_BLOCKS_PER_ROW; }

  int32_t num_current_sub_stages_;
};

template <int32_t BLOCK_H, int32_t BLOCK_W, int32_t N, int32_t D,
          int32_t MAX_DIM>
struct ConsumerScheduler : public Scheduler<BLOCK_H, BLOCK_W, N, D, MAX_DIM> {
  using Base = Scheduler<BLOCK_H, BLOCK_W, N, D, MAX_DIM>;

  DEVICE void plan(int32_t stage) {}
};

template <int32_t BLOCK_H, int32_t BLOCK_W, int32_t MAX_DIM>
struct DynamicScheduler {
  static constexpr int32_t MAX_BLOCKS_PER_STAGE = MAX_DIM / BLOCK_H;
  // static_assert(MAX_BLOCKS_PER_STAGE == 44,
  //               "For current implementation, "
  //               "MAX_BLOCKS_PER_STAGE should be 44");

  DEVICE DynamicScheduler(int32_t start, int32_t end, int32_t embedding_dim)
      : start_ofs_(start), end_ofs_(end),
        num_blocks_per_row_(embedding_dim / BLOCK_H) {}

  DEVICE void init() {}

  DEVICE void step() {
    start_col_blk = start_ofs_ % num_blocks_per_row_;
    curr_row = start_ofs_ / num_blocks_per_row_;
    if (start_col_blk + MAX_BLOCKS_PER_STAGE < num_blocks_per_row_ &&
        start_ofs_ + MAX_BLOCKS_PER_STAGE < end_ofs_) {
      end_col_blk = start_col_blk + MAX_BLOCKS_PER_STAGE;
      start_ofs_ += MAX_BLOCKS_PER_STAGE;
    } else if (num_blocks_per_row_ - start_col_blk <= end_ofs_ - start_ofs_) {
      end_col_blk = num_blocks_per_row_;
      start_ofs_ += num_blocks_per_row_ - start_col_blk;
    } else {
      end_col_blk = end_ofs_ % num_blocks_per_row_;
      start_ofs_ = end_ofs_;
    }
  }

  int32_t curr_row; // number of TC_blocks of the current row_window.

  int32_t start_col_blk;
  int32_t end_col_blk;

  const int32_t num_blocks_per_row_; // const
  int32_t start_ofs_;                // const
  const int32_t end_ofs_;            // const
};

template <int32_t BLOCK_H, int32_t BLOCK_W, int32_t MAX_DIM>
struct DynamicSchedulerV2 {
  static constexpr int32_t MAX_BLOCKS_PER_STAGE = MAX_DIM / BLOCK_H;
  // static_assert(MAX_BLOCKS_PER_STAGE == 44,
  //               "For current implementation, "
  //               "MAX_BLOCKS_PER_STAGE should be 44");

  DEVICE DynamicSchedulerV2(int32_t start, int32_t end, int32_t embedding_dim)
      : start_ofs_(start), end_ofs_(end),
        num_blocks_per_row_(embedding_dim / BLOCK_H) {}

  DEVICE void step() {
    start_col_blk = start_ofs_ % num_blocks_per_row_;
    curr_row = start_ofs_ / num_blocks_per_row_;

    int blocks_remaining_in_row = num_blocks_per_row_ - start_col_blk;
    int blocks_remaining_in_total = end_ofs_ - start_ofs_;

    step_blocks = min(MAX_BLOCKS_PER_STAGE,
                      min(blocks_remaining_in_row, blocks_remaining_in_total));

    start_ofs_ += step_blocks;
  }

  int32_t curr_row; // number of TC_blocks of the current row_window.

  int32_t start_col_blk;
  int32_t step_blocks;

  const int32_t num_blocks_per_row_; // const
  int32_t start_ofs_;                // const
  const int32_t end_ofs_;            // const
};

template <int32_t BLOCK_H, int32_t BLOCK_W, int32_t MAX_DIM>
struct DynamicWaveScheduler {
  static constexpr int32_t MAX_BLOCKS_PER_STAGE = MAX_DIM / BLOCK_H;

  DEVICE DynamicWaveScheduler(int32_t embedding_dim, int32_t num_nodes)
      : start_col_blk(0), curr_row(blockIdx.x),
        num_blocks_per_row_(embedding_dim / BLOCK_H),
        num_blocks_per_col_(num_nodes / BLOCK_H) {
    step_blocks = min(MAX_BLOCKS_PER_STAGE, num_blocks_per_row_);
    // step_blocks = MAX_BLOCKS_PER_STAGE < num_blocks_per_row_
    //                   ? MAX_BLOCKS_PER_STAGE
    //                   : num_blocks_per_row_;
  }

  DEVICE void step() {
    if (start_col_blk + step_blocks >= num_blocks_per_row_) { // next wave
      curr_row += gridDim.x;
      start_col_blk = 0;

      step_blocks = min(MAX_BLOCKS_PER_STAGE, num_blocks_per_row_);
      // step_blocks = MAX_BLOCKS_PER_STAGE < num_blocks_per_row_
      //                   ? MAX_BLOCKS_PER_STAGE
      //                   : num_blocks_per_row_;
    } else {
      start_col_blk += step_blocks;
      int blocks_remaining_in_row = num_blocks_per_row_ - start_col_blk;
      step_blocks = min(MAX_BLOCKS_PER_STAGE, blocks_remaining_in_row);
      // step_blocks = MAX_BLOCKS_PER_STAGE < blocks_remaining_in_row
      //                   ? MAX_BLOCKS_PER_STAGE
      //                   : blocks_remaining_in_row;
    }
  }

  DEVICE bool isvalid() { return curr_row < num_blocks_per_col_; }

  int32_t curr_row; // number of TC_blocks of the current row_window.

  int32_t start_col_blk;
  int32_t step_blocks;

  const int32_t num_blocks_per_row_; // const
  const int32_t num_blocks_per_col_; // const
};

template <int32_t M, int32_t N, int32_t K> struct MMA {};

template <> struct MMA<16, 8, 8> {
  template <typename AT, typename BT, typename CT>
  static void DEVICE mma(AT *a_frag,  // 4 tf32
                         BT *b_frag,  // 2 tf32
                         CT *acc_frag // 4 f32
  ) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+f"(acc_frag[0]), "+f"(acc_frag[1]), "+f"(acc_frag[2]),
                   "+f"(acc_frag[3])
                 : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]),
                   "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
  }
};

template <int32_t M, int32_t N, int32_t K, uint32_t SELECTOR> struct SparseMMA {
  static_assert(SELECTOR >= 0 && SELECTOR <= 3, "Invalid selector");
};

template <> struct SparseMMA<16, 8, 8, 0x0> {
  template <typename AT, typename BT, typename CT>
  static void DEVICE mma(AT *a_frag, BT *b_frag, CT *acc_frag,
                         uint32_t &a_meta) {
    asm volatile(
        "{\n\t"
        "mma.sp::ordered_metadata.sync.aligned.m16n8k8.row.col.f32.tf32.tf32."
        "f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6, %7}, "
        "{%0, %1, %2, %3}, "
        "%8, 0x0;\n"
        "\t}"
        : "+f"(acc_frag[0]), "+f"(acc_frag[1]), "+f"(acc_frag[2]),
          "+f"(acc_frag[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(b_frag[0]), "r"(b_frag[1]),
          "r"(a_meta));
  }
};

template <typename T>
__device__ void empty_store_matrix_sync(float *output, T &acc_frag) {
  asm volatile("mov.u64 %0, %0;\n"
               "mov.f32 %1, %1;\n"
               "mov.f32 %2, %2;\n"
               "mov.f32 %3, %3;\n"
               "mov.f32 %4, %4;\n"
               : "+l"(output), "+f"(acc_frag.x[0]), "+f"(acc_frag.x[1]),
                 "+f"(acc_frag.x[2]), "+f"(acc_frag.x[3]));
}

template <int32_t NUM_AGENTS, int32_t NUM_BUFFERS, int32_t MAX_MMAS_PER_WARP,
          int32_t CONSUMER_WARPS_PER_BLOCK, int32_t PADDING_SIZE = 0>
__global__ void
spmm_mma16168(const int *__restrict__ blks_offsets,
              const uint32_t *__restrict__ hspa_packed,
              const int *__restrict__ hind, const int num_nodes,
              const int num_edges,
              const int embedding_dim,         // embedding dimension.
              const float *__restrict__ input, // input feature matrix.
              float *output // aggreAGNNed output feature matrix.
) {
  using Traits =
      PersistenKernelTraits<float, NUM_AGENTS, NUM_BUFFERS, MAX_MMAS_PER_WARP,
                            CONSUMER_WARPS_PER_BLOCK, PADDING_SIZE>;

  constexpr int32_t NUM_BARRIERS = Traits::NUM_BARRIERS;

  constexpr int32_t THREADS_PER_WARP = Traits::THREADS_PER_WARP;
  constexpr int32_t THREADS_PER_BLOCK = Traits::THREADS_PER_BLOCK;

  constexpr int32_t MAX_FEATURE_DIM_PER_BLOCK =
      Traits::MAX_FEATURE_DIM_PER_BLOCK;
  constexpr int32_t MAX_FEATURE_DIM_PER_BLOCK_PADDED =
      Traits::MAX_FEATURE_DIM_PER_BLOCK_PADDED;
  constexpr int32_t NUM_DENSE_X_SHARED_PER_BUFFER_PADDED =
      Traits::NUM_DENSE_X_SHARED_PER_BUFFER_PADDED;
  constexpr int32_t DENSE_X_STRIDE_PER_MMA = Traits::DENSE_X_STRIDE_PER_MMA;

  const int32_t wid = threadIdx.y; // warp_index handling multi-dimension > 16.
  const int32_t laneid = threadIdx.x; // lanid of each warp.  [0, 31]
  const int32_t tid =
      threadIdx.y * THREADS_PER_WARP + laneid; // threadid of each block.

  alignas(
      16) extern __shared__ float dense_X[]; // row-major dense X shared memory.
  alignas(16) __shared__ float
      sparse_A[NUM_BUFFERS * BLK_H * BLK_W]; // row-major sparse matrix

  DynamicWaveScheduler<BLK_H, BLK_W, MAX_FEATURE_DIM_PER_BLOCK> scheduler(
      embedding_dim, num_nodes);

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bars[NUM_BARRIERS]; // double buffers for load/mma
  if (tid == 0) {
#pragma unroll
    for (int32_t i = 0; i < NUM_BARRIERS; i++) {
      init(&bars[i], THREADS_PER_BLOCK);
    }

    cuda::device::experimental::fence_proxy_async_shared_cta();
  }

  __syncthreads();

  int32_t current_buffer = NUM_BUFFERS - 1;

  const bool is_producer_warp = (wid >= CONSUMER_WARPS_PER_BLOCK);
  if (is_producer_warp) {
    for (; scheduler.isvalid(); scheduler.step()) {
      int32_t num_TC_blocks = blks_offsets[scheduler.curr_row + 1] -
                              blks_offsets[scheduler.curr_row];

      auto tma_load_size = cuda::aligned_size_t<16>(scheduler.step_blocks *
                                                    BLK_H * sizeof(float));

      auto spa_packed_start =
          blks_offsets[scheduler.curr_row] * BLK_H * BLK_W / 32;
      auto ind_start = blks_offsets[scheduler.curr_row] * BLK_W;

#pragma unroll 4
      for (unsigned i = 0; i < num_TC_blocks; ++i) {
        // int offset_spa_packed = i * BLK_H * BLK_W / 32 + spa_packed_start;
        current_buffer = (current_buffer + 1) % NUM_BUFFERS;

        auto &ld_bar = bars[current_buffer];
        auto &mma_bar = bars[NUM_BUFFERS + current_buffer];

        auto sparse_A_ptr = &sparse_A[current_buffer * BLK_W * BLK_H];
        auto dense_X_ptr =
            &dense_X[current_buffer * NUM_DENSE_X_SHARED_PER_BUFFER_PADDED];

        {
          mma_bar.arrive_and_wait();
          { // TESTING
            // auto phase = ld_bar.arrive();
            // continue;
          }

          if (laneid < warpSize) {
            int32_t offset_spa_packed =
                i * BLK_H * BLK_W / 32 + spa_packed_start;
            Uint4 packed = *reinterpret_cast<const Uint4 *>(
                &hspa_packed[offset_spa_packed]);

#pragma unroll
            for (int j = 0; j < 4; ++j) {
              sparse_A_ptr[laneid * 4 + j] =
                  (packed[laneid / 8] & (1u << ((laneid % 8) * 4 + j))) ? 1.0f
                                                                        : 0.0f;
            }
          }

          int32_t offset_ind = i * BLK_W + ind_start;
          auto src_ofs = hind[laneid + offset_ind];
          if (laneid < BLK_W) {
            cuda::memcpy_async(dense_X_ptr +
                                   laneid * MAX_FEATURE_DIM_PER_BLOCK_PADDED,
                               input + src_ofs * embedding_dim +
                                   scheduler.start_col_blk * BLK_H,
                               tma_load_size, ld_bar);
          }

          auto phase = ld_bar.arrive();
        }
      }
    }
  } else {
#pragma unroll
    for (int32_t b = 0; b < NUM_BUFFERS; b++) {
      auto phase = bars[NUM_BUFFERS + b].arrive();
    }

    uint32_t a_frag[4];
    uint32_t b_frag[MAX_MMAS_PER_WARP][2][2];
    float acc_frag[MAX_MMAS_PER_WARP][2][4];

    static_assert(sizeof(float[2][4]) == 32);

    for (; scheduler.isvalid(); scheduler.step()) {
      int32_t num_TC_blocks = blks_offsets[scheduler.curr_row + 1] -
                              blks_offsets[scheduler.curr_row];

      const int32_t num_blocks = scheduler.step_blocks;
      const int32_t num_mmas = num_blocks / CONSUMER_WARPS_PER_BLOCK +
                               (wid < num_blocks % CONSUMER_WARPS_PER_BLOCK);

#pragma unroll
      for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++) {
#pragma unroll
        for (int32_t i = 0; i < 2; ++i) {
#pragma unroll
          for (int32_t v = 0; v < 4; ++v) {
            acc_frag[f][i][v] = 0.0f;
          }
        }
      }

      const auto group_id = laneid >> 2;
      const auto lane_id_in_group = laneid % 4;

#pragma unroll 4
      for (int32_t i = 0; i < num_TC_blocks; i++) {
        current_buffer = (current_buffer + 1) % NUM_BUFFERS;

        auto &ld_bar = bars[current_buffer];
        auto &mma_bar = bars[NUM_BUFFERS + current_buffer];

        uint32_t sparse_A_ptr =
            cast_smem_ptr_to_uint(&sparse_A[current_buffer * BLK_W * BLK_H]);
        auto dense_X_ptr =
            &dense_X[current_buffer * NUM_DENSE_X_SHARED_PER_BUFFER_PADDED];

        {
          ld_bar.arrive_and_wait();

          // uint32_t a_ptr = sparse_A_ptr +
          //                  ((((laneid % 16) << 3) + ((laneid >> 4) << 2)) <<
          //                  2);
          uint32_t a_ptr =
              sparse_A_ptr + ((((laneid % 16) * 8) + ((laneid / 16) * 4)) * 4);

          asm volatile("{\n"
                       "ldmatrix.sync.aligned.x4.m8n8.shared.b16"
                       "{%0, %1, %2, %3}, [%4];\n}"
                       : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]),
                         "=r"(a_frag[3])
                       : "r"(a_ptr));

#pragma unroll
          for (unsigned t = 0; t < 4; t++) {
            asm volatile("cvt.rna.tf32.f32 %0, %0;\n" : "+r"(a_frag[t]));
          }

          auto dense_base = dense_X_ptr + wid * BLK_H;
#pragma unroll
          for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++) { /// move it out
            if (f < num_mmas) {
              b_frag[f][0][0] = *reinterpret_cast<uint32_t *>(
                  dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id +
                  lane_id_in_group * MAX_FEATURE_DIM_PER_BLOCK_PADDED);
              b_frag[f][0][1] = *reinterpret_cast<uint32_t *>(
                  dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id +
                  (lane_id_in_group + 4) * MAX_FEATURE_DIM_PER_BLOCK_PADDED);
              b_frag[f][1][0] = *reinterpret_cast<uint32_t *>(
                  dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id + 8 +
                  lane_id_in_group * MAX_FEATURE_DIM_PER_BLOCK_PADDED);
              b_frag[f][1][1] = *reinterpret_cast<uint32_t *>(
                  dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id + 8 +
                  (lane_id_in_group + 4) * MAX_FEATURE_DIM_PER_BLOCK_PADDED);

#pragma unroll
              for (int32_t t = 0; t < 2; t++) {
#pragma unroll
                for (int32_t v = 0; v < 2; v++) {
                  asm volatile("cvt.rna.tf32.f32 %0, %0;"
                               : "+r"(b_frag[f][t][v]));
                }
              }

              // Perform the matrix multiplication.
              MMA<16, 8, 8>::mma(a_frag, b_frag[f][0], acc_frag[f][0]);
              MMA<16, 8, 8>::mma(a_frag, b_frag[f][1], acc_frag[f][1]);
            }
          }

          auto phase = mma_bar.arrive();
        }
      }

      auto offset_base = output + scheduler.curr_row * BLK_H * embedding_dim +
                         (scheduler.start_col_blk + wid) * BLK_H;

#pragma unroll
      for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++) {
        if (f < num_mmas) {
#pragma unroll
          for (int32_t i = 0; i < 2; i++) {
#pragma unroll
            for (int32_t v = 0; v < 4; v += 2) {
              auto row = group_id + (v / 2) * 8;
              // auto col = (lane_id_in_group * 2) + (v & 0x1) + i * 8;
              auto col = (lane_id_in_group * 2) + i * 8;

              auto ptr = offset_base + f * DENSE_X_STRIDE_PER_MMA +
                         row * embedding_dim + col;

              // *(offset_base + f * DENSE_X_STRIDE_PER_MMA + row *
              // embedding_dim +
              //   col) = acc_frag[f][i][v];

              // asm volatile(
              //     "st.global.wt.v2.f32 [%0], {%1, %2};\n" ::"l"(ptr),
              //     "f"(acc_frag[f][i][v]), "f"(acc_frag[f][i][v + 1]));

              double val = *(double *)&acc_frag[f][i][v]; // ld.local.f64
              asm volatile("st.global.wt.f64 [%0], %1;\n" ::"l"(ptr), "d"(val));
            }
          }
        }
      }
    }
  }

  if (tid == 0) {
    (&bars[0])->~barrier();
    (&bars[1])->~barrier();
    (&bars[2])->~barrier();
    (&bars[3])->~barrier();
  }
}

template <int32_t NUM_AGENTS, int32_t NUM_BUFFERS, int32_t MAX_MMAS_PER_WARP,
          int32_t CONSUMER_WARPS_PER_BLOCK, int32_t PADDING_SIZE = 0>
__global__ void
spmm_sp_mma16168(const int *__restrict__ blks_offsets,
                 const uint32_t *__restrict__ hspa_packed,
                 const int *__restrict__ hind, const int num_nodes,
                 const int num_edges,
                 const int embedding_dim,         // embedding dimension.
                 const float *__restrict__ input, // input feature matrix.
                 float *output // aggreAGNNed output feature matrix.
) {
  using Traits =
      PersistenKernelTraits<float, NUM_AGENTS, NUM_BUFFERS, MAX_MMAS_PER_WARP,
                            CONSUMER_WARPS_PER_BLOCK, PADDING_SIZE>;

  constexpr int32_t NUM_BARRIERS = Traits::NUM_BARRIERS;

  constexpr int32_t THREADS_PER_WARP = Traits::THREADS_PER_WARP;
  constexpr int32_t THREADS_PER_BLOCK = Traits::THREADS_PER_BLOCK;

  constexpr int32_t MAX_FEATURE_DIM_PER_BLOCK =
      Traits::MAX_FEATURE_DIM_PER_BLOCK;
  constexpr int32_t MAX_FEATURE_DIM_PER_BLOCK_PADDED =
      Traits::MAX_FEATURE_DIM_PER_BLOCK_PADDED;
  constexpr int32_t NUM_DENSE_X_SHARED_PER_BUFFER_PADDED =
      Traits::NUM_DENSE_X_SHARED_PER_BUFFER_PADDED;
  constexpr int32_t DENSE_X_STRIDE_PER_MMA = Traits::DENSE_X_STRIDE_PER_MMA;

  const int32_t wid = threadIdx.y; // warp_index handling multi-dimension > 16.
  const int32_t laneid = threadIdx.x; // lanid of each warp.  [0, 31]
  const int32_t tid =
      threadIdx.y * THREADS_PER_WARP + laneid; // threadid of each block.

  alignas(
      16) extern __shared__ float dense_X[]; // row-major dense X shared memory.
  alignas(16) __shared__ float
      sparse_A[NUM_BUFFERS * BLK_H * BLK_W]; // row-major sparse matrix

  const bool is_producer_warp = (wid >= CONSUMER_WARPS_PER_BLOCK);

  DynamicWaveScheduler<BLK_H, BLK_W, MAX_FEATURE_DIM_PER_BLOCK> scheduler(
      embedding_dim, num_nodes);

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bars[NUM_BARRIERS]; // double buffers for load/mma
  if (tid == 0) {
#pragma unroll
    for (int32_t i = 0; i < NUM_BARRIERS; i++) {
      init(&bars[i], THREADS_PER_BLOCK);
    }

    cuda::device::experimental::fence_proxy_async_shared_cta();
  }

  __syncthreads();

  int32_t current_buffer = NUM_BUFFERS - 1;

  if (is_producer_warp) {
    for (; scheduler.isvalid(); scheduler.step()) {
      int32_t num_TC_blocks = blks_offsets[scheduler.curr_row + 1] -
                              blks_offsets[scheduler.curr_row];

      auto tma_load_size = cuda::aligned_size_t<16>(scheduler.step_blocks *
                                                    BLK_H * sizeof(float));

      auto spa_packed_start =
          blks_offsets[scheduler.curr_row] * BLK_H * BLK_W / 32;
      auto ind_start = blks_offsets[scheduler.curr_row] * BLK_W;

#pragma unroll 4
      for (unsigned i = 0; i < num_TC_blocks; ++i) {
        // int offset_spa_packed = i * BLK_H * BLK_W / 32 + spa_packed_start;
        current_buffer = (current_buffer + 1) % NUM_BUFFERS;

        auto &ld_bar = bars[current_buffer];
        auto &mma_bar = bars[NUM_BUFFERS + current_buffer];

        auto sparse_A_ptr = &sparse_A[current_buffer * BLK_W * BLK_H];
        auto dense_X_ptr =
            &dense_X[current_buffer * NUM_DENSE_X_SHARED_PER_BUFFER_PADDED];

        {
          mma_bar.arrive_and_wait();
          { // TESTING
            // auto phase = ld_bar.arrive();
            // continue;
          }

          if (laneid < warpSize) {
            // int byte_idx = idx / 32;
            // // int bit_idx = laneid;
            // uint32_t byte = hspa_packed[byte_idx + offset_spa_packed];
            // sparse_A_ptr[idx] = (byte & (1u << laneid)) ? 1.0f : 0.0f;
            int32_t offset_spa_packed =
                i * BLK_H * BLK_W / 32 + spa_packed_start;
            Uint4 packed = *reinterpret_cast<const Uint4 *>(
                &hspa_packed[offset_spa_packed]);

#pragma unroll
            for (int j = 0; j < 4; ++j) {
              sparse_A_ptr[laneid * 4 + j] =
                  (packed[laneid / 8] & (1u << ((laneid % 8) * 4 + j))) ? 1.0f
                                                                        : 0.0f;
            }
          }

          int32_t offset_ind = i * BLK_W + ind_start;
          auto src_ofs = hind[laneid + offset_ind];
          if (laneid < BLK_W) {
            cuda::memcpy_async(dense_X_ptr +
                                   laneid * MAX_FEATURE_DIM_PER_BLOCK_PADDED,
                               input + src_ofs * embedding_dim +
                                   scheduler.start_col_blk * BLK_H,
                               tma_load_size, ld_bar);
          }

          auto phase = ld_bar.arrive();
        }
      }
    }
  } else {
#pragma unroll
    for (int32_t b = 0; b < NUM_BUFFERS; b++) {
      auto phase = bars[NUM_BUFFERS + b].arrive();
    }

    uint32_t a_frag[2];
    uint32_t a_meta = 0b11101110111011101110111011101110;
    uint32_t b_frag[MAX_MMAS_PER_WARP][2][2];
    float acc_frag[MAX_MMAS_PER_WARP][2][4];

    static_assert(sizeof(float[2][4]) == 32);

    for (; scheduler.isvalid(); scheduler.step()) {
      int32_t num_TC_blocks = blks_offsets[scheduler.curr_row + 1] -
                              blks_offsets[scheduler.curr_row];

      const int32_t num_blocks = scheduler.step_blocks;
      const int32_t num_mmas = num_blocks / CONSUMER_WARPS_PER_BLOCK +
                               (wid < num_blocks % CONSUMER_WARPS_PER_BLOCK);

#pragma unroll
      for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++) {
#pragma unroll
        for (int32_t i = 0; i < 2; ++i) {
#pragma unroll
          for (int32_t v = 0; v < 4; ++v) {
            acc_frag[f][i][v] = 0.0f;
          }
        }
      }

      const auto group_id = laneid >> 2;
      const auto lane_id_in_group = laneid % 4;

#pragma unroll 4
      for (int32_t i = 0; i < num_TC_blocks; i++) {
        current_buffer = (current_buffer + 1) % NUM_BUFFERS;

        auto &ld_bar = bars[current_buffer];
        auto &mma_bar = bars[NUM_BUFFERS + current_buffer];

        uint32_t sparse_A_ptr =
            cast_smem_ptr_to_uint(&sparse_A[current_buffer * BLK_W * BLK_H]);
        auto dense_X_ptr =
            &dense_X[current_buffer * NUM_DENSE_X_SHARED_PER_BUFFER_PADDED];

        {
          ld_bar.arrive_and_wait();

          // uint32_t a_ptr = sparse_A_ptr +
          //                  ((((laneid % 16) << 3) + ((laneid >> 4) << 2)) <<
          //                  2);
          uint32_t a_ptr =
              sparse_A_ptr + ((((laneid % 16) * 8) + ((laneid / 16) * 4)) * 4);

          asm volatile("{\n"
                       "ldmatrix.sync.aligned.x2.m8n8.shared.b16"
                       "{%0, %1}, [%2];\n}"
                       : "=r"(a_frag[0]), "=r"(a_frag[1])
                       : "r"(a_ptr));

#pragma unroll
          for (unsigned t = 0; t < 2; t++) {
            asm volatile("cvt.rna.tf32.f32 %0, %0;\n" : "+r"(a_frag[t]));
          }

          auto dense_base = dense_X_ptr + wid * BLK_H;
#pragma unroll
          for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++) { /// move it out
            if (f < num_mmas) {
              b_frag[f][0][0] = *reinterpret_cast<uint32_t *>(
                  dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id +
                  lane_id_in_group * MAX_FEATURE_DIM_PER_BLOCK_PADDED);
              b_frag[f][0][1] = *reinterpret_cast<uint32_t *>(
                  dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id +
                  (lane_id_in_group + 4) * MAX_FEATURE_DIM_PER_BLOCK_PADDED);
              b_frag[f][1][0] = *reinterpret_cast<uint32_t *>(
                  dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id + 8 +
                  lane_id_in_group * MAX_FEATURE_DIM_PER_BLOCK_PADDED);
              b_frag[f][1][1] = *reinterpret_cast<uint32_t *>(
                  dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id + 8 +
                  (lane_id_in_group + 4) * MAX_FEATURE_DIM_PER_BLOCK_PADDED);

#pragma unroll
              for (int32_t t = 0; t < 2; t++) {
#pragma unroll
                for (int32_t v = 0; v < 2; v++) {
                  asm volatile("cvt.rna.tf32.f32 %0, %0;"
                               : "+r"(b_frag[f][t][v]));
                }
              }

              // Perform the matrix multiplication.
              constexpr uint32_t SELECTOR = 0x0;
              SparseMMA<16, 8, 8, SELECTOR>::mma(a_frag, b_frag[f][0],
                                                 acc_frag[f][0], a_meta);
              SparseMMA<16, 8, 8, SELECTOR>::mma(a_frag, b_frag[f][1],
                                                 acc_frag[f][1], a_meta);
            }
          }

          auto phase = mma_bar.arrive();
        }
      }

      auto offset_base = output + scheduler.curr_row * BLK_H * embedding_dim +
                         (scheduler.start_col_blk + wid) * BLK_H;

#pragma unroll
      for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++) {
        if (f < num_mmas) {
#pragma unroll
          for (int32_t i = 0; i < 2; i++) {
#pragma unroll
            for (int32_t v = 0; v < 4; v += 2) {
              auto row = group_id + (v / 2) * 8;
              // auto col = (lane_id_in_group * 2) + (v & 0x1) + i * 8;
              auto col = (lane_id_in_group * 2) + i * 8;

              auto ptr = offset_base + f * DENSE_X_STRIDE_PER_MMA +
                         row * embedding_dim + col;

              // *(offset_base + f * DENSE_X_STRIDE_PER_MMA + row *
              // embedding_dim +
              //   col) = acc_frag[f][i][v];

              // asm volatile(
              //     "st.global.wt.v2.f32 [%0], {%1, %2};\n" ::"l"(ptr),
              //     "f"(acc_frag[f][i][v]), "f"(acc_frag[f][i][v + 1]));

              double val = *(double *)&acc_frag[f][i][v]; // ld.local.f64
              asm volatile("st.global.wt.f64 [%0], %1;\n" ::"l"(ptr), "d"(val));
            }
          }
        }
      }
    }
  }

  if (tid == 0) {
    (&bars[0])->~barrier();
    (&bars[1])->~barrier();
    (&bars[2])->~barrier();
    (&bars[3])->~barrier();
  }
}

DEVICE void memcpy_async(float *dst, const float *src, int32_t size,
                         uint64_t &barrier) {
  uint32_t dst32 = cast_smem_ptr_to_uint(dst);
  uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier);

  asm volatile(
      "{\n\t"
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
      "[%0], [%1], %2, [%3];\n\t"
      "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%3], %2;\n\t"
      "}" ::"r"(dst32),
      "l"(src), "r"(size), "r"(smem_addr));
}

DEVICE void mbarrier_init(uint64_t &barrier, int32_t count) {
  auto smem_addr = cast_smem_ptr_to_uint(&barrier);
  asm volatile("{\n\t"
               "mbarrier.init.shared::cta.b64 [%0], %1;\n"
               "\t}" ::"r"(smem_addr),
               "r"(count));
}

DEVICE void mbarrier_arrive(uint64_t &barrier) {
  uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier);
  asm volatile("{\n\t"
               "mbarrier.arrive.shared::cta.b64 _, [%0];\n"
               "\t}"
               :
               : "r"(smem_addr));
}

DEVICE void mbarrier_arrive_and_wait(uint64_t &barrier) {
  uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier);
  asm volatile("{\n\t"
               ".reg .b64 phase;\n\t"
               ".reg .pred p;\n\t"
               "mbarrier.arrive.shared::cta.b64 phase, [%0];\n\t"
               // "WAIT_LOOP_LABEL:\n\t"
               // "mbarrier.test_wait.shared.b64 p, [%0], phase;\n\t"
               // "@!p nanosleep.u32 20;\n\t"
               // "@!p bra WAIT_LOOP_LABEL;\n\t"

               "LAB_WAIT: \n\t"
               "mbarrier.try_wait.shared.b64 p, [%0], phase; \n\t"
               "@p bra.uni DONE; \n\t"
               "bra.uni     LAB_WAIT; \n\t"
               "DONE: \n\t"
               "}"
               :
               : "r"(smem_addr));
}

DEVICE void invalidate(uint64_t &barrier) {
  uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier);
  asm volatile("{\n\t"
               "mbarrier.inval.shared.b64 [%0]; \n\t"
               "}"
               :
               : "r"(smem_addr));
}

DEVICE void cp_async_mbarrier_arrive(uint64_t &barrier) {
  uint32_t smem_ptr = cast_smem_ptr_to_uint(&barrier);
  asm volatile("{\n\t"
               "cp.async.mbarrier.arrive.shared.b64 [%0];\n"
               "\t}" ::"r"(smem_ptr));
}

template <int32_t NUM_AGENTS, int32_t NUM_BUFFERS, int32_t MAX_MMAS_PER_WARP,
          int32_t CONSUMER_WARPS_PER_BLOCK, int32_t PADDING_SIZE = 0>
__global__ void spmm_mma16168_spa_swizzle(
    const int *__restrict__ blks_offsets,
    const uint32_t *__restrict__ hspa_packed, const int *__restrict__ hind,
    const int num_nodes, const int num_edges,
    const int embedding_dim,         // embedding dimension.
    const float *__restrict__ input, // input feature matrix.
    float *output                    // aggreAGNNed output feature matrix.
) {
  using Traits =
      PersistenKernelTraits<float, NUM_AGENTS, NUM_BUFFERS, MAX_MMAS_PER_WARP,
                            CONSUMER_WARPS_PER_BLOCK, PADDING_SIZE>;

  constexpr int32_t NUM_BARRIERS = Traits::NUM_BARRIERS;

  constexpr int32_t THREADS_PER_WARP = Traits::THREADS_PER_WARP;
  constexpr int32_t THREADS_PER_BLOCK = Traits::THREADS_PER_BLOCK;

  constexpr int32_t MAX_FEATURE_DIM_PER_BLOCK =
      Traits::MAX_FEATURE_DIM_PER_BLOCK;
  constexpr int32_t MAX_FEATURE_DIM_PER_BLOCK_PADDED =
      Traits::MAX_FEATURE_DIM_PER_BLOCK_PADDED;
  constexpr int32_t NUM_DENSE_X_SHARED_PER_BUFFER_PADDED =
      Traits::NUM_DENSE_X_SHARED_PER_BUFFER_PADDED;
  constexpr int32_t DENSE_X_STRIDE_PER_MMA = Traits::DENSE_X_STRIDE_PER_MMA;

  const int32_t wid = threadIdx.y; // warp_index handling multi-dimension > 16.
  const int32_t laneid = threadIdx.x; // lanid of each warp.  [0, 31]
  const int32_t tid =
      threadIdx.y * THREADS_PER_WARP + laneid; // threadid of each block.

  alignas(
      16) extern __shared__ float dense_X[]; // row-major dense X shared memory.
  alignas(16)
      __shared__ uint32_t sparse_A[NUM_BUFFERS * 4]; // row-major sparse matrix

  DynamicWaveScheduler<BLK_H, BLK_W, MAX_FEATURE_DIM_PER_BLOCK> scheduler(
      embedding_dim, num_nodes);

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ uint64_t bars[NUM_BARRIERS];
  if (tid == 0) {
#pragma unroll
    for (int32_t i = 0; i < NUM_BARRIERS; i++) {
      // init(&bars[i], THREADS_PER_BLOCK);
      mbarrier_init(bars[i], THREADS_PER_BLOCK);
    }
  }

  __syncthreads();

  int32_t current_buffer = NUM_BUFFERS - 1;

  const bool is_producer_warp = (wid >= CONSUMER_WARPS_PER_BLOCK);
  if (is_producer_warp) {
    for (; scheduler.isvalid(); scheduler.step()) {
      int32_t num_TC_blocks = blks_offsets[scheduler.curr_row + 1] -
                              blks_offsets[scheduler.curr_row];

      // auto tma_load_size = cuda::aligned_size_t<16>(scheduler.step_blocks *
      //                                               BLK_H * sizeof(float));
      int32_t tma_load_size = scheduler.step_blocks * BLK_H * sizeof(float);

      auto spa_packed_start =
          blks_offsets[scheduler.curr_row] * BLK_H * BLK_W / 32;
      auto ind_start = blks_offsets[scheduler.curr_row] * BLK_W;

#pragma unroll 4
      for (unsigned i = 0; i < num_TC_blocks; ++i) {
        // int offset_spa_packed = i * BLK_H * BLK_W / 32 + spa_packed_start;
        current_buffer = (current_buffer + 1) % NUM_BUFFERS;

        auto &ld_bar = bars[current_buffer];
        auto &mma_bar = bars[NUM_BUFFERS + current_buffer];

        auto sparse_A_ptr =
            cast_smem_ptr_to_uint(&sparse_A[current_buffer * 4]);
        // auto sparse_A_ptr = &sparse_A[current_buffer * 4];
        auto dense_X_ptr =
            &dense_X[current_buffer * NUM_DENSE_X_SHARED_PER_BUFFER_PADDED];

        {
          // mma_bar.arrive_and_wait();
          mbarrier_arrive_and_wait(mma_bar);
          { // TESTING
            // auto phase = ld_bar.arrive();
            // continue;
          }

          if (laneid == 0) {
            int32_t offset_spa_packed =
                i * BLK_H * BLK_W / 32 + spa_packed_start;
            auto packed_ptr = &hspa_packed[offset_spa_packed];

            asm volatile("{\n\t"
                         "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                         "\t}" ::"r"(sparse_A_ptr),
                         "l"(packed_ptr));

            // Uint4 packed = *reinterpret_cast<const Uint4 *>(packed_ptr);
            // *reinterpret_cast<Uint4 *>(sparse_A_ptr) = packed;
          }

          int32_t offset_ind = i * BLK_W + ind_start;
          auto src_ofs = hind[laneid + offset_ind];
          if (laneid < BLK_W) {
            memcpy_async(dense_X_ptr +
                             laneid * MAX_FEATURE_DIM_PER_BLOCK_PADDED,
                         input + src_ofs * embedding_dim +
                             scheduler.start_col_blk * BLK_H,
                         tma_load_size, ld_bar);
          }

          // auto phase = ld_bar.arrive();
          cp_async_mbarrier_arrive(ld_bar);
          mbarrier_arrive(ld_bar);
        }
      }
    }
  } else {
#pragma unroll
    for (int32_t b = 0; b < NUM_BUFFERS; b++) {
      // auto phase = bars[NUM_BUFFERS + b].arrive();
      mbarrier_arrive(bars[NUM_BUFFERS + b]);
    }

    uint32_t a_frag[4];
    uint32_t b_frag[MAX_MMAS_PER_WARP][2][2];
    float acc_frag[MAX_MMAS_PER_WARP][2][4];

    static_assert(sizeof(float[2][4]) == 32);

    for (; scheduler.isvalid(); scheduler.step()) {
      int32_t num_TC_blocks = blks_offsets[scheduler.curr_row + 1] -
                              blks_offsets[scheduler.curr_row];

      const int32_t num_blocks = scheduler.step_blocks;
      const int32_t num_mmas = num_blocks / CONSUMER_WARPS_PER_BLOCK +
                               (wid < num_blocks % CONSUMER_WARPS_PER_BLOCK);

#pragma unroll
      for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++) {
#pragma unroll
        for (int32_t i = 0; i < 2; ++i) {
#pragma unroll
          for (int32_t v = 0; v < 4; ++v) {
            acc_frag[f][i][v] = 0.0f;
          }
        }
      }

      const auto group_id = laneid >> 2;
      const auto lane_id_in_group = laneid % 4;

#pragma unroll 4
      for (int32_t i = 0; i < num_TC_blocks; i++) {
        current_buffer = (current_buffer + 1) % NUM_BUFFERS;

        auto &ld_bar = bars[current_buffer];
        auto &mma_bar = bars[NUM_BUFFERS + current_buffer];

        // uint32_t sparse_A_ptr =
        // cast_smem_ptr_to_uint(&sparse_A[current_buffer * BLK_W * BLK_H]);
        auto sparse_A_ptr = &sparse_A[current_buffer * 4];
        auto dense_X_ptr =
            &dense_X[current_buffer * NUM_DENSE_X_SHARED_PER_BUFFER_PADDED];

        {
          // ld_bar.arrive_and_wait();
          mbarrier_arrive_and_wait(ld_bar);

#pragma unroll
          for (int32_t t = 0; t < 4; t++) {
            a_frag[t] = *reinterpret_cast<uint32_t *>(sparse_A_ptr + t);
            constexpr uint32_t float_one_uint32 = 0x3f800000;
            constexpr uint32_t float_zero_uint32 = 0x0;
            a_frag[t] = (a_frag[t] & (0x1 << laneid)) ? float_one_uint32
                                                      : float_zero_uint32;
            asm volatile("cvt.rna.tf32.f32 %0, %0;\n" : "+r"(a_frag[t]));
          }

          auto dense_base = dense_X_ptr + wid * BLK_H;
#pragma unroll
          for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++) { /// move it out
            if (f < num_mmas) {
              b_frag[f][0][0] = *reinterpret_cast<uint32_t *>(
                  dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id +
                  lane_id_in_group * MAX_FEATURE_DIM_PER_BLOCK_PADDED);
              b_frag[f][0][1] = *reinterpret_cast<uint32_t *>(
                  dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id +
                  (lane_id_in_group + 4) * MAX_FEATURE_DIM_PER_BLOCK_PADDED);
              b_frag[f][1][0] = *reinterpret_cast<uint32_t *>(
                  dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id + 8 +
                  lane_id_in_group * MAX_FEATURE_DIM_PER_BLOCK_PADDED);
              b_frag[f][1][1] = *reinterpret_cast<uint32_t *>(
                  dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id + 8 +
                  (lane_id_in_group + 4) * MAX_FEATURE_DIM_PER_BLOCK_PADDED);

#pragma unroll
              for (int32_t t = 0; t < 2; t++) {
#pragma unroll
                for (int32_t v = 0; v < 2; v++) {
                  asm volatile("cvt.rna.tf32.f32 %0, %0;"
                               : "+r"(b_frag[f][t][v]));
                }
              }

              // Perform the matrix multiplication.
              MMA<16, 8, 8>::mma(a_frag, b_frag[f][0], acc_frag[f][0]);
              MMA<16, 8, 8>::mma(a_frag, b_frag[f][1], acc_frag[f][1]);
            }
          }

          // auto phase = mma_bar.arrive();
          mbarrier_arrive(mma_bar);
        }
      }

      auto offset_base = output + scheduler.curr_row * BLK_H * embedding_dim +
                         (scheduler.start_col_blk + wid) * BLK_H;

#pragma unroll
      for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++) {
        if (f < num_mmas) {
#pragma unroll
          for (int32_t i = 0; i < 2; i++) {
#pragma unroll
            for (int32_t v = 0; v < 4; v += 2) {
              auto row = group_id + (v / 2) * 8;
              // auto col = (lane_id_in_group * 2) + (v & 0x1) + i * 8;
              auto col = (lane_id_in_group * 2) + i * 8;

              auto ptr = offset_base + f * DENSE_X_STRIDE_PER_MMA +
                         row * embedding_dim + col;

              // *(offset_base + f * DENSE_X_STRIDE_PER_MMA + row *
              // embedding_dim +
              //   col) = acc_frag[f][i][v];

              // asm volatile(
              //     "st.global.wt.v2.f32 [%0], {%1, %2};\n" ::"l"(ptr),
              //     "f"(acc_frag[f][i][v]), "f"(acc_frag[f][i][v + 1]));

              double val = *(double *)&acc_frag[f][i][v]; // ld.local.f64
              asm volatile("st.global.wt.f64 [%0], %1;\n" ::"l"(ptr), "d"(val));
            }
          }
        }
      }
    }
  }

  if (tid == 0) {
#pragma unroll
    for (int32_t i = 0; i < NUM_BARRIERS; i++) {
      // (&bars[i])->~barrier();
      invalidate(bars[i]);
    }
  }
}

template <int32_t NUM_AGENTS, int32_t NUM_BUFFERS, int32_t MAX_MMAS_PER_WARP,
          int32_t CONSUMER_WARPS_PER_BLOCK, int32_t PADDING_SIZE = 0>
__global__ void spmm_mma161616_spa_swizzle_d(
    const int *__restrict__ blks_offsets,
    const uint32_t *__restrict__ hspa_packed, const int32_t *__restrict__ hind,
    const int num_nodes, const int num_edges,
    const int embedding_dim,         // embedding dimension.
    const float *__restrict__ input, // input feature matrix.
    float *__restrict__ output       // aggreAGNNed output feature matrix.
) {
  using Traits =
      PersistenKernelTraits<float, NUM_AGENTS, NUM_BUFFERS, MAX_MMAS_PER_WARP,
                            CONSUMER_WARPS_PER_BLOCK, PADDING_SIZE>;

  constexpr int32_t NUM_BARRIERS = Traits::NUM_BARRIERS;

  constexpr int32_t THREADS_PER_WARP = Traits::THREADS_PER_WARP;
  constexpr int32_t THREADS_PER_BLOCK = Traits::THREADS_PER_BLOCK;

  constexpr int32_t MAX_FEATURE_DIM_PER_BLOCK =
      Traits::MAX_FEATURE_DIM_PER_BLOCK;
  constexpr int32_t MAX_FEATURE_DIM_PER_BLOCK_PADDED =
      Traits::MAX_FEATURE_DIM_PER_BLOCK_PADDED;
  constexpr int32_t NUM_DENSE_X_SHARED_PER_BUFFER_PADDED =
      2 * Traits::NUM_DENSE_X_SHARED_PER_BUFFER_PADDED;
  constexpr int32_t DENSE_X_STRIDE_PER_MMA = Traits::DENSE_X_STRIDE_PER_MMA;

  const int32_t bid = blockIdx.x;
  const int32_t wid = threadIdx.y; // warp_index handling multi-dimension > 16.
  const int32_t laneid = threadIdx.x; // lanid of each warp.  [0, 31]
  const int32_t tid =
      threadIdx.y * THREADS_PER_WARP + laneid; // threadid of each block.

  alignas(
      16) extern __shared__ float dense_X[]; // row-major dense X shared memory.
  alignas(16) __shared__ uint32_t sparse_A[NUM_BUFFERS * 4 * 2];

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ uint64_t bars[NUM_BARRIERS];
  if (tid == 0) {
#pragma unroll
    for (int32_t i = 0; i < NUM_BARRIERS; i++) {
      // init(&bars[i], THREADS_PER_BLOCK);
      mbarrier_init(bars[i], THREADS_PER_BLOCK);
    }
  }

  __syncthreads();

  int32_t current_buffer = NUM_BUFFERS - 1;

  const int32_t num_TC_blocks = blks_offsets[bid + 1] - blks_offsets[bid];

  const int32_t num_blocks_per_row = embedding_dim / BLK_H;
  constexpr int32_t MAX_BLOCKS_PER_STAGE = MAX_FEATURE_DIM_PER_BLOCK / BLK_H;

  if (bid >= num_nodes / BLK_H) {
    return;
  }

  const bool is_producer_warp = (wid >= CONSUMER_WARPS_PER_BLOCK);
  if (is_producer_warp) {
    auto spa_packed_start = blks_offsets[bid] * BLK_H * BLK_W / 32;
    auto ind_start = blks_offsets[bid] * BLK_W;

    for (int32_t s = 0; s < num_blocks_per_row; s += MAX_BLOCKS_PER_STAGE) {
      int32_t step_blocks = min(MAX_BLOCKS_PER_STAGE, num_blocks_per_row - s);

      int32_t tma_load_size = step_blocks * BLK_H * sizeof(float);
      // PRINT_BT(0, 0, 11, "producer: %d, %d, %d, %d, %d\n", s, bid,
      // num_TC_blocks,
      //           num_blocks_per_row, step_blocks);

      for (unsigned i = 0; i < num_TC_blocks; i += 2) {
        // int offset_spa_packed = i * BLK_H * BLK_W / 32 + spa_packed_start;
        current_buffer = (current_buffer + 1) % NUM_BUFFERS;

        auto &ld_bar = bars[current_buffer];
        auto &mma_bar = bars[NUM_BUFFERS + current_buffer];

        auto dense_X_ptr =
            &dense_X[current_buffer * NUM_DENSE_X_SHARED_PER_BUFFER_PADDED];

        {
          mbarrier_arrive_and_wait(mma_bar);
          { // TESTING
            // mbarrier_arrive(ld_bar);
            //  continue;
          }

          if (laneid < 2 && i + laneid < num_TC_blocks) {
            auto sparse_A_ptr = cast_smem_ptr_to_uint(
                &sparse_A[current_buffer * 8 + laneid * 4]);
            // auto sparse_A_ptr = &sparse_A[current_buffer * 4];

            int32_t offset_spa_packed =
                i * BLK_H * BLK_W / 32 + spa_packed_start;
            auto packed_ptr = &hspa_packed[offset_spa_packed + laneid * 4];

            asm volatile("{\n\t"
                         "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                         "\t}" ::"r"(sparse_A_ptr),
                         "l"(packed_ptr));
          }

          int32_t offset_ind = i * BLK_W + ind_start;
          if (laneid < BLK_W * 2 && i + laneid / BLK_W < num_TC_blocks) {
            auto src_ofs = hind[laneid + offset_ind];
            memcpy_async(dense_X_ptr +
                             laneid * MAX_FEATURE_DIM_PER_BLOCK_PADDED,
                         input + src_ofs * embedding_dim + s * BLK_H,
                         tma_load_size, ld_bar);
          }

          cp_async_mbarrier_arrive(ld_bar);
          mbarrier_arrive(ld_bar);
        }
      }
    }
  } else {
#pragma unroll
    for (int32_t b = 0; b < NUM_BUFFERS; b++) {
      // auto phase = bars[NUM_BUFFERS + b].arrive();
      mbarrier_arrive(bars[NUM_BUFFERS + b]);
    }

    uint32_t a_frag[2][4];
    uint32_t b_frag[2][MAX_MMAS_PER_WARP][2][2];
    float acc_frag[MAX_MMAS_PER_WARP][2][4];

    static_assert(sizeof(float[2][4]) == 32);

    for (int32_t s = 0; s < num_blocks_per_row; s += MAX_BLOCKS_PER_STAGE) {
      const int32_t step_blocks =
          min(MAX_BLOCKS_PER_STAGE, num_blocks_per_row - s);

      const int32_t num_blocks = step_blocks;
      const int32_t num_mmas = num_blocks / CONSUMER_WARPS_PER_BLOCK +
                               (wid < num_blocks % CONSUMER_WARPS_PER_BLOCK);

#pragma unroll
      for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++) {
#pragma unroll
        for (int32_t i = 0; i < 2; ++i) {
#pragma unroll
          for (int32_t v = 0; v < 4; ++v) {
            acc_frag[f][i][v] = 0.0f;
          }
        }
      }

      const auto group_id = laneid >> 2;
      const auto lane_id_in_group = laneid % 4;

      for (int32_t i = 0; i < num_TC_blocks; i += 2) {
        current_buffer = (current_buffer + 1) % NUM_BUFFERS;

        auto &ld_bar = bars[current_buffer];
        auto &mma_bar = bars[NUM_BUFFERS + current_buffer];

        // uint32_t sparse_A_ptr =
        // cast_smem_ptr_to_uint(&sparse_A[current_buffer * BLK_W * BLK_H]);
        auto sparse_A_ptr = &sparse_A[current_buffer * 8];
        auto dense_X_ptr =
            &dense_X[current_buffer * NUM_DENSE_X_SHARED_PER_BUFFER_PADDED];

        {
          mbarrier_arrive_and_wait(ld_bar);
          {
            // mbarrier_arrive(mma_bar);
            // continue;
          }

#pragma unroll
          for (int32_t k = 0; k < 2; k++) {
#pragma unroll
            for (int32_t t = 0; t < 4; t++) {
              a_frag[k][t] =
                  *reinterpret_cast<uint32_t *>(sparse_A_ptr + k * 4 + t);
              constexpr uint32_t float_one_uint32 = 0x3f800000;
              constexpr uint32_t float_zero_uint32 = 0x0;
              a_frag[k][t] = (a_frag[k][t] & (0x1 << laneid))
                                 ? float_one_uint32
                                 : float_zero_uint32;
              asm volatile("cvt.rna.tf32.f32 %0, %0;\n" : "+r"(a_frag[k][t]));
            }
          }

#pragma unroll
          for (int32_t k = 0; k < 2; k++) {
            auto dense_base = dense_X_ptr + wid * BLK_H +
                              k * BLK_W * MAX_FEATURE_DIM_PER_BLOCK_PADDED;
#pragma unroll
            for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++) { /// move it out
              if (f < num_mmas && i + k < num_TC_blocks) {
                // 0.3 ms
                b_frag[k][f][0][0] = *reinterpret_cast<uint32_t *>(
                    dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id +
                    lane_id_in_group * MAX_FEATURE_DIM_PER_BLOCK_PADDED);
                b_frag[k][f][0][1] = *reinterpret_cast<uint32_t *>(
                    dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id +
                    (lane_id_in_group + 4) * MAX_FEATURE_DIM_PER_BLOCK_PADDED);
                b_frag[k][f][1][0] = *reinterpret_cast<uint32_t *>(
                    dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id + 8 +
                    lane_id_in_group * MAX_FEATURE_DIM_PER_BLOCK_PADDED);
                b_frag[k][f][1][1] = *reinterpret_cast<uint32_t *>(
                    dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id + 8 +
                    (lane_id_in_group + 4) * MAX_FEATURE_DIM_PER_BLOCK_PADDED);

#pragma unroll
                for (int32_t t = 0; t < 2; t++) {
#pragma unroll
                  for (int32_t v = 0; v < 2; v++) {
                    asm volatile("cvt.rna.tf32.f32 %0, %0;"
                                 : "+r"(b_frag[k][f][t][v]));
                  }
                }

                // Perform the matrix multiplication.
                MMA<16, 8, 8>::mma(a_frag[k], b_frag[k][f][0], acc_frag[f][0]);
                MMA<16, 8, 8>::mma(a_frag[k], b_frag[k][f][1], acc_frag[f][1]);
              }
            }
          }

          mbarrier_arrive(mma_bar);
        }
      }

      auto offset_base =
          output + bid * BLK_H * embedding_dim + (s + wid) * BLK_H;

#pragma unroll
      for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++) {
        if (f < num_mmas) {
#pragma unroll
          for (int32_t i = 0; i < 2; i++) {
#pragma unroll
            for (int32_t v = 0; v < 4; v += 2) {
              auto row = group_id + (v / 2) * 8;
              // auto col = (lane_id_in_group * 2) + (v & 0x1) + i * 8;
              auto col = (lane_id_in_group * 2) + i * 8;

              auto ptr = offset_base + f * DENSE_X_STRIDE_PER_MMA +
                         row * embedding_dim + col;

              // ptr[0] = acc_frag[f][i][v];
              // ptr[1] = acc_frag[f][i][v + 1];

              // asm volatile(
              //     "st.global.wt.v2.f32 [%0], {%1, %2};\n" ::"l"(ptr),
              //     "f"(acc_frag[f][i][v]), "f"(acc_frag[f][i][v + 1]));

              double val = *(double *)&acc_frag[f][i][v]; // ld.local.f64
              asm volatile("st.global.wt.f64 [%0], %1;\n" ::"l"(ptr), "d"(val));
            }
          }
        }
      }
    }
  }

  if (tid == 0) {
#pragma unroll
    for (int32_t i = 0; i < NUM_BARRIERS; i++) {
      // (&bars[i])->~barrier();
      invalidate(bars[i]);
    }
  }
}

template <int32_t NUM_AGENTS, int32_t NUM_BUFFERS, int32_t MAX_MMAS_PER_WARP,
          int32_t CONSUMER_WARPS_PER_BLOCK, int32_t PADDING_SIZE = 0>
__global__ void spmm_mma161616_spa_swizzle_dd(
    const int *__restrict__ blks_offsets,
    const uint32_t *__restrict__ hspa_packed, const int32_t *__restrict__ hind,
    const int num_nodes, const int num_edges,
    const int embedding_dim,         // embedding dimension.
    const float *__restrict__ input, // input feature matrix.
    float *__restrict__ output       // aggreAGNNed output feature matrix.
) {
  using Traits =
      PersistenKernelTraits<float, NUM_AGENTS, NUM_BUFFERS, MAX_MMAS_PER_WARP,
                            CONSUMER_WARPS_PER_BLOCK, PADDING_SIZE>;

  constexpr int32_t NUM_BARRIERS = Traits::NUM_BARRIERS;

  constexpr int32_t THREADS_PER_WARP = Traits::THREADS_PER_WARP;
  constexpr int32_t THREADS_PER_BLOCK = Traits::THREADS_PER_BLOCK;

  constexpr int32_t MAX_FEATURE_DIM_PER_BLOCK =
      Traits::MAX_FEATURE_DIM_PER_BLOCK;
  constexpr int32_t MAX_FEATURE_DIM_PER_BLOCK_PADDED =
      Traits::MAX_FEATURE_DIM_PER_BLOCK_PADDED;
  constexpr int32_t NUM_DENSE_X_SHARED_PER_BUFFER_PADDED =
      2 * Traits::NUM_DENSE_X_SHARED_PER_BUFFER_PADDED;
  constexpr int32_t DENSE_X_STRIDE_PER_MMA = Traits::DENSE_X_STRIDE_PER_MMA;

  const int32_t wid = threadIdx.y; // warp_index handling multi-dimension > 16.
  const int32_t laneid = threadIdx.x; // lanid of each warp.  [0, 31]
  const int32_t tid =
      threadIdx.y * THREADS_PER_WARP + laneid; // threadid of each block.

  alignas(
      16) extern __shared__ float dense_X[]; // row-major dense X shared memory.
  alignas(16) __shared__ uint32_t sparse_A[NUM_BUFFERS * 4 * 2];

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ uint64_t bars[NUM_BARRIERS];
  if (tid == 0) {
#pragma unroll
    for (int32_t i = 0; i < NUM_BARRIERS; i++) {
      // init(&bars[i], THREADS_PER_BLOCK);
      mbarrier_init(bars[i], THREADS_PER_BLOCK);
    }
  }

  __syncthreads();

  int32_t current_buffer = NUM_BUFFERS - 1;

  const int32_t num_blocks_per_row = embedding_dim / BLK_H;
  constexpr int32_t MAX_BLOCKS_PER_STAGE = MAX_FEATURE_DIM_PER_BLOCK / BLK_H;

  // int32_t row = bid / div_round_up(embedding_dim, MAX_FEATURE_DIM_PER_BLOCK);
  // int32_t start_col_blk =
  //     bid % (div_round_up(embedding_dim, MAX_FEATURE_DIM_PER_BLOCK)) *
  //     MAX_BLOCKS_PER_STAGE;
  // int32_t step_blocks =
  //     min(MAX_BLOCKS_PER_STAGE, num_blocks_per_row - start_col_blk);

  int32_t row = blockIdx.x;
  int32_t start_col_blk = blockIdx.y * MAX_BLOCKS_PER_STAGE;
  int32_t step_blocks =
      min(MAX_BLOCKS_PER_STAGE, num_blocks_per_row - start_col_blk);

  const int32_t num_TC_blocks = blks_offsets[row + 1] - blks_offsets[row];

  if (row >= num_nodes / BLK_H) {
    return;
  }

  const bool is_producer_warp = (wid >= CONSUMER_WARPS_PER_BLOCK);
  if (is_producer_warp) {
    auto spa_packed_start = blks_offsets[row] * BLK_H * BLK_W / 32;
    auto ind_start = blks_offsets[row] * BLK_W;

    int32_t tma_load_size = step_blocks * BLK_H * sizeof(float);
    // PRINT_BT(0, 0, 11, "producer: %d, %d, %d, %d, %d\n", s, bid,
    // num_TC_blocks,
    //           num_blocks_per_row, step_blocks);

    for (unsigned i = 0; i < num_TC_blocks; i += 2) {
      // int offset_spa_packed = i * BLK_H * BLK_W / 32 + spa_packed_start;
      current_buffer = (current_buffer + 1) % NUM_BUFFERS;

      auto &ld_bar = bars[current_buffer];
      auto &mma_bar = bars[NUM_BUFFERS + current_buffer];

      auto dense_X_ptr =
          &dense_X[current_buffer * NUM_DENSE_X_SHARED_PER_BUFFER_PADDED];

      {
        // mma_bar.arrive_and_wait();
        mbarrier_arrive_and_wait(mma_bar);
        { // TESTING
          // mbarrier_arrive(ld_bar);
          // continue;
        }

        if (laneid < 2 && i + laneid < num_TC_blocks) {
          auto sparse_A_ptr =
              cast_smem_ptr_to_uint(&sparse_A[current_buffer * 8 + laneid * 4]);
          // auto sparse_A_ptr = &sparse_A[current_buffer * 4];

          int32_t offset_spa_packed = i * BLK_H * BLK_W / 32 + spa_packed_start;
          auto packed_ptr = &hspa_packed[offset_spa_packed + laneid * 4];

          asm volatile("{\n\t"
                       "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                       "\t}" ::"r"(sparse_A_ptr),
                       "l"(packed_ptr));
        }

        int32_t offset_ind = i * BLK_W + ind_start;
        if (laneid < BLK_W * 2 && i + laneid / BLK_W < num_TC_blocks) {
          auto src_ofs = hind[laneid + offset_ind];
          memcpy_async(dense_X_ptr + laneid * MAX_FEATURE_DIM_PER_BLOCK_PADDED,
                       input + src_ofs * embedding_dim + start_col_blk * BLK_H,
                       tma_load_size, ld_bar);
        }

        // auto phase = ld_bar.arrive();
        cp_async_mbarrier_arrive(ld_bar);
        mbarrier_arrive(ld_bar);
      }
    }
  } else {
#pragma unroll
    for (int32_t b = 0; b < NUM_BUFFERS; b++) {
      // auto phase = bars[NUM_BUFFERS + b].arrive();
      mbarrier_arrive(bars[NUM_BUFFERS + b]);
    }

    uint32_t a_frag[2][4];
    uint32_t b_frag[2][MAX_MMAS_PER_WARP][2][2];
    float acc_frag[MAX_MMAS_PER_WARP][2][4];

    static_assert(sizeof(float[2][4]) == 32);

    const int32_t num_blocks = step_blocks;
    const int32_t num_mmas = num_blocks / CONSUMER_WARPS_PER_BLOCK +
                             (wid < num_blocks % CONSUMER_WARPS_PER_BLOCK);

#pragma unroll
    for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++) {
#pragma unroll
      for (int32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (int32_t v = 0; v < 4; ++v) {
          acc_frag[f][i][v] = 0.0f;
        }
      }
    }

    const auto group_id = laneid >> 2;
    const auto lane_id_in_group = laneid % 4;

    for (int32_t i = 0; i < num_TC_blocks; i += 2) {
      current_buffer = (current_buffer + 1) % NUM_BUFFERS;

      auto &ld_bar = bars[current_buffer];
      auto &mma_bar = bars[NUM_BUFFERS + current_buffer];

      // uint32_t sparse_A_ptr =
      // cast_smem_ptr_to_uint(&sparse_A[current_buffer * BLK_W * BLK_H]);
      auto sparse_A_ptr = &sparse_A[current_buffer * 8];
      auto dense_X_ptr =
          &dense_X[current_buffer * NUM_DENSE_X_SHARED_PER_BUFFER_PADDED];

      {
        // ld_bar.arrive_and_wait();
        mbarrier_arrive_and_wait(ld_bar);
        {
          // mbarrier_arrive(mma_bar);
          // continue;
        }

#pragma unroll
        for (int32_t k = 0; k < 2; k++) {
#pragma unroll
          for (int32_t t = 0; t < 4; t++) {
            a_frag[k][t] =
                *reinterpret_cast<uint32_t *>(sparse_A_ptr + k * 4 + t);
            constexpr uint32_t float_one_uint32 = 0x3f800000;
            constexpr uint32_t float_zero_uint32 = 0x0;
            a_frag[k][t] = (a_frag[k][t] & (0x1 << laneid)) ? float_one_uint32
                                                            : float_zero_uint32;
            asm volatile("cvt.rna.tf32.f32 %0, %0;\n" : "+r"(a_frag[k][t]));
          }
        }

#pragma unroll
        for (int32_t k = 0; k < 2; k++) {
          auto dense_base = dense_X_ptr + wid * BLK_H +
                            k * BLK_W * MAX_FEATURE_DIM_PER_BLOCK_PADDED;
#pragma unroll
          for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++) { /// move it out
            if (f < num_mmas && i + k < num_TC_blocks) {
              // 0.3 ms
              b_frag[k][f][0][0] = *reinterpret_cast<uint32_t *>(
                  dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id +
                  lane_id_in_group * MAX_FEATURE_DIM_PER_BLOCK_PADDED);
              b_frag[k][f][0][1] = *reinterpret_cast<uint32_t *>(
                  dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id +
                  (lane_id_in_group + 4) * MAX_FEATURE_DIM_PER_BLOCK_PADDED);
              b_frag[k][f][1][0] = *reinterpret_cast<uint32_t *>(
                  dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id + 8 +
                  lane_id_in_group * MAX_FEATURE_DIM_PER_BLOCK_PADDED);
              b_frag[k][f][1][1] = *reinterpret_cast<uint32_t *>(
                  dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id + 8 +
                  (lane_id_in_group + 4) * MAX_FEATURE_DIM_PER_BLOCK_PADDED);

#pragma unroll
              for (int32_t t = 0; t < 2; t++) {
#pragma unroll
                for (int32_t v = 0; v < 2; v++) {
                  asm volatile("cvt.rna.tf32.f32 %0, %0;"
                               : "+r"(b_frag[k][f][t][v]));
                }
              }

              // Perform the matrix multiplication.
              MMA<16, 8, 8>::mma(a_frag[k], b_frag[k][f][0], acc_frag[f][0]);
              MMA<16, 8, 8>::mma(a_frag[k], b_frag[k][f][1], acc_frag[f][1]);
            }
          }
        }

        // auto phase = mma_bar.arrive();
        mbarrier_arrive(mma_bar);
      }
    }

    auto offset_base =
        output + row * BLK_H * embedding_dim + (start_col_blk + wid) * BLK_H;

#pragma unroll
    for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++) {
      if (f < num_mmas) {
#pragma unroll
        for (int32_t i = 0; i < 2; i++) {
#pragma unroll
          for (int32_t v = 0; v < 4; v += 2) {
            auto row = group_id + (v / 2) * 8;
            // auto col = (lane_id_in_group * 2) + (v & 0x1) + i * 8;
            auto col = (lane_id_in_group * 2) + i * 8;

            auto ptr = offset_base + f * DENSE_X_STRIDE_PER_MMA +
                       row * embedding_dim + col;

            // ptr[0] = acc_frag[f][i][v];
            // ptr[1] = acc_frag[f][i][v + 1];

            // asm volatile(
            //     "st.global.wt.v2.f32 [%0], {%1, %2};\n" ::"l"(ptr),
            //     "f"(acc_frag[f][i][v]), "f"(acc_frag[f][i][v + 1]));

            double val = *(double *)&acc_frag[f][i][v]; // ld.local.f64
            asm volatile("st.global.wt.f64 [%0], %1;\n" ::"l"(ptr), "d"(val));
          }
        }
      }
    }
  }

  if (tid == 0) {
#pragma unroll
    for (int32_t i = 0; i < NUM_BARRIERS; i++) {
      // (&bars[i])->~barrier();
      invalidate(bars[i]);
    }
  }
}

void voltrix_spmm_forward_cuda(
    const int *__restrict__ blks_offsets,
    const uint32_t *__restrict__ hspa_packed, const int32_t *__restrict__ hind,
    const int num_nodes, const int num_edges,
    const int embedding_dim,         // embedding dimension.
    const float *__restrict__ input, // input feature matrix.
    float *__restrict__ output,      // aggreAGNNed output feature matrix.
    int32_t model, cudaStream_t stream) {

  //   cudaStream_t steam = at::cuda::getCurrentCUDAStream();
  if (model == 0) {
    constexpr int32_t NUM_AGENTS = 2;
    constexpr int32_t PADDING_SIZE = 8;
    constexpr int32_t NUM_BUFFERS = 2;
    constexpr int32_t MAX_MMAS_PER_WARP = 8;
    constexpr int32_t CONSUMER_WARPS_PER_BLOCK = 4;

    constexpr int32_t NUM_THREAD_BLOCKS = 114 * 2;

    using Config =
        PersistenKernelConfig<float, NUM_AGENTS, NUM_BUFFERS, MAX_MMAS_PER_WARP,
                              CONSUMER_WARPS_PER_BLOCK, NUM_THREAD_BLOCKS,
                              PADDING_SIZE>;

    const dim3 GRID = dim3(num_nodes / BLK_H, 1, 1);
    constexpr dim3 BLOCK = Config::BLOCK;
    constexpr int32_t DENSE_X_SHARED_MEMORY_SIZE =
        Config::DENSE_X_SHARED_MEMORY_SIZE;

    static auto spmm_func =
        spmm_mma161616_spa_swizzle_d<NUM_AGENTS, NUM_BUFFERS, MAX_MMAS_PER_WARP,
                                     CONSUMER_WARPS_PER_BLOCK, PADDING_SIZE>;

    static int32_t _unused = [&]() {
      constexpr auto extra_dynamic_shared_memory = 3.5 * 64 * 1024;
      cudaFuncSetAttribute(spmm_func,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           extra_dynamic_shared_memory);
      return 0;
    }();

    spmm_func<<<GRID, BLOCK, DENSE_X_SHARED_MEMORY_SIZE * 2, stream>>>(
        blks_offsets, hspa_packed, hind, num_nodes, num_edges, embedding_dim,
        input, output);
  } else if (model == 1) {
    constexpr int32_t NUM_AGENTS = 2;
    constexpr int32_t PADDING_SIZE = 8;
    constexpr int32_t NUM_BUFFERS = 2;
    constexpr int32_t MAX_MMAS_PER_WARP = 4;
    constexpr int32_t CONSUMER_WARPS_PER_BLOCK = 4;

    constexpr int32_t NUM_THREAD_BLOCKS = 114 * 2;

    using Config =
        PersistenKernelConfig<float, NUM_AGENTS, NUM_BUFFERS, MAX_MMAS_PER_WARP,
                              CONSUMER_WARPS_PER_BLOCK, NUM_THREAD_BLOCKS,
                              PADDING_SIZE>;

    const dim3 GRID = dim3(num_nodes / BLK_H, 1, 1);
    constexpr dim3 BLOCK = Config::BLOCK;
    constexpr int32_t DENSE_X_SHARED_MEMORY_SIZE =
        Config::DENSE_X_SHARED_MEMORY_SIZE;

    static auto spmm_func =
        spmm_mma161616_spa_swizzle_d<NUM_AGENTS, NUM_BUFFERS, MAX_MMAS_PER_WARP,
                                     CONSUMER_WARPS_PER_BLOCK, PADDING_SIZE>;

    static int32_t _unused = [&]() {
      constexpr auto extra_dynamic_shared_memory = 3.5 * 64 * 1024;
      cudaFuncSetAttribute(spmm_func,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           extra_dynamic_shared_memory);
      return 0;
    }();

    spmm_func<<<GRID, BLOCK, DENSE_X_SHARED_MEMORY_SIZE * 2, stream>>>(
        blks_offsets, hspa_packed, hind, num_nodes, num_edges, embedding_dim,
        input, output);
  } else if (model == 2) {
    constexpr int32_t NUM_AGENTS = 2;
    constexpr int32_t PADDING_SIZE = 8;
    constexpr int32_t NUM_BUFFERS = 4;
    constexpr int32_t MAX_MMAS_PER_WARP = 2;
    constexpr int32_t CONSUMER_WARPS_PER_BLOCK = 4;

    constexpr int32_t NUM_THREAD_BLOCKS = 114 * 2;

    using Config =
        PersistenKernelConfig<float, NUM_AGENTS, NUM_BUFFERS, MAX_MMAS_PER_WARP,
                              CONSUMER_WARPS_PER_BLOCK, NUM_THREAD_BLOCKS,
                              PADDING_SIZE>;

    const dim3 GRID =
        dim3(num_nodes / BLK_H,
             div_round_up(static_cast<int32_t>(embedding_dim),
                          Config::Traits::MAX_FEATURE_DIM_PER_BLOCK),
             1);
    constexpr dim3 BLOCK = Config::BLOCK;
    constexpr int32_t DENSE_X_SHARED_MEMORY_SIZE =
        Config::DENSE_X_SHARED_MEMORY_SIZE;

    static auto spmm_func =
        spmm_mma161616_spa_swizzle_dd<NUM_AGENTS, NUM_BUFFERS,
                                      MAX_MMAS_PER_WARP,
                                      CONSUMER_WARPS_PER_BLOCK, PADDING_SIZE>;

    static int32_t _unused = [&]() {
      constexpr auto extra_dynamic_shared_memory = 3.5 * 64 * 1024;
      cudaFuncSetAttribute(spmm_func,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           extra_dynamic_shared_memory);
      return 0;
    }();

    spmm_func<<<GRID, BLOCK, DENSE_X_SHARED_MEMORY_SIZE * 2, stream>>>(
        blks_offsets, hspa_packed, hind, num_nodes, num_edges, embedding_dim,
        input, output);

  } else {
    throw std::runtime_error("Invalid model");
  }
}

} // namespace voltrix

#endif // VOLTRIX_SPMM_KERNELS_CUH_
