#ifndef VOLTRIX_TRAITS_H_
#define VOLTRIX_TRAITS_H_

#include <cstdint>

#define BLK_H 16
#define BLK_W 8
#define WARP_SIZE 32
#define WPB 4
#define NUM_PIP 2
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

template <typename T, int32_t NUM_AGENTS, int32_t NUM_BUFFERS,
          int32_t MAX_MMAS_PER_WARP, int32_t CONSUMER_WARPS_PER_BLOCK,
          int32_t PADDING_SIZE = 0>
struct PersistenKernelTraits {
  static_assert(NUM_AGENTS == 2, "NUM_AGENTS should be 2");
  // static_assert(PADDING_SIZE == 2, "NUM_AGENTS should be 2");

  static constexpr int32_t NUM_BARRIERS_PER_BUFFER = 2;

  static constexpr int32_t NUM_BARRIERS =
      NUM_BUFFERS * NUM_BARRIERS_PER_BUFFER * (NUM_AGENTS - 1);

  static constexpr int32_t BYTES_PER_SCALAR = sizeof(T);
  static constexpr int32_t DENSE_X_STRIDE_PER_MMA =
      CONSUMER_WARPS_PER_BLOCK * BLK_H;
  static constexpr int32_t MAX_FEATURE_DIM_PER_BLOCK =
      MAX_MMAS_PER_WARP * CONSUMER_WARPS_PER_BLOCK * BLK_H;

  static constexpr int32_t MAX_FEATURE_DIM_PER_BLOCK_PADDED =
      MAX_MMAS_PER_WARP * CONSUMER_WARPS_PER_BLOCK * BLK_H + PADDING_SIZE;

  static constexpr int32_t NUM_DENSE_X_SHARED_PER_BUFFER =
      MAX_FEATURE_DIM_PER_BLOCK * BLK_W;
  static constexpr int32_t NUM_DENSE_X_SHARED_PER_BUFFER_PADDED =
      MAX_FEATURE_DIM_PER_BLOCK * BLK_W + PADDING_SIZE * BLK_W;

  static_assert(MAX_FEATURE_DIM_PER_BLOCK % BLK_H == 0,
                "MaxFeatureDimPerBlock should be divisible by BLK_H");
  static_assert(MAX_FEATURE_DIM_PER_BLOCK / BLK_H % CONSUMER_WARPS_PER_BLOCK ==
                    0,
                "MaxFeatureDimPerBlock / BLK_H should be divisible by "
                "ConsumerWarpsPerBlock");

  static constexpr int32_t MaxMMAsPerWarp =
      MAX_FEATURE_DIM_PER_BLOCK / BLK_H / CONSUMER_WARPS_PER_BLOCK;
  static constexpr int32_t PRODUCER_WARPS_PER_BLOCK = 1;
  static constexpr int32_t WARPS_PER_BLOCK =
      CONSUMER_WARPS_PER_BLOCK + PRODUCER_WARPS_PER_BLOCK;  // 12

  static constexpr int32_t THREADS_PER_WARP = 32;

  static constexpr int32_t THREADS_PER_BLOCK =
      WARPS_PER_BLOCK * THREADS_PER_WARP;  // 384
  static constexpr int32_t THREADS_PER_PRODUCER =
      PRODUCER_WARPS_PER_BLOCK * THREADS_PER_WARP;  // 32
  static constexpr int32_t THREADS_PER_CONSUMER =
      CONSUMER_WARPS_PER_BLOCK * THREADS_PER_WARP;  // 352
};

template <typename T, int32_t NUM_BUFFERS, int32_t MAX_MMAS_PER_WARP,
          int32_t CONSUMER_WARPS_PER_BLOCK>
struct PersistenKernelTraits<T, 3, NUM_BUFFERS, MAX_MMAS_PER_WARP,

                             CONSUMER_WARPS_PER_BLOCK> {
  static constexpr int32_t NUM_AGENTS = 3;
  static constexpr int32_t NUM_BARRIERS_PER_BUFFER = 2;

  static constexpr int32_t NUM_BARRIERS =
      NUM_BUFFERS * NUM_BARRIERS_PER_BUFFER * (NUM_AGENTS - 1);

  static constexpr int32_t BYTES_PER_SCALAR = sizeof(T);
  static constexpr int32_t DENSE_X_STRIDE_PER_MMA =
      CONSUMER_WARPS_PER_BLOCK * BLK_H;
  static constexpr int32_t MAX_FEATURE_DIM_PER_BLOCK =
      MAX_MMAS_PER_WARP * CONSUMER_WARPS_PER_BLOCK * BLK_H;

  static constexpr int32_t NUM_DENSE_X_SHARED_PER_BUFFER =
      MAX_FEATURE_DIM_PER_BLOCK * BLK_W;
  static_assert(MAX_FEATURE_DIM_PER_BLOCK % BLK_H == 0,
                "MaxFeatureDimPerBlock should be divisible by BLK_H");
  static_assert(MAX_FEATURE_DIM_PER_BLOCK / BLK_H % CONSUMER_WARPS_PER_BLOCK ==
                    0,
                "MaxFeatureDimPerBlock / BLK_H should be divisible by "
                "ConsumerWarpsPerBlock");

  static constexpr int32_t MaxMMAsPerWarp =
      MAX_FEATURE_DIM_PER_BLOCK / BLK_H /
      CONSUMER_WARPS_PER_BLOCK;  // 4 = 44 / 11
  static constexpr int32_t PRODUCER_WARPS_PER_BLOCK = 2;
  static constexpr int32_t WARPS_PER_BLOCK =
      CONSUMER_WARPS_PER_BLOCK + PRODUCER_WARPS_PER_BLOCK;  // 12

  static constexpr int32_t THREADS_PER_WARP = 32;

  static constexpr int32_t THREADS_PER_BLOCK =
      WARPS_PER_BLOCK * THREADS_PER_WARP;  // 384
  static constexpr int32_t THREADS_PER_PRODUCER =
      PRODUCER_WARPS_PER_BLOCK * THREADS_PER_WARP;  // 32
  static constexpr int32_t THREADS_PER_CONSUMER =
      CONSUMER_WARPS_PER_BLOCK * THREADS_PER_WARP;  // 352
};

template <typename T, int32_t NUM_AGENTS, int32_t NUM_BUFFERS,
          int32_t MAX_MMAS_PER_WARP, int32_t CONSUMER_WARPS_PER_BLOCK,
          int32_t GLOBAL_NUM_BLOCKS, int32_t PADDING_SIZE = 0>
struct PersistenKernelConfig {
  using Traits =
      PersistenKernelTraits<T, NUM_AGENTS, NUM_BUFFERS, MAX_MMAS_PER_WARP,
                            CONSUMER_WARPS_PER_BLOCK, PADDING_SIZE>;

  static constexpr dim3 GRID = dim3(GLOBAL_NUM_BLOCKS, 1, 1);
  static constexpr dim3 BLOCK =
      dim3(Traits::THREADS_PER_WARP, Traits::WARPS_PER_BLOCK, 1);
  static constexpr int32_t DENSE_X_SHARED_MEMORY_SIZE =
      Traits::NUM_DENSE_X_SHARED_PER_BUFFER_PADDED * NUM_BUFFERS *
      Traits::BYTES_PER_SCALAR;
};



#endif // VOLTRIX_TRAITS_H_
