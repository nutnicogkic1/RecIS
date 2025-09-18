#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/types.h>

namespace recis {
namespace functional {

constexpr size_t MAX_THREADS_PER_BLOCK = 1024;

// Derived from FBGEMM: walk_down_tensor_storage_tree_, StackArray,
// RAGGED_TENSOR_DISPATCH_DIMS
// Source: https://github.com/pytorch/FBGEMM
//
// Original notice:
//   Copyright (c) Meta Platforms, Inc. and affiliates.
//   Licensed under the BSD-style license.
//   All rights reserved.
//   found in the root directory of this source tree
//
// In this project, a copy of the applicable BSD 3-Clause license is provided
// at: third_party/licenses/LICENSE.BSD
//
// Modifications: adapted to mutli type index
constexpr size_t kStackArrayMaxDims = 5;
template <typename T>
struct StackArray {
  T vals[kStackArrayMaxDims];
  size_t ndim;
};

template <int NUM_JAGGED_DIM, typename index_t>
__device__ bool walk_down_tensor_storage_tree_(
    index_t &offset, const index_t flattened_jagged_idx,
    const StackArray<int64_t> &jagged_dims,
    const StackArray<index_t *> &x_offsets) {
  // compute coorindates
  index_t jagged_coords[NUM_JAGGED_DIM];
  index_t j_temp = flattened_jagged_idx;
#pragma unroll
  for (int d = NUM_JAGGED_DIM - 1; d >= 0; --d) {
    const int64_t jagged_size = jagged_dims.vals[d];
    jagged_coords[d] = j_temp % jagged_size;
    j_temp /= jagged_size;
  }

  // walk down the tree
  bool is_zero = false;
#pragma unroll
  for (int d = 0; d < NUM_JAGGED_DIM; ++d) {
    const index_t begin = x_offsets.vals[d][offset];
    const index_t end = x_offsets.vals[d][offset + 1];
    if (jagged_coords[d] >= end - begin) {
      is_zero = true;
      break;
    }
    offset = begin + jagged_coords[d];
  }
  return is_zero;
}

#define RAGGED_TENSOR_DISPATCH_DIMS()                                          \
  switch (num_ragged_dim) {                                                    \
    case 1:                                                                    \
      INVOKE_KERNEL_WITH_DIM(1);                                               \
      break;                                                                   \
    case 2:                                                                    \
      INVOKE_KERNEL_WITH_DIM(2);                                               \
      break;                                                                   \
    case 3:                                                                    \
      INVOKE_KERNEL_WITH_DIM(3);                                               \
      break;                                                                   \
    case 4:                                                                    \
      INVOKE_KERNEL_WITH_DIM(4);                                               \
      break;                                                                   \
    case 5:                                                                    \
      INVOKE_KERNEL_WITH_DIM(5);                                               \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(false, "unsupported number of ragged dim ", num_ragged_dim); \
  }

template <typename index_t>
__device__ __host__ __forceinline__ index_t binary_search(index_t index,
                                                          const index_t *splits,
                                                          index_t splits_size) {
  index_t l = 0, r = splits_size - 2;  // splits_size is definitely >= 2
  while (l <= r) {
    index_t mid = (l + r) >> 1;
    if (splits[mid] <= index && splits[mid + 1] > index) {
      return mid;
    } else if (index < splits[mid]) {
      r = mid - 1;
    } else {
      l = mid + 1;
    }
  }
  return -1;
}

__device__ __host__ __forceinline__ uint64_t murmur_hash_64(uint64_t *keys,
                                                            uint64_t seed,
                                                            int size) {
  const uint64_t m = 0xc6a4a7935bd1e995ULL;
  const int r = 47;
  uint64_t len = size * sizeof(uint64_t);
  uint64_t h = seed ^ (len * m);
  for (int i = 0; i < size; ++i) {
    uint64_t k = keys[i];
    k *= m;
    k ^= k >> r;
    k *= m;
    h ^= k;
    h *= m;
  }

  h ^= h >> r;
  h *= m;
  h ^= h >> r;
  return h;
}

inline bool all_cuda(const std::vector<at::Tensor> &ts) {
  if (ts.empty()) {
    return true;
  }
  return std::all_of(ts.begin(), ts.end(),
                     [&](const at::Tensor &t) { return t.is_cuda(); });
}

inline bool all_cpu(const std::vector<at::Tensor> &ts) {
  if (ts.empty()) {
    return true;
  }
  return std::all_of(ts.begin(), ts.end(),
                     [&](const at::Tensor &t) { return !t.is_cuda(); });
}

inline bool all_same_type(const std::vector<at::Tensor> &ts,
                          torch::Dtype dtype = torch::Dtype::Undefined) {
  if (ts.empty()) {
    return true;
  }
  if (dtype == torch::Dtype::Undefined) {
    dtype = c10::typeMetaToScalarType(ts.front().dtype());
  }
  return std::all_of(ts.begin(), ts.end(), [&](const at::Tensor &t) {
    return c10::typeMetaToScalarType(t.dtype()) == dtype;
  });
}

template <typename index_t>
inline bool IsSegCompleted(const std::vector<index_t> &pos, int dim,
                           const std::vector<index_t *> &offset) {
  return pos[dim + 1] >= offset[dim][pos[dim] + 1];
}

}  // namespace functional
}  // namespace recis
