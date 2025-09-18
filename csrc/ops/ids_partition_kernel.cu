#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/ThrustAllocator.h>
#include <thrust/adjacent_difference.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <torch/extension.h>

#include "ids_partition.h"

namespace recis {
namespace functional {

template <typename T>
__device__ __forceinline__ T left_shift(T x, int k) {
  using U = typename std::make_unsigned<T>::type;
  return static_cast<T>(static_cast<U>(x) << static_cast<U>(k));
}

template <typename T>
__device__ __forceinline__ T right_shift(T x, int k) {
  using U = typename std::make_unsigned<T>::type;
  return static_cast<T>(static_cast<U>(x) >> static_cast<U>(k));
}

template <typename T>
struct HashKey {
  int hash_bits;

  __host__ __device__ T operator()(const T& x) const {
    return left_shift(x, sizeof(T) * 8 - hash_bits) | right_shift(x, hash_bits);
  }
};

template <typename T>
struct HashSlice {
  int num_parts;
  __host__ __device__ int operator()(const T& x) const {
    int part_slice_size = kSliceSize / num_parts;
    int part_extra = kSliceSize % num_parts;
    int part_slice_plus_one = part_slice_size + 1;
    int cur_part = x & (T)(kSliceSize - 1);
    return max(cur_part / part_slice_plus_one,
               (cur_part - part_extra) / part_slice_size);
  }
};

__global__ void unique_by_key_kernel(const int32_t* __restrict__ unique_indices,
                                     const int32_t* __restrict__ sorted_indices,
                                     const int64_t* __restrict__ ids,
                                     int64_t* __restrict__ unique_ids,
                                     int32_t* __restrict__ reverse_indices,
                                     int64_t N, int64_t N_Unique) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    int des_idx = unique_indices[tid];
    int src_idx = sorted_indices[tid];
    reverse_indices[src_idx] = des_idx;
    if (des_idx < N_Unique) {
      unique_ids[des_idx] = ids[src_idx];
    }
  }
}

__global__ void adjacent_difference_kernel(const int64_t* __restrict__ ids_hash,
                                           int32_t* __restrict__ unique_index,
                                           int64_t N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N && tid > 0) {
    unique_index[tid] = ids_hash[tid] != ids_hash[tid - 1];
  }
  if (tid == 0) {
    unique_index[0] = 0;
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ids_partition_cuda(
    const torch::Tensor& ids, int64_t num_parts) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::cuda::ThrustAllocator allocator;
  auto policy = thrust::cuda::par(allocator).on(stream);
  auto device = ids.device();
  int64_t N = ids.numel();

  // Allocate data
  torch::Tensor reverse_indices = torch::empty(
      {N}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  torch::Tensor ids_hash = torch::empty(
      {N}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  torch::Tensor sorted_indices = torch::arange(
      0, N + 1, torch::TensorOptions().dtype(torch::kInt32).device(device));
  torch::Tensor unique_index = torch::empty(
      {N}, torch::TensorOptions().dtype(torch::kInt32).device(device));

  // Unique by hash key
  thrust::transform(policy, ids.data_ptr<int64_t>(),
                    ids.data_ptr<int64_t>() + N, ids_hash.data_ptr<int64_t>(),
                    HashKey<int64_t>{kSliceBits});

  uint64_t* keys_u_sort =
      reinterpret_cast<uint64_t*>(ids_hash.data_ptr<int64_t>());
  thrust::sort_by_key(policy, keys_u_sort, keys_u_sort + N,
                      sorted_indices.data_ptr<int32_t>());

  adjacent_difference_kernel<<<(N + 1023) / 1024, 1024, 0, stream>>>(
      ids_hash.data_ptr<int64_t>(), unique_index.data_ptr<int32_t>(), N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  thrust::inclusive_scan(policy, unique_index.data_ptr<int32_t>(),
                         unique_index.data_ptr<int32_t>() + N,
                         unique_index.data_ptr<int32_t>());

  int32_t last_element = unique_index[-1].item<int32_t>();
  int64_t unique_ids_size = last_element + 1;

  torch::Tensor results = torch::empty(
      {unique_ids_size},
      torch::TensorOptions().dtype(torch::kInt64).device(ids.device()));
  unique_by_key_kernel<<<(N + 1023) / 1024, 1024, 0, stream>>>(
      unique_index.data_ptr<int32_t>(), sorted_indices.data_ptr<int32_t>(),
      ids.data_ptr<int64_t>(), results.data_ptr<int64_t>(),
      reverse_indices.data_ptr<int32_t>(), N, unique_ids_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  torch::Tensor part_keys =
      torch::empty({unique_ids_size},
                   torch::TensorOptions().dtype(torch::kInt32).device(device));
  thrust::transform(policy, results.data_ptr<int64_t>(),
                    results.data_ptr<int64_t>() + unique_ids_size,
                    part_keys.data_ptr<int32_t>(),
                    HashSlice<int64_t>{static_cast<int>(num_parts)});

  // Reduce by key to count per partition
  auto segment_size = torch::zeros(
      num_parts, torch::TensorOptions().dtype(torch::kInt64).device(device));

  auto keys = torch::zeros(
      num_parts, torch::TensorOptions().dtype(torch::kInt64).device(device));

  auto vals = torch::zeros(
      num_parts, torch::TensorOptions().dtype(torch::kInt64).device(device));

  auto new_end =
      thrust::reduce_by_key(policy, part_keys.data_ptr<int32_t>(),
                            part_keys.data_ptr<int32_t>() + unique_ids_size,
                            thrust::constant_iterator<int64_t>(1),
                            keys.data_ptr<int64_t>(), vals.data_ptr<int64_t>());

  int num_unique = new_end.first - keys.data_ptr<int64_t>();

  thrust::scatter(policy, vals.data_ptr<int64_t>(),
                  vals.data_ptr<int64_t>() + num_unique,
                  keys.data_ptr<int64_t>(), segment_size.data_ptr<int64_t>());

  return std::make_tuple(results, segment_size, reverse_indices);
}
}  // namespace functional
}  // namespace recis