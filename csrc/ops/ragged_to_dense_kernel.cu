#include <ATen/cuda/CUDAContext.h>
#include <thrust/device_vector.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <numeric>

#include "c10/cuda/CUDAStream.h"
#include "ragged_common.cuh"

namespace recis {
namespace functional {

// Derived from FBGEMM: ragged_elementwise_to_dense_kernel
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
// Modifications:
//   Introduced shared memory
//   removed launch bounds
//   simplified logic for multi-dimensional inner values.
template <int NUM_RAGGED_DIM, typename index_t, typename scalar_t>
__global__ void ragged_elementwise_to_dense_kernel(
    torch::PackedTensorAccessor64<scalar_t, 1, at::RestrictPtrTraits> values,
    StackArray<index_t*> offsets,
    torch::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> output,
    StackArray<int64_t> ragged_dims, const scalar_t padding_value,
    bool use_shared_mem, StackArray<index_t> offsets_size) {
  const index_t outer_dense_size = output.size(0);
  const index_t ragged_folded_size = output.size(1);

  // for recis, the inner dense of dense value is 1
  const auto outer_begin =
      blockIdx.x * blockDim.x + threadIdx.x;  // each thread deal with one val
  const auto outer_stride = gridDim.x * blockDim.x;
  const auto tid = threadIdx.x;

  extern __shared__ unsigned char shm[];
  index_t offoffsets[NUM_RAGGED_DIM];
  if (use_shared_mem) {
    index_t* _sm_offsets = reinterpret_cast<index_t*>(shm);
    for (int d = 0; d < NUM_RAGGED_DIM; ++d) {
      offoffsets[d] = d == 0 ? 0 : offoffsets[d - 1] + offsets_size.vals[d - 1];
      for (index_t i = tid; i < offsets_size.vals[d]; i += blockDim.x) {
        _sm_offsets[i + offoffsets[d]] = offsets.vals[d][i];
      }
    }
    __syncthreads();
    for (int d = 0; d < NUM_RAGGED_DIM; ++d) {
      offsets.vals[d] = _sm_offsets + offoffsets[d];
    }
  }

  for (index_t outer = outer_begin;
       outer < outer_dense_size * ragged_folded_size; outer += outer_stride) {
    const index_t oidx = outer / ragged_folded_size;
    const index_t jidx = outer % ragged_folded_size;
    index_t offset = oidx;
    const bool is_zero = walk_down_tensor_storage_tree_<NUM_RAGGED_DIM>(
        offset, jidx, ragged_dims, offsets);

    if (is_zero) {
      output[oidx][jidx] = padding_value;
    } else {
      output[oidx][jidx] = __ldg(&values[offset]);
    }
  }
}

// Note: ragged of recis with no inner dense
void ragged_to_dense_cuda(torch::Tensor values,
                          const std::vector<torch::Tensor>& offsets,
                          torch::Tensor output, torch::Scalar default_value) {
  TORCH_CHECK(
      all_cuda(offsets) && all_cuda({values}),
      "ragged_to_dense: each dim offsets and values should be placed on gpu");

  auto stream = c10::cuda::getCurrentCUDAStream();
  // dim of offsets
  StackArray<int64_t> ragged_dims_tensor;
  const int num_ragged_dim = output.dim() - 1;
  TORCH_CHECK(num_ragged_dim <= kStackArrayMaxDims,
              "ragged_to_dense: output.dim should < kStackArrayMaxDims");
  TORCH_CHECK(offsets.size() == num_ragged_dim,
              "ragged_to_dense: offsets size must equal num_ragged_dim");
  ragged_dims_tensor.ndim = num_ragged_dim;
  // record the ragged dim each dim except the batch size
  std::memcpy(&(ragged_dims_tensor.vals[0]), output.sizes().data() + 1,
              num_ragged_dim * sizeof(int64_t));

#define INVOKE_KERNEL_WITH_DIM(NUM_RAGGED_DIM)                                 \
  {                                                                            \
    ragged_elementwise_to_dense_kernel<NUM_RAGGED_DIM, index_t, scalar_t>      \
        <<<blocks, threads, shared_mem_size, stream>>>(                        \
            values.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), \
            offset_ptrs,                                                       \
            output_view                                                        \
                .packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),   \
            ragged_dims_tensor, default_value.to<scalar_t>(), use_shared_mem,  \
            offsets_size);                                                     \
  }

  AT_DISPATCH_INDEX_TYPES(
      offsets.front().scalar_type(), "ragged_to_dense_cuda_op_0", ([&] {
        AT_DISPATCH_ALL_TYPES(
            values.scalar_type(), "ragged_to_dense_cuda_op_1", ([&] {
              // make each dim offset contig
              StackArray<index_t*> offset_ptrs;
              std::vector<torch::Tensor> offsets_contig(num_ragged_dim);
              for (int d = 0; d < num_ragged_dim; ++d) {
                offsets_contig[d] = offsets[d].contiguous();
                offset_ptrs.vals[d] = offsets_contig[d].data_ptr<index_t>();
              }
              // view to [bs, flatten]
              int batch_size = output.size(0);
              torch::Tensor output_view = output.view({batch_size, -1});

              // prepare for offsets on the shared mem
              StackArray<index_t> offsets_size;
              offsets_size.ndim = offsets.size();
              for (int d = 0; d < offsets.size(); ++d) {
                offsets_size.vals[d] = offsets[d].numel();
              }
              const index_t offset_ele = std::accumulate(
                  offsets_size.vals, offsets_size.vals + offsets_size.ndim, 0);
              size_t shared_mem_per_block =
                  at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
              size_t shared_mem_size = sizeof(index_t) * offset_ele;
              bool use_shared_mem = shared_mem_size < shared_mem_per_block;
              if (!use_shared_mem) {
                shared_mem_size = 0;
              }
              const auto blocks =
                  dim3((output.numel() + MAX_THREADS_PER_BLOCK - 1) /
                       MAX_THREADS_PER_BLOCK);
              const auto threads = dim3(MAX_THREADS_PER_BLOCK);

              RAGGED_TENSOR_DISPATCH_DIMS();
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }));
      }));

#undef INVOKE_KERNEL_WITH_DIM
}

}  // namespace functional
}  // namespace recis
