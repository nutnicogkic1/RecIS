#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>

#include "cuda/cuda_param.cuh"

namespace recis {
namespace functional {
__global__ void boolean_mask_cuda_kernel(int64_t* output, const bool* mask,
                                         const int64_t* select_index,
                                         const int64_t* input,
                                         const int64_t output_size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < output_size) {
    output[i] =
        mask[i] * input[select_index[i] - 1] + (1 - mask[i]) * output[i];
  }
}

void boolean_mask_cuda_op(torch::Tensor output, torch::Tensor mask,
                          torch::Tensor select_index, torch::Tensor input,
                          const int64_t output_size) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int threads = 128;
  const int blocks = (output_size + threads - 1) / threads;
  boolean_mask_cuda_kernel<<<blocks, threads, 0, stream>>>(
      output.data_ptr<int64_t>(), mask.data_ptr<bool>(),
      select_index.data_ptr<int64_t>(), input.data_ptr<int64_t>(), output_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void __device__ generate_ids_kernel(int64_t& out, const int64_t& cur_count,
                                    const int64_t& free_index,
                                    int64_t** free_blocks,
                                    const int64_t& free_block_size) {
  bool from_free = (free_index >= 0);
  if (from_free) {
    int64_t block_index = (free_index / free_block_size);
    int64_t row_index = (free_index % free_block_size);
    int64_t free_id = free_blocks[block_index][row_index];
    out = free_id;
  } else {
    out = cur_count - (free_index + 1);
  }
}

__global__ void generate_ids_cuda_kernel(int64_t* output, const int64_t gen_num,
                                         int64_t** free_blocks,
                                         const int64_t free_count,
                                         const int64_t cur_count,
                                         const int64_t free_block_size) {
  const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;  // pack 2
  if (idx >= gen_num) return;
  int64_t free_index;
  if (idx + 2 <= gen_num) {
    int64_t pack_out1, pack_out2;
    free_index = free_count - gen_num + idx;
    generate_ids_kernel(pack_out1, cur_count, free_index, free_blocks,
                        free_block_size);
    free_index++;
    generate_ids_kernel(pack_out2, cur_count, free_index, free_blocks,
                        free_block_size);
    *(longlong2*)(output + idx) = make_longlong2(pack_out1, pack_out2);
  } else {
    free_index = free_count - gen_num + idx;
    generate_ids_kernel(output[idx], cur_count, free_index, free_blocks,
                        free_block_size);
  }
}

void generate_ids_cuda_op(torch::Tensor output, const int64_t gen_num,
                          std::vector<torch::Tensor>& free_blocks,
                          const int64_t free_count, const int64_t cur_count,
                          const int64_t free_block_size) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t threads = 128;
  const int64_t blocks = ((gen_num + 1) / 2 + threads - 1) / threads;

  auto block_num = free_blocks.size();
  recis::cuda::CudaVecParam<int64_t*> free_blocks_ptrs(block_num, stream);
  for (auto i = 0; i < block_num; ++i) {
    free_blocks_ptrs[i] = free_blocks[i].data_ptr<int64_t>();
  }

  generate_ids_cuda_kernel<<<blocks, threads, 0, stream>>>(
      output.data_ptr<int64_t>(), gen_num, (int64_t**)(free_blocks_ptrs.data()),
      free_count, cur_count, free_block_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

__global__ void free_ids_cuda_kernel(int64_t* free_ids, const int64_t free_num,
                                     int64_t** free_blocks,
                                     const int64_t free_count,
                                     const int64_t free_block_size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < free_num) {
    int64_t free_index = free_count + i;
    int64_t block_index = free_index / free_block_size;
    int64_t row_index = free_index % free_block_size;
    free_blocks[block_index][row_index] = free_ids[i];
  }
}

void free_ids_cuda_op(torch::Tensor free_ids, const int64_t free_num,
                      std::vector<torch::Tensor>& free_blocks,
                      const int64_t free_count, const int64_t free_block_size) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t threads = 128;
  const int64_t blocks = (free_num + threads - 1) / threads;

  auto block_num = free_blocks.size();
  recis::cuda::CudaVecParam<int64_t*> free_blocks_ptrs(block_num, stream);
  for (auto i = 0; i < block_num; ++i) {
    free_blocks_ptrs[i] = free_blocks[i].data_ptr<int64_t>();
  }
  free_ids_cuda_kernel<<<blocks, threads, 0, stream>>>(
      free_ids.data_ptr<int64_t>(), free_num,
      (int64_t**)(free_blocks_ptrs.data()), free_count, free_block_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

__global__ void mask_key_index_cuda_kernel(
    const int64_t* in_keys, const bool* mask, const int64_t* in_out_index,
    int64_t* out_keys, int64_t* out_index, int64_t in_size) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= in_size || !(mask[i])) return;
  out_keys[in_out_index[i] - 1] = in_keys[i];
  out_index[in_out_index[i] - 1] = i;
}

void mask_key_index_cuda_op(torch::Tensor in_keys, torch::Tensor mask,
                            torch::Tensor in_out_index, torch::Tensor out_keys,
                            torch::Tensor out_index, int64_t in_size) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t threads = 128;
  const int64_t blocks = (in_size + threads - 1) / threads;
  mask_key_index_cuda_kernel<<<blocks, threads, 0, stream>>>(
      in_keys.data_ptr<int64_t>(), mask.data_ptr<bool>(),
      in_out_index.data_ptr<int64_t>(), out_keys.data_ptr<int64_t>(),
      out_index.data_ptr<int64_t>(), in_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void scatter_ids_with_mask_cuda_kernel(int64_t* out_ids,
                                                  const int64_t* in_ids,
                                                  const int64_t* mask_index,
                                                  const int64_t in_size) {
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;  // packe 2
  if (i >= in_size) return;
  if (i + 2 <= in_size) {
    longlong2 pack_in_id = *(longlong2*)(in_ids + i);
    longlong2 pack_mask_index = *(longlong2*)(mask_index + i);
    out_ids[(&pack_mask_index.x)[0]] = (&pack_in_id.x)[0];
    out_ids[(&pack_mask_index.x)[1]] = (&pack_in_id.x)[1];
  } else {
    out_ids[mask_index[i]] = in_ids[i];
  }
}

void scatter_ids_with_mask_cuda_op(torch::Tensor out_ids, torch::Tensor in_ids,
                                   torch::Tensor mask_index,
                                   const int64_t in_size) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t threads = 128;
  const int64_t blocks = ((in_size + 1) / 2 + threads - 1) / threads;
  scatter_ids_with_mask_cuda_kernel<<<blocks, threads, 0, stream>>>(
      out_ids.data_ptr<int64_t>(), in_ids.data_ptr<int64_t>(),
      mask_index.data_ptr<int64_t>(), in_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace functional
}  // namespace recis
