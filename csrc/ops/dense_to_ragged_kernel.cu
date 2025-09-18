#include <ATen/cuda/ThrustAllocator.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <torch/extension.h>

namespace recis {
namespace functional {
template <typename scalar_t>
__global__ void compute_valid_lengths_kernel(const scalar_t* input,
                                             int* lengths, int rows, int cols,
                                             scalar_t x) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;

  int last_index = -1;
  for (int col = 0; col < cols; ++col) {
    if (input[row * cols + col] != x) {
      last_index = col;
    }
  }

  lengths[row] = last_index + 1;
}

template <typename scalar_t>
__global__ void gather_values_kernel(const scalar_t* input, scalar_t* values,
                                     const int* lengths, int* offsets, int rows,
                                     int cols, scalar_t x) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;

  int start_idx = offsets[row];
  int len = lengths[row];

  for (int i = 0; i < len; ++i) {
    values[start_idx + i] = input[row * cols + i];
  }
}

std::tuple<torch::Tensor, torch::Tensor> dense_to_ragged_cuda(
    const torch::Tensor& data, const torch::Tensor& invalid_value) {
  int rows = data.size(0);
  int cols = data.size(1);
  auto device = data.device();
  torch::Tensor offsets = torch::empty(
      {rows + 1}, torch::TensorOptions().dtype(torch::kInt32).device(device));

  torch::Tensor values;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Int, at::ScalarType::Long, data.scalar_type(),
      "ragged_to_dense_cuda", ([&] {
        scalar_t scalar_default_value = invalid_value.data_ptr<scalar_t>()[0];
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        at::cuda::ThrustAllocator allocator;
        auto policy = thrust::cuda::par_nosync(allocator).on(stream);

        thrust::device_vector<int> lengths(rows);

        int threads = 256;
        int blocks = (rows + threads - 1) / threads;
        compute_valid_lengths_kernel<<<blocks, threads, 0, stream>>>(
            data.data_ptr<scalar_t>(), lengths.data().get(), rows, cols,
            scalar_default_value);

        int total_length =
            thrust::reduce(policy, lengths.begin(), lengths.end());

        values = torch::empty({total_length}, data.options());
        thrust::inclusive_scan(
            policy, lengths.begin(), lengths.end(),
            thrust::device_ptr<int>(offsets.data_ptr<int>()).get() + 1);

        gather_values_kernel<<<blocks, threads, 0, stream>>>(
            data.data_ptr<scalar_t>(), values.data_ptr<scalar_t>(),
            lengths.data().get(), offsets.data_ptr<int>(), rows, cols,
            scalar_default_value);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
  return std::tuple(values, offsets);
}
}  // namespace functional
}  // namespace recis
