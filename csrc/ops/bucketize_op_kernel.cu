#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
namespace recis {
namespace functional {
template <typename scalar_t>
__global__ void bucketize_cuda_kernel(const scalar_t* __restrict__ values,
                                      const scalar_t* __restrict__ boundaries,
                                      int64_t* __restrict__ output,
                                      const int64_t input_size,
                                      const int boundary_len) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < input_size) {
    auto value = values[i];
    int bucket = 0;
    int count = boundary_len;
    while (count > 0) {
      int left = bucket;
      int step = count / 2;
      left += step;
      if (!(value < boundaries[left])) {
        bucket = ++left;
        count -= step + 1;
      } else {
        count = step;
      }
    }
    output[i] = bucket;
  }
}

void bucketize_cuda_op(torch::Tensor values, torch::Tensor boundaries,
                       torch::Tensor output, int64_t input_size,
                       int boundary_len) {
  const int threads = 128;
  const int blocks = (input_size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(
      values.scalar_type(), "bucketize_cuda", ([&] {
        bucketize_cuda_kernel<scalar_t><<<blocks, threads>>>(
            values.data_ptr<scalar_t>(), boundaries.data_ptr<scalar_t>(),
            output.data_ptr<int64_t>(), input_size, boundary_len);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
}
}  // namespace functional
}  // namespace recis
