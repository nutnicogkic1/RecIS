#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
namespace recis {
namespace functional {
__global__ void uint64_mod_cuda_kernel(const int64_t* inputs, int64_t* output,
                                       const int64_t input_size,
                                       const int64_t num) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < input_size) {
    uint64_t out = static_cast<uint64_t>(inputs[i]) % num;
    output[i] = static_cast<int64_t>(out);
  }
}

void uint64_mod_cuda(torch::Tensor inputs, torch::Tensor output,
                     const int64_t input_size, torch::Scalar num) {
  const int threads = 128;
  const int blocks = (input_size + threads - 1) / threads;

  uint64_mod_cuda_kernel<<<blocks, threads>>>(inputs.data_ptr<int64_t>(),
                                              output.data_ptr<int64_t>(),
                                              input_size, num.to<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace functional
}  // namespace recis
