#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
namespace recis {
namespace functional {
template <typename scalar_t>
__global__ void adam_tf_apply_cuda_kernel(scalar_t* param, scalar_t* grad,
                                          scalar_t* avg, scalar_t* avg_sq,
                                          float step, float lr, float b1,
                                          float b2, float eps,
                                          const int64_t param_size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float b1_power = powf(b1, step);
  float b2_power = powf(b2, step);
  float alpha = -lr / (1. - b1_power) * sqrtf(1. - b2_power);
  if (i < param_size) {
    avg[i] = avg[i] * b1 + grad[i] * (1. - b1);
    avg_sq[i] = avg_sq[i] * b2 + (1. - b2) * grad[i] * grad[i];
    param[i] = param[i] + alpha * (avg[i] / (sqrtf(avg_sq[i]) + eps));
  }
}

void adam_tf_apply_cuda(torch::Tensor param, torch::Tensor grad,
                        torch::Tensor avg, torch::Tensor avg_sq, float step,
                        float lr, float beta1, float beta2, float eps,
                        int64_t param_size) {
  const int threads = 128;
  const int blocks = (param_size + threads - 1) / threads;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      param.scalar_type(), "adam_tf_apply_cuda", ([&] {
        adam_tf_apply_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            param.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(),
            avg.data_ptr<scalar_t>(), avg_sq.data_ptr<scalar_t>(), step, lr,
            beta1, beta2, eps, param_size);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
}
}  // namespace functional
}  // namespace recis
