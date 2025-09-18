#include <ATen/cuda/CUDAContext.h>

#include "cuda/cuda_param.cuh"
#include "cuda/element_wise_kernel.cuh"
#include "cuda/utils.cuh"
#include "cuda_runtime.h"
#include "ops/fused_uint64_mod.h"

namespace recis {
namespace functional {

struct Uint64ModFactory {
  __device__ int64_t operator()(const int64_t value, const int64_t mod) {
    return static_cast<int64_t>(static_cast<uint64_t>(value) % mod);
  }
};

void fused_uint64_mod_cuda(std::vector<torch::Tensor>& inputs,
                           std::vector<torch::Tensor>& outputs,
                           torch::Tensor mod_vec) {
  using namespace recis::cuda;
  int64_t N = inputs.size();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  std::vector<int64_t> sizes(N);
  CudaVecParam<int64_t*> inputs_ptrs(N, stream);
  CudaVecParam<int64_t*> outputs_ptrs(N, stream);

  for (int64_t i = 0; i < N; ++i) {
    sizes[i] = inputs[i].numel();
    inputs_ptrs[i] = inputs[i].data_ptr<int64_t>();
    outputs_ptrs[i] = outputs[i].data_ptr<int64_t>();
  }
  // with pack
  fused_element_wise_launcher<int64_t, int64_t, int64_t, Uint64ModFactory>(
      const_cast<const int64_t**>(inputs_ptrs.data()),
      mod_vec.data_ptr<int64_t>(), outputs_ptrs.data(), sizes.data(), N,
      Uint64ModFactory(), true, stream);
}
}  // namespace functional
}  // namespace recis
