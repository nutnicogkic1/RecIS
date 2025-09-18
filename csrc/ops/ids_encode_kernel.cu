#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <bitset>
#include <vector>

#include "cuda/cuda_param.cuh"
#include "cuda/element_wise_kernel.cuh"
#include "cuda/utils.cuh"
#include "ops/ids_encode.h"

namespace recis {
namespace functional {

struct EncodeFactory {
  __device__ int64_t operator()(const int64_t value, const int64_t offset) {
    return (value & _MASK) | (offset << (_MAX_BIT_SIZE - _MAX_ENCODE_SIZE));
  }
};

torch::Tensor ids_encode_cuda(std::vector<torch::Tensor> inputs,
                              torch::Tensor table_ids) {
  using namespace recis::cuda;
  int64_t N = inputs.size();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto options = inputs[0].options();

  std::vector<int64_t> sizes(N);
  CudaVecParam<int64_t*> inputs_ptrs(N, stream);
  CudaVecParam<int64_t*> outputs_ptrs(N, stream);
  CudaVecParam<int64_t> cumulative_sizes(N, stream);

  int64_t total_size = 0;

  for (int64_t i = 0; i < N; ++i) {
    inputs_ptrs[i] = inputs[i].data_ptr<int64_t>();
    sizes[i] = inputs[i].numel();
    if (i > 0) {
      cumulative_sizes[i] = cumulative_sizes[i - 1] + sizes[i - 1];
    } else {
      cumulative_sizes[i] = 0;
    }
    total_size += sizes[i];
  }

  torch::Tensor output = torch::empty({total_size}, inputs[0].options());
  for (int i = 0; i < N; i++) {
    outputs_ptrs[i] = output.data_ptr<int64_t>() + cumulative_sizes[i];
  }

  // without pack
  fused_element_wise_launcher<int64_t, int64_t, int64_t, EncodeFactory>(
      const_cast<const int64_t**>(inputs_ptrs.data()),
      table_ids.data_ptr<int64_t>(), outputs_ptrs.data(), sizes.data(), N,
      EncodeFactory(), false, stream);

  return output;
}

}  // namespace functional
}  // namespace recis
