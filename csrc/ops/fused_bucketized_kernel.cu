#include <ATen/cuda/CUDAContext.h>

#include "cuda/cuda_param.cuh"
#include "cuda/element_wise_kernel.cuh"
#include "cuda/utils.cuh"
#include "cuda_runtime.h"
#include "ops/fused_bucketized.h"

namespace recis {
namespace functional {
struct BucketizeData {
  float* boundaries;
  int len;
  BucketizeData() : boundaries(nullptr), len(0) {}
  BucketizeData(float* boundaries, int len)
      : boundaries(boundaries), len(len) {}
};

struct BucketizeFactory {
  __device__ int operator()(const float value, const BucketizeData& data) {
    int bucket = 0;
    int count = data.len;
    auto boundaries = data.boundaries;
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
    return bucket;
  }
};

void fused_bucketized_cuda(std::vector<torch::Tensor>& inputs,
                           std::vector<torch::Tensor>& outputs,
                           std::vector<torch::Tensor>& boundaries) {
  using namespace recis::cuda;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t N = inputs.size();
  std::vector<int64_t> sizes(N);
  CudaVecParam<float*> inputs_ptrs(N, stream);
  CudaVecParam<int64_t*> outputs_ptrs(N, stream);
  CudaVecParam<BucketizeData> bucketize_datas(N, stream);
  for (int64_t i = 0; i < N; ++i) {
    sizes[i] = inputs[i].numel();
    inputs_ptrs[i] = inputs[i].data_ptr<float>();
    outputs_ptrs[i] = outputs[i].data_ptr<int64_t>();
    bucketize_datas[i] =
        BucketizeData(boundaries[i].data_ptr<float>(), boundaries[i].numel());
  }

  fused_element_wise_launcher<float, BucketizeData, int64_t, BucketizeFactory>(
      const_cast<const float**>(inputs_ptrs.data()), bucketize_datas.data(),
      outputs_ptrs.data(), sizes.data(), N, BucketizeFactory(), false, stream);
}
}  // namespace functional
}  // namespace recis
