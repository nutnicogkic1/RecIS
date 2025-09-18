#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "cuda/cuda_param.cuh"
#include "cuda/element_wise_kernel.cuh"
#include "cuda/utils.cuh"
#include "cuda_runtime.h"
#include "ops/multi_hash.h"
namespace recis {
namespace functional {
struct YxMultiHashData {
  int64_t* muls;
  int64_t* primes;
  int64_t* bucket_num;
  int index;

  YxMultiHashData(int64_t* muls, int64_t* primes, int64_t* bucket_num,
                  int index)
      : muls(muls), primes(primes), bucket_num(bucket_num), index(index) {}
  YxMultiHashData()
      : muls(nullptr), primes(nullptr), bucket_num(nullptr), index(0) {}
};

struct YxMultiHashFactory {
  __device__ int64_t operator()(const int64_t input_value,
                                const YxMultiHashData& data) {
    int j = data.index;
    return ((((input_value * data.muls[j]) % data.primes[j]) + data.primes[j]) %
            data.primes[j]) %
           data.bucket_num[j];
  }
};

void fused_multi_hash_cuda(std::vector<torch::Tensor>& inputs,
                           std::vector<torch::Tensor>& outputs,
                           std::vector<torch::Tensor>& muls,
                           std::vector<torch::Tensor>& primes,
                           std::vector<torch::Tensor>& bucket_nums) {
  using namespace recis::cuda;
  int64_t N = inputs.size();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  std::vector<int64_t> multi_hash_nums(N);
  int64_t multi_hash_num_count = 0;
  for (int64_t i = 0; i < N; ++i) {
    multi_hash_nums[i] = muls[i].numel();
    multi_hash_num_count += multi_hash_nums[i];
  }

  std::vector<int64_t> sizes(multi_hash_num_count);
  CudaVecParam<int64_t*> inputs_ptrs(multi_hash_num_count, stream);
  CudaVecParam<int64_t*> outputs_ptrs(multi_hash_num_count, stream);
  CudaVecParam<YxMultiHashData> hash_datas(multi_hash_num_count, stream);

  int64_t output_idx = 0;
  for (int64_t i = 0; i < N; ++i) {
    int64_t multi_hash_num = multi_hash_nums[i];
    for (int j = 0; j < multi_hash_num; j++) {
      inputs_ptrs[output_idx + j] = inputs[i].data_ptr<int64_t>();
      hash_datas[output_idx + j] = YxMultiHashData(
          muls[i].data_ptr<int64_t>(), primes[i].data_ptr<int64_t>(),
          bucket_nums[i].data_ptr<int64_t>(), j);
      outputs_ptrs[output_idx + j] =
          outputs[output_idx + j].data_ptr<int64_t>();
      sizes[output_idx + j] = inputs[i].numel();
    }
    output_idx += multi_hash_num;
  }

  fused_element_wise_launcher<int64_t, YxMultiHashData, int64_t,
                              YxMultiHashFactory>(
      const_cast<const int64_t**>(inputs_ptrs.data()), hash_datas.data(),
      outputs_ptrs.data(), sizes.data(), multi_hash_num_count,
      YxMultiHashFactory(), false, stream);
}

}  // namespace functional
}  // namespace recis
