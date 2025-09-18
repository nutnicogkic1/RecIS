#include "multi_hash.h"
namespace recis {
namespace functional {
void fused_multi_hash_cpu(std::vector<torch::Tensor>& inputs,
                          std::vector<torch::Tensor>& outputs,
                          std::vector<torch::Tensor>& muls,
                          std::vector<torch::Tensor>& primes,
                          std::vector<torch::Tensor>& bucket_nums) {
  int64_t output_idx = 0;
  for (size_t tensor_idx = 0; tensor_idx < inputs.size(); ++tensor_idx) {
    auto input_size = inputs[tensor_idx].numel();
    auto input_data = inputs[tensor_idx].data_ptr<int64_t>();

    torch::parallel_for(0, input_size, 1, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        auto multi_hash_num = muls[tensor_idx].numel();
        for (int64_t j = 0; j < multi_hash_num; ++j) {
          auto bucket_num_data = bucket_nums[tensor_idx].data_ptr<int64_t>();
          auto muls_data = muls[tensor_idx].data_ptr<int64_t>();
          auto primes_data = primes[tensor_idx].data_ptr<int64_t>();
          int64_t out = ((((input_data[i] * muls_data[j]) % primes_data[j]) +
                          primes_data[j]) %
                         primes_data[j]) %
                        bucket_num_data[j];
          outputs[output_idx + j].data_ptr<int64_t>()[i] = out;
        }
      }
    });
    output_idx += muls[tensor_idx].numel();
  }
}

std::vector<torch::Tensor> fused_multi_hash(
    std::vector<torch::Tensor> inputs, std::vector<torch::Tensor> muls,
    std::vector<torch::Tensor> primes, std::vector<torch::Tensor> bucket_num) {
  int64_t num_tensors = inputs.size();

  std::vector<torch::Tensor> outputs;
  int64_t multi_hash_num_count = 0;
  for (int64_t i = 0; i < num_tensors; ++i) {
    multi_hash_num_count += muls[i].numel();
  }
  outputs.reserve(multi_hash_num_count);

  for (int64_t i = 0; i < num_tensors; ++i) {
    int64_t multi_hash_num = muls[i].numel();
    for (int64_t j = 0; j < multi_hash_num; ++j) {
      auto output = torch::empty_like(inputs[i]);
      outputs.push_back(output);
    }
  }

  if (inputs[0].device().is_cuda()) {
    fused_multi_hash_cuda(inputs, outputs, muls, primes, bucket_num);
  } else {
    fused_multi_hash_cpu(inputs, outputs, muls, primes, bucket_num);
  }

  return outputs;
}
}  // namespace functional
}  // namespace recis
