#pragma once
#include <torch/extension.h>

#include <vector>

namespace recis {
namespace functional {

std::vector<torch::Tensor> fused_multi_hash(
    std::vector<torch::Tensor> inputs, std::vector<torch::Tensor> muls,
    std::vector<torch::Tensor> primes, std::vector<torch::Tensor> bucket_num);
void fused_multi_hash_cuda(std::vector<torch::Tensor>& inputs,
                           std::vector<torch::Tensor>& outputs,
                           std::vector<torch::Tensor>& muls,
                           std::vector<torch::Tensor>& primes,
                           std::vector<torch::Tensor>& bucket_nums);

}  // namespace functional
}  // namespace recis
