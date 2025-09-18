#include <torch/extension.h>

#include <vector>
namespace recis {
namespace functional {
std::vector<torch::Tensor> fused_bucketized(
    std::vector<torch::Tensor> inputs, std::vector<torch::Tensor> boundaries);
void fused_bucketized_cpu(std::vector<torch::Tensor> &inputs,
                          std::vector<torch::Tensor> &outputs,
                          std::vector<torch::Tensor> &boundaries);
void fused_bucketized_cuda(std::vector<torch::Tensor> &inputs,
                           std::vector<torch::Tensor> &outputs,
                           std::vector<torch::Tensor> &boundaries);
}  // namespace functional
}  // namespace recis
