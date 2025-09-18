#include <torch/extension.h>

#include <vector>
namespace recis {
namespace functional {
std::vector<torch::Tensor> fused_uint64_mod(std::vector<torch::Tensor> inputs,
                                            torch::Tensor mod_vec);
void fused_uint64_mod_cpu(std::vector<torch::Tensor> &inputs,
                          std::vector<torch::Tensor> &outputs,
                          torch::Tensor mod_vec);
void fused_uint64_mod_cuda(std::vector<torch::Tensor> &inputs,
                           std::vector<torch::Tensor> &outputs,
                           torch::Tensor mod_vec);
}  // namespace functional
}  // namespace recis
