#include <pybind11/pytypes.h>
#include <torch/extension.h>

#include <string>
#include <vector>

namespace recis {
namespace functional {
std::vector<torch::Tensor> fused_hash(std::vector<torch::Tensor> inputs,
                                      std::vector<torch::Tensor> input_offsets,
                                      const std::string& hash_type);

void fused_hash_cuda(const std::vector<torch::Tensor>& inputs,
                     const std::vector<torch::Tensor>& input_offsets,
                     const std::vector<torch::Tensor>& outputs,
                     const std::string& hash_type);
}  // namespace functional
}  // namespace recis
