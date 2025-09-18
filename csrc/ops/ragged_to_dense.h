#include <torch/extension.h>

namespace recis {
namespace functional {
torch::Tensor ragged_to_dense(torch::Tensor values,
                              const std::vector<torch::Tensor> &offsets,
                              torch::Scalar default_value);

void ragged_to_dense_cuda(torch::Tensor values,
                          const std::vector<torch::Tensor> &offsets,
                          torch::Tensor output, torch::Scalar default_value);

}  // namespace functional
}  // namespace recis
