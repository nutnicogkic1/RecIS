#include <torch/extension.h>

namespace recis {
namespace functional {
torch::Tensor ragged_to_sparse(torch::Tensor values,
                               std::vector<torch::Tensor> offset_splits);
}  // namespace functional
}  // namespace recis
