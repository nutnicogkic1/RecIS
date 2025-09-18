#include <torch/extension.h>

namespace recis {
namespace functional {
torch::Tensor bucketize_op(torch::Tensor values, torch::Tensor boundaries);
void bucketize_cuda_op(torch::Tensor values, torch::Tensor boundaries,
                       torch::Tensor output, const int64_t input_size,
                       const int boundary_len);
}  // namespace functional
}  // namespace recis
