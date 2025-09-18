#include <torch/extension.h>

namespace recis {
namespace functional {
torch::Tensor uint64_mod(torch::Tensor inputs, torch::Scalar num);
void uint64_mod_cuda(torch::Tensor inputs, torch::Tensor output,
                     const int64_t input_size, torch::Scalar num);
}  // namespace functional
}  // namespace recis
