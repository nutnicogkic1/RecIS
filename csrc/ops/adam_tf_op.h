#include <torch/extension.h>

namespace recis {
namespace functional {
void adam_tf_apply(torch::Tensor param, torch::Tensor grad, torch::Tensor avg,
                   torch::Tensor avg_sq, torch::Scalar step, torch::Scalar lr,
                   torch::Scalar beta1, torch::Scalar beta2, torch::Scalar eps);
void adam_tf_apply_cuda(torch::Tensor param, torch::Tensor grad,
                        torch::Tensor avg, torch::Tensor avg_sq, float step,
                        float lr, float beta1, float beta2, float eps,
                        int64_t param_size);
}  // namespace functional
}  // namespace recis
