#include "adam_tf_op.h"

#include <cmath>

namespace recis {
namespace functional {

template <typename scalar_t>
void adam_tf_apply_cpu_kernel(scalar_t* param, scalar_t* grad, scalar_t* avg,
                              scalar_t* avg_sq, float step, float lr, float b1,
                              float b2, float eps, const int64_t param_size) {
  float b1_power = std::pow(b1, step);
  float b2_power = std::pow(b2, step);
  float alpha = -lr / (1. - b1_power) * std::sqrt(1. - b2_power);
  for (int64_t i = 0; i < param_size; ++i) {
    avg[i] = avg[i] * b1 + grad[i] * (1. - b1);
    avg_sq[i] = avg_sq[i] * b2 + (1. - b2) * grad[i] * grad[i];
    param[i] = param[i] + alpha * (avg[i] / (std::sqrt(avg_sq[i]) + eps));
  }
}

void adam_tf_apply(torch::Tensor param, torch::Tensor grad, torch::Tensor avg,
                   torch::Tensor avg_sq, torch::Scalar step, torch::Scalar lr,
                   torch::Scalar beta1, torch::Scalar beta2,
                   torch::Scalar eps) {
  auto param_size = param.numel();
  float step_v = step.to<float>();
  float lr_v = lr.to<float>();
  float b1_v = beta1.to<float>();
  float b2_v = beta2.to<float>();
  float eps_v = eps.to<float>();
  if (param_size == 0) {
    return;
  }

  if (param.device().is_cuda()) {
    adam_tf_apply_cuda(param, grad, avg, avg_sq, step_v, lr_v, b1_v, b2_v,
                       eps_v, param_size);
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        param.scalar_type(), "adam_tf_apply_cpu", ([&] {
          adam_tf_apply_cpu_kernel<scalar_t>(
              param.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(),
              avg.data_ptr<scalar_t>(), avg_sq.data_ptr<scalar_t>(), step_v,
              lr_v, b1_v, b2_v, eps_v, param_size);
        }));
  }
}
}  // namespace functional
}  // namespace recis
