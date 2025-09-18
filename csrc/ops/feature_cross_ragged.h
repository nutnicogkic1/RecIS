#include <ATen/ATen.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "ATen/Parallel.h"

namespace recis {
namespace functional {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> feature_cross_ragged(
    const torch::Tensor &x_value, const torch::Tensor &x_offsets,
    const torch::Tensor &x_weight, const torch::Tensor &y_value,
    const torch::Tensor &y_offsets, const torch::Tensor &y_weight);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
feature_cross_ragged_cuda(torch::Tensor x_values, torch::Tensor x_offsets,
                          torch::Tensor x_weight, torch::Tensor y_values,
                          torch::Tensor y_offsets, torch::Tensor y_weight);

}  // namespace functional
}  // namespace recis
