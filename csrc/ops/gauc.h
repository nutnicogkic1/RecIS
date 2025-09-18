#pragma once
#include <tuple>

#include "ATen/core/TensorBody.h"
#include "torch/extension.h"
#include "torch/serialize/input-archive.h"
namespace recis {
namespace functional {
std::tuple<torch::Tensor, torch::Tensor> GaucCalc(torch::Tensor labels,
                                                  torch::Tensor predictions,
                                                  torch::Tensor indicators);
}
}  // namespace recis