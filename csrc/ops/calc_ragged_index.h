#pragma once
#include <torch/extension.h>

#include <tuple>

#include "ATen/core/TensorBody.h"
namespace recis {
namespace functional {
/**
 * @brief Calculate ragged index
 * @return std::tuple<at::Tensor, at::Tensor>
 * calculate value index and offset of other topk features according
 * to the cutoff info of topk key feature and topk index.
 */
std::tuple<at::Tensor, at::Tensor> calc_ragged_index(
    at::Tensor drop_num, at::Tensor pad_num, at::Tensor drop_side,
    at::Tensor pad_side, at::Tensor offset, at::Tensor topk_index,
    at::Tensor indicator);
}  // namespace functional
}  // namespace recis