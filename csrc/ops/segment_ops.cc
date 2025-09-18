#include "segment_ops.h"

namespace recis {
namespace functional {

torch::Tensor segment_sum(torch::Tensor data,
                          c10::optional<torch::Tensor> weight,
                          torch::Tensor indices, torch::Tensor segment_ids,
                          torch::Scalar num_segments) {
  TORCH_CHECK(indices.dim() == 1);
  TORCH_CHECK(data.dim() == 2);
  TORCH_CHECK(segment_ids.dim() == 1);
  bool use_weight = weight.has_value();
  std::vector<int64_t> in_shape = data.sizes().vec();
  int64_t num_segment = num_segments.toInt();

  auto options = data.options();
  torch::Tensor weight_data;
  if (!use_weight) {
    weight_data = torch::ones({1}, options);
  } else {
    weight_data = weight.value();
  }
  in_shape[0] = num_segment;
  torch::Tensor output = torch::empty(in_shape, data.options());
  segment_sum_cuda(data, weight_data, use_weight, indices, segment_ids,
                   num_segment, output);
  return output;
}

torch::Tensor segment_mean(torch::Tensor data,
                           c10::optional<torch::Tensor> weight,
                           torch::Tensor segment_ids,
                           torch::Scalar num_segments) {
  TORCH_CHECK(segment_ids.dim() == 1);
  bool use_weight = weight.has_value();
  int64_t num_segment = num_segments.toInt();
  auto options = data.options();
  torch::Tensor weight_data;
  if (!use_weight) {
    weight_data = torch::ones({1}, options);
  } else {
    weight_data = weight.value();
  }
  TORCH_CHECK(weight_data.dim() == 1);
  torch::Tensor weight_sum = torch::empty({num_segment}, weight_data.options());
  torch::Tensor weight_norm =
      torch::empty({segment_ids.numel()}, weight_data.options());
  segment_mean_cuda(weight_data, use_weight, weight_sum, weight_norm,
                    segment_ids, num_segment);
  return weight_norm;
}
}  // namespace functional
}  // namespace recis
