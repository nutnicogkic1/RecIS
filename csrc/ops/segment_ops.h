#include <torch/extension.h>

namespace recis {
namespace functional {
torch::Tensor segment_sum(torch::Tensor data,
                          c10::optional<torch::Tensor> weight,
                          torch::Tensor indices, torch::Tensor segment_ids,
                          torch::Scalar num_segments);

torch::Tensor segment_mean(torch::Tensor data,
                           c10::optional<torch::Tensor> weight,
                           torch::Tensor segment_ids,
                           torch::Scalar num_segments);

void segment_sum_cuda(torch::Tensor data, torch::Tensor weight, bool use_weight,
                      torch::Tensor indices, torch::Tensor segment_ids,
                      const int64_t num_segments, torch::Tensor& out);

bool segment_mean_cuda(torch::Tensor weight, bool use_weight,
                       torch::Tensor weight_sum, torch::Tensor& weight_norm,
                       torch::Tensor segment_ids, const int64_t num_segments);

}  // namespace functional
}  // namespace recis
