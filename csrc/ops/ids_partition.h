#include <torch/extension.h>

#include <tuple>
#include <vector>
namespace recis {
namespace functional {

constexpr int kSliceBits = 16;
constexpr int kSliceSize = 1 << kSliceBits;
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ids_partition_cuda(
    const torch::Tensor &ids, int64_t num_parts);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ids_partition_cpu(
    const torch::Tensor &ids, int64_t num_parts);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ids_partition(
    const torch::Tensor &ids, int64_t num_parts);
}  // namespace functional
}  // namespace recis
