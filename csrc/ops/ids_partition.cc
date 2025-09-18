#include "ops/ids_partition.h"

namespace recis {
namespace functional {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ids_partition_cpu(
    const torch::Tensor &ids, int64_t num_parts) {
  auto [unique_ids, reverse_indice, count] =
      at::native::_unique2_cpu(ids, true, true, false);
  auto ids_module = torch::abs(torch::fmod(unique_ids, kSliceSize));

  auto arg_pos = torch::argsort(ids_module);
  auto sorted_ids = ids_module.index({arg_pos});
  ids_module.index({arg_pos});

  auto m = kSliceSize / num_parts;
  auto n = kSliceSize % num_parts;
  std::vector<int64_t> parts(num_parts, m);
  for (int64_t i = 0; i < n; ++i) {
    parts[i] += 1;
  }
  auto parts_boundary = torch::cumsum(
      torch::tensor(parts, torch::TensorOptions().dtype(torch::kInt32)), 0);

  auto boundary_indices = torch::searchsorted(sorted_ids, parts_boundary);

  boundary_indices = torch::cat(
      {torch::tensor({0}, torch::TensorOptions().dtype(torch::kInt32)),
       boundary_indices});

  auto segment_size =
      boundary_indices.slice(0, 1) - boundary_indices.slice(0, 0, -1);

  auto range_index = torch::argsort(arg_pos);

  return std::make_tuple(unique_ids.index({arg_pos}), segment_size,
                         range_index.index({reverse_indice}));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ids_partition(
    const torch::Tensor &ids, int64_t num_parts) {
  TORCH_CHECK(ids.dtype() == torch::kInt64, "ids must be int64");
  TORCH_CHECK(ids.dim() == 1, "ids must be 1-dim");

  if (ids.device().is_cuda()) {
    return ids_partition_cuda(ids, num_parts);
  }
  return ids_partition_cpu(ids, num_parts);
}
}  // namespace functional
}  // namespace recis
