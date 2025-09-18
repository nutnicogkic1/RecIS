#include "ragged_to_sparse.h"

#include "ragged_common.cuh"

namespace recis {
namespace functional {

template <typename index_t>
void ragged_to_sparse_cpu_kernel(const std::vector<index_t*>& offset_list,
                                 const int32_t splits_len,
                                 const index_t max_final_pos,
                                 const index_t nvals,
                                 int64_t* __restrict__ indices_value) {
  std::vector<index_t> pos(splits_len);
  std::vector<index_t> index_prefix(splits_len);
  index_t& final_pos = pos[splits_len - 1];
  index_t next_index = 0;
  for (; final_pos < max_final_pos; ++final_pos) {
    for (index_t dim = splits_len - 2; dim >= 0; --dim) {
      while (IsSegCompleted(pos, dim, offset_list)) {
        pos[dim] += 1;
      }
    }
    for (size_t dim = 0; dim < index_prefix.size(); ++dim) {
      index_t start = dim > 0 ? offset_list[dim - 1][pos[dim - 1]] : 0;
      index_prefix[dim] = pos[dim] - start;
    }
    const auto& final_splits = offset_list[splits_len - 1];
    index_t slice_len = final_splits[final_pos + 1] - final_splits[final_pos];
    for (index_t i = 0; i < slice_len; ++i) {
      int dim = 0;
      for (index_t index : index_prefix) {  // index_prefix
        indices_value[next_index + nvals * dim] = index;
        ++dim;
      }
      indices_value[next_index + nvals * dim] = i;  // index_middle
      ++dim;
      ++next_index;
    }
  }
}

torch::Tensor ragged_to_sparse(torch::Tensor values,
                               std::vector<torch::Tensor> offset_splits) {
  TORCH_CHECK(all_same_type(offset_splits, torch::kInt32) ||
                  all_same_type(offset_splits, torch::kInt64),
              "ragged_to_sparse: each offset_splits should be torch::kInt32 or "
              "torch::kInt64");
  TORCH_CHECK(
      (all_cpu(offset_splits) && all_cpu({values})) or
          (all_cuda(offset_splits) && all_cuda({values})),
      "ragged_to_sparse: offset_splits and value should be all on cpu or cuda");

  const int splits_len = offset_splits.size();
  // set dense shape
  std::vector<int64_t> dense_shape;
  const int indices_len = splits_len + 1;
  dense_shape.reserve(indices_len);
  dense_shape.push_back(offset_splits.front().size(0) - 1);
  for (int dim = 0; dim < splits_len; ++dim) {
    const auto& offset = offset_splits[dim];
    auto max_width =
        (offset.slice(0, 1) - offset.slice(0, 0, -1)).max().item<int64_t>();
    dense_shape.push_back(max_width);
  }
  torch::Tensor indices;
  if (values.device().is_cuda()) {
    TORCH_CHECK_NOT_IMPLEMENTED(false,
                                "Not implemented for cuda ragged to sparse");
  } else {
    AT_DISPATCH_INDEX_TYPES(
        offset_splits.front().scalar_type(), "ragged_to_sparse_cpu_op", ([&] {
          std::vector<index_t*> offset_list;
          offset_list.reserve(splits_len);
          for (int i = 0; i < splits_len; ++i) {
            offset_list.push_back(offset_splits[i].data_ptr<index_t>());
          }
          const index_t max_final_pos = offset_splits.back().numel() - 1;
          const index_t nvals = offset_list.back()[max_final_pos];
          indices = torch::empty({indices_len, nvals}, torch::kInt64);
          ragged_to_sparse_cpu_kernel<index_t>(offset_list, splits_len,
                                               max_final_pos, nvals,
                                               indices.data_ptr<int64_t>());
        }));
  }
  auto out_tensor =
      torch::sparse_coo_tensor(indices, values, dense_shape).coalesce();
  return out_tensor;
}
}  // namespace functional
}  // namespace recis
