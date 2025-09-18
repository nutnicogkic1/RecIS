#include "ragged_to_dense.h"

#include "ragged_common.cuh"

namespace recis {
namespace functional {

template <typename index_t, typename scalar_t>
void ragged_to_dense_cpu_kernel(const scalar_t* __restrict__ values,
                                const std::vector<index_t*>& offsets,
                                const int32_t num_offsets,
                                const index_t max_final_pos,
                                const std::vector<int64_t>& steps,
                                scalar_t* __restrict__ output) {
  std::vector<index_t> index_prefix(num_offsets);  // index except the last dim
  std::vector<index_t> pos(num_offsets);
  index_t& final_pos = pos[num_offsets - 1];

  // final_pos: deal with a basic seg (inner offsets) each loop
  // pos[dim]: record how many logical members have been processed in the cur
  // dim layer
  for (; final_pos < max_final_pos; ++final_pos) {
    for (int dim = num_offsets - 2; dim >= 0; --dim) {
      while (IsSegCompleted(pos, dim, offsets)) {
        // process the next seg in the dim layer
        pos[dim] += 1;
      }
    }
    for (size_t dim = 0; dim < index_prefix.size(); ++dim) {
      index_t start = dim > 0 ? offsets[dim - 1][pos[dim - 1]] : 0;
      // prefix index of val in final_pos seg
      index_prefix[dim] = pos[dim] - start;
    }
    int64_t layer_base_offset = 0;
    for (size_t dim = 0; dim < index_prefix.size(); ++dim) {
      layer_base_offset += index_prefix[dim] * steps[dim];
    }
    const auto& final_splits = offsets[num_offsets - 1];
    index_t slice_len = final_splits[final_pos + 1] - final_splits[final_pos];
    for (index_t last_dim_offset = 0; last_dim_offset < slice_len;
         ++last_dim_offset) {
      output[layer_base_offset + last_dim_offset] =
          values[final_splits[final_pos] + last_dim_offset];
    }
  }
}

torch::Tensor ragged_to_dense(torch::Tensor values,
                              const std::vector<torch::Tensor>& offsets,
                              torch::Scalar default_value) {
  TORCH_CHECK(values.dim() == 1, "values must dim == 1");
  TORCH_CHECK(all_same_type(offsets, torch::kInt32) ||
                  all_same_type(offsets, torch::kInt64),
              "ragged_to_dense: each offsets should be torch::kInt32 or "
              "torch::kInt64");
  TORCH_CHECK(
      (all_cpu(offsets) && all_cpu({values})) or
          (all_cuda(offsets) && all_cuda({values})),
      "ragged_to_dense: offsets and value should be all on cpu or cuda");

  const int num_offsets = offsets.size();
  torch::Tensor output;
  std::vector<int64_t> dense_shape;
  dense_shape.push_back(offsets[0].size(0) - 1);
  for (int dim = 0; dim < num_offsets; ++dim) {
    const auto& offset = offsets[dim];
    auto max_width =
        (offset.slice(0, 1) - offset.slice(0, 0, -1)).max().item<int64_t>();
    dense_shape.push_back(max_width);
  }
  const auto& options = values.options();

  if (values.device().is_cuda()) {
    output = torch::empty(dense_shape, options);
    ragged_to_dense_cuda(values, offsets, output, default_value);
  } else {
    std::vector<int64_t> dense_step(dense_shape.size());
    for (int dim = num_offsets; dim >= 0; --dim) {
      dense_step[dim] =
          (dim == num_offsets) ? 1 : dense_shape[dim + 1] * dense_step[dim + 1];
    }
    output = torch::full(dense_shape, default_value, options);

    AT_DISPATCH_INDEX_TYPES(
        offsets.front().scalar_type(), "ragged_to_dense_cpu_op_0", ([&] {
          AT_DISPATCH_ALL_TYPES(
              values.scalar_type(), "ragged_to_dense_cpu_op_1", ([&] {
                std::vector<index_t*> offsets_list;
                offsets_list.reserve(num_offsets);
                for (int i = 0; i < num_offsets; ++i) {
                  offsets_list.push_back(offsets[i].data_ptr<index_t>());
                }
                ragged_to_dense_cpu_kernel<index_t, scalar_t>(
                    values.data_ptr<scalar_t>(), offsets_list, num_offsets,
                    offsets.back().numel() - 1, dense_step,
                    output.data_ptr<scalar_t>());
              }));
        }));
  }

  return output;
}
}  // namespace functional
}  // namespace recis
