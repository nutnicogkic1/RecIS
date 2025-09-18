#include "feature_cross_ragged.h"

#include <torch/torch.h>

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "c10/util/Exception.h"
#include "c10/util/Logging.h"
#include "ragged_common.cuh"

namespace recis {
namespace functional {

template <typename index_t, typename scalar_t>
inline static void get_uniq_id_weight_map(
    const torch::Tensor& values, const torch::Tensor& weights, index_t start,
    index_t end, std::unordered_map<int64_t, scalar_t>& uniq) {
  bool without_weight = weights.numel() == 0;
  const auto& weights_accessor = weights.accessor<scalar_t, 1>();
  const auto& values_accessor = values.accessor<int64_t, 1>();
  uniq.clear();
  for (index_t i = start; i < end; ++i) {
    int64_t id = values_accessor[i];
    scalar_t weight = without_weight ? 1.0 : weights_accessor[i];
    if (uniq.find(id) == uniq.end()) {
      uniq[id] = weight;
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> feature_cross_ragged(
    const torch::Tensor& x_value, const torch::Tensor& x_offsets,
    const torch::Tensor& x_weight, const torch::Tensor& y_value,
    const torch::Tensor& y_offsets, const torch::Tensor& y_weight) {
  TORCH_CHECK(all_same_type({x_offsets, y_offsets}, torch::kInt32) ||
                  all_same_type({x_offsets, y_offsets}, torch::kInt64),
              "feature_cross_ragged: each offset should be torch::kInt32 or "
              "torch::kInt64");
  TORCH_CHECK(all_same_type({x_value, y_value}, torch::kInt64),
              "feature_cross_ragged: ids for x and y must be torch::kInt64");

  if (x_value.device().is_cuda()) {
    return feature_cross_ragged_cuda(x_value, x_offsets, x_weight, y_value,
                                     y_offsets, y_weight);
  }

  TORCH_CHECK(
      all_cpu({x_value, y_value, x_offsets, y_offsets, x_weight, y_weight}));

  int64_t num_seg = std::min(x_offsets.numel(), y_offsets.numel()) - 1;

  torch::Tensor out_tensor, out_weight_tensor, out_offsets_tensor;

  AT_DISPATCH_INDEX_TYPES(
      x_offsets.scalar_type(), "feature_cross_ragged_cpu_op_0", [&] {
        AT_DISPATCH_ALL_TYPES(
            x_weight.scalar_type(), "feature_cross_ragged_cpu_op_1", [&] {
              std::unordered_map<int64_t, scalar_t> uniq_x;
              std::unordered_map<int64_t, scalar_t> uniq_y;
              std::vector<int64_t> out_ids;
              std::vector<scalar_t> out_weights;

              std::vector<index_t> out_offsets = {0};
              auto x_accesseor = x_offsets.accessor<index_t, 1>();
              auto y_accesseor = y_offsets.accessor<index_t, 1>();

              // TODO: at::parallel_for get_uniq_id_weight_map and
              // at::parallel_for hash
              for (auto i = 0; i < num_seg; i++) {
                get_uniq_id_weight_map(x_value, x_weight, x_accesseor[i],
                                       x_accesseor[i + 1], uniq_x);
                get_uniq_id_weight_map(y_value, y_weight, y_accesseor[i],
                                       y_accesseor[i + 1], uniq_y);

                index_t k = 0;
                for (const auto& x_kv : uniq_x) {
                  for (const auto& y_kv : uniq_y) {
                    std::vector<uint64_t> hash_keys = {
                        static_cast<uint64_t>(x_kv.first),
                        static_cast<uint64_t>(y_kv.first)};
                    out_ids.push_back(static_cast<int64_t>(
                        murmur_hash_64(hash_keys.data(), 0, hash_keys.size())));
                    out_weights.push_back(x_kv.second * y_kv.second);
                    k++;
                  }
                }
                out_offsets.push_back(out_offsets[i] + k);
              }
              out_tensor = torch::tensor(out_ids, x_value.options());
              out_weight_tensor =
                  torch::tensor(out_weights, x_weight.options());
              out_offsets_tensor =
                  torch::tensor(out_offsets, x_offsets.options());
            });
      });
  return std::make_tuple(out_tensor, out_offsets_tensor, out_weight_tensor);
}

}  // namespace functional
}  // namespace recis
