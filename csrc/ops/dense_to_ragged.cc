#include "ops/dense_to_ragged.h"

#include <iostream>

namespace recis {
namespace functional {

std::tuple<torch::Tensor, torch::Tensor> dense_to_ragged_cpu(
    const torch::Tensor& data, const torch::Tensor& invalid_value) {
  torch::Tensor values_tensor;
  torch::Tensor offsets_tensor;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Int, at::ScalarType::Long, data.scalar_type(),
      "ragged_to_dense_cuda", ([&] {
        const auto rows = data.size(0);
        const auto cols = data.size(1);
        scalar_t x = invalid_value.data_ptr<scalar_t>()[0];

        std::vector<int> lengths(rows, 0);

        auto input_accessor = data.accessor<scalar_t, 2>();
        for (int i = 0; i < rows; ++i) {
          int last_index = -1;
          for (int j = 0; j < cols; ++j) {
            if (input_accessor[i][j] != x) {
              last_index = j;
            }
          }
          lengths[i] = last_index + 1;
        }

        std::vector<int> offsets(rows + 1, 0);
        std::partial_sum(lengths.begin(), lengths.end(), offsets.begin() + 1);

        int total_length = offsets.back();
        std::vector<scalar_t> values(total_length);

        for (int i = 0; i < rows; ++i) {
          int start = offsets[i];
          int len = lengths[i];
          for (int j = 0; j < len; ++j) {
            values[start + j] = input_accessor[i][j];
          }
        }

        auto options = data.options();
        values_tensor =
            torch::from_blob(values.data(), {total_length}, options).clone();
        offsets_tensor =
            torch::from_blob(offsets.data(), {rows + 1},
                             torch::TensorOptions().dtype(torch::kInt32))
                .clone();
      }));
  return std::make_tuple(values_tensor, offsets_tensor);
}

std::tuple<torch::Tensor, torch::Tensor> dense_to_ragged(
    const torch::Tensor& data, bool check_invalid,
    const torch::Tensor& invalid_value) {
  TORCH_CHECK(data.dim() == 2, "Input must be a 2D tensor");
  auto device = data.device();

  if (check_invalid) {
    if (device.is_cuda()) {
      return dense_to_ragged_cuda(data, invalid_value);
    } else {
      return dense_to_ragged_cpu(data, invalid_value);
    }
  } else {
    int rows = data.size(0);
    int cols = data.size(1);
    torch::Tensor offsets = torch::arange(
        0, rows + 1, cols,
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    return std::make_tuple(data.view(-1), offsets);
  }
}
}  // namespace functional
}  // namespace recis
