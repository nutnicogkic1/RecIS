#include "ops/calc_ragged_index.h"

#include <tuple>

#include "ATen/Dispatch.h"
#include "ATen/Utils.h"
#include "ATen/core/TensorBody.h"
#include "c10/core/DeviceType.h"
#include "c10/util/Exception.h"
#include "c10/util/irange.h"
#include "ops/ragged_common.cuh"
namespace recis {
namespace functional {

std::tuple<at::Tensor, at::Tensor> calc_ragged_index_drop_left_pad_right_cpu(
    at::Tensor drop_num, at::Tensor pad_num, at::Tensor offset,
    at::Tensor topk_index, at::Tensor indicator) {
  drop_num = drop_num.cpu();
  topk_index = topk_index.cpu();
  offset = offset.cpu();
  indicator = indicator.cpu();
  int64_t row_num = topk_index.size(0);
  int64_t col_num = topk_index.size(1);
  at::Tensor out_value_index = torch::empty(
      {row_num * col_num}, topk_index.options().device(torch::kCPU));
  at::Tensor out_offset =
      torch::empty({row_num + 1}, offset.options().device(torch::kCPU));
  int64_t valid_value_num = 0;
  TORCH_CHECK(indicator.numel() == row_num, "indicator==row_num",
              ";indicator.numel():", indicator.numel(), "; row_num:", row_num);
  AT_DISPATCH_INDEX_TYPES(
      indicator.scalar_type(), "calc_ragged_index_drop_left_pad_right_cpu_0",
      [&]() {
        using indicator_t = index_t;
        AT_DISPATCH_INDEX_TYPES(
            drop_num.scalar_type(),
            "calc_ragged_index_drop_left_pad_right_cpu_1", [&]() {
              using drop_num_t = index_t;
              AT_DISPATCH_INDEX_TYPES(
                  topk_index.scalar_type(),
                  "calc_ragged_index_drop_left_pad_right_cpu_2", [&]() {
                    using topk_index_t = index_t;
                    AT_DISPATCH_INDEX_TYPES(
                        offset.scalar_type(),
                        "calc_ragged_index_drop_left_pad_right_cpu_3", [&]() {
                          using offset_t = index_t;
                          auto drop_num_v = drop_num.data_ptr<drop_num_t>();
                          auto topk_index_v =
                              topk_index.data_ptr<topk_index_t>();
                          auto offset_v = offset.data_ptr<offset_t>();
                          auto out_value_index_v =
                              out_value_index.data_ptr<topk_index_t>();
                          auto out_offset_v = out_offset.data_ptr<offset_t>();
                          auto indicator_v = indicator.data_ptr<indicator_t>();
                          if (out_offset.numel() > 0) {
                            out_offset_v[0] = 0;
                          }
                          for (auto row_index : c10::irange(row_num)) {
                            topk_index_t src_row_index = indicator_v[row_index];
                            topk_index_t row_beg = offset_v[src_row_index];
                            topk_index_t row_end = offset_v[src_row_index + 1];
                            topk_index_t row_drop = drop_num_v[src_row_index];
                            for (auto col_index : c10::irange(col_num)) {
                              topk_index_t topk_offset =
                                  topk_index_v[row_index * col_num +
                                               col_index] +
                                  row_beg;
                              topk_offset += row_drop;
                              if (topk_offset >= row_beg &&
                                  topk_offset < row_end) {
                                out_value_index_v[valid_value_num] =
                                    topk_offset;
                                valid_value_num += 1;
                              }
                            }
                            out_offset_v[row_index + 1] = valid_value_num;
                          }
                        });
                  });
            });
      });
  out_value_index = out_value_index.narrow(0, 0, valid_value_num);
  return std::make_tuple(out_value_index, out_offset);
}

std::tuple<at::Tensor, at::Tensor> calc_ragged_index_drop_left_pad_right(
    at::Tensor drop_num, at::Tensor pad_num, at::Tensor offset,
    at::Tensor topk_index, at::Tensor indicator) {
  return calc_ragged_index_drop_left_pad_right_cpu(drop_num, pad_num, offset,
                                                   topk_index, indicator);
}

std::tuple<at::Tensor, at::Tensor> calc_ragged_index(
    at::Tensor drop_num, at::Tensor pad_num, at::Tensor drop_side,
    at::Tensor pad_side, at::Tensor offset, at::Tensor topk_index,
    at::Tensor indicator) {
  TORCH_CHECK(all_same_type({drop_side, pad_side}, torch::kBool),
              "drop_side and pad_side must be bool");
  int key = static_cast<int>(pad_side.cpu().item<bool>() * 2) +
            static_cast<int>(drop_side.cpu().item<bool>());
  at::Tensor out_value_index, out_offset;
  switch (key) {
    // case 0: drop left, pad left
    case 3: {
      TORCH_CHECK(false, "drop left, pad left is not supported yet");
    };
    // case 1: drop left, pad right
    case 1: {
      std::tie(out_value_index, out_offset) =
          calc_ragged_index_drop_left_pad_right(drop_num, pad_num, offset,
                                                topk_index, indicator);
      break;
    };
    // case 2: drop right, pad left
    case 2: {
      TORCH_CHECK(false, "drop right, pad left is not supported yet");
    };
    // case 3: drop right , pad right
    case 0: {
      TORCH_CHECK(false, "drop right, pad right is not supported yet");
    }
  };
  return std::make_tuple(out_value_index, out_offset);
}
}  // namespace functional
}  // namespace recis
