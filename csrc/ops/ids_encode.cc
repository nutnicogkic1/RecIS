#include "ops/ids_encode.h"

namespace recis {
namespace functional {
torch::Tensor ids_encode_cpu(std::vector<torch::Tensor> inputs,
                             torch::Tensor table_ids) {
  int num_tensors = inputs.size();
  std::vector<torch::Tensor> out_puts;
  for (int i = 0; i < num_tensors; ++i) {
    auto mask = torch::ones_like(inputs[i]);
    mask =
        torch::bitwise_left_shift(mask, _MAX_BIT_SIZE - _MAX_ENCODE_SIZE) - 1;
    auto offset = torch::bitwise_left_shift(table_ids[i],
                                            _MAX_BIT_SIZE - _MAX_ENCODE_SIZE);
    auto en_ids = torch::bitwise_and(inputs[i], mask) + offset;
    out_puts.push_back(en_ids);
  }
  auto output = torch::cat(out_puts);
  return output;
}
torch::Tensor ids_encode(std::vector<torch::Tensor> inputs,
                         torch::Tensor table_ids) {
  if (inputs[0].device().is_cuda()) {
    return ids_encode_cuda(inputs, table_ids);
  } else {
    return ids_encode_cpu(inputs, table_ids);
  }
}
}  // namespace functional
}  // namespace recis
