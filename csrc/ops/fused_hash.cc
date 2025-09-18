#include "ops/fused_hash.h"
namespace recis {
namespace functional {

std::vector<torch::Tensor> fused_hash(std::vector<torch::Tensor> inputs,
                                      std::vector<torch::Tensor> input_offsets,
                                      const std::string& hash_type) {
  int num_tensors = inputs.size();
  std::vector<torch::Tensor> outputs(num_tensors);
  for (int i = 0; i < num_tensors; i++) {
    outputs[i] = torch::empty(
        {input_offsets[i].numel() - 1},
        torch::TensorOptions().dtype(torch::kInt64).device(inputs[i].device()));
  }
  if (inputs[0].device().is_cuda()) {
    fused_hash_cuda(inputs, input_offsets, outputs, hash_type);
  } else {
    throw std::runtime_error("Fused hash op only supports cuda tensors.");
  }
  return outputs;
}
}  // namespace functional
}  // namespace recis
