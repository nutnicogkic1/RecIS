#include "ops/fused_uint64_mod.h"
namespace recis {
namespace functional {

int64_t uint64_mod(int64_t x, int64_t mod) {
  return static_cast<int64_t>(static_cast<uint64_t>(x) % mod);
}

void fused_uint64_mod_cpu(std::vector<torch::Tensor>& inputs,
                          std::vector<torch::Tensor>& outputs,
                          torch::Tensor mod_vec) {
  // use torch::parallel_for
  int num_tensors = inputs.size();
  auto mod_vec_data = mod_vec.data_ptr<int64_t>();
  for (int i = 0; i < num_tensors; i++) {
    torch::parallel_for(
        0, inputs[i].size(0), 1, [&](int64_t begin, int64_t end) {
          for (int64_t j = begin; j < end; j++) {
            auto input_data = inputs[i].data_ptr<int64_t>();
            auto output_data = outputs[i].data_ptr<int64_t>();
            output_data[j] = uint64_mod(input_data[j], mod_vec_data[i]);
          }
        });
  }
}

std::vector<torch::Tensor> fused_uint64_mod(std::vector<torch::Tensor> inputs,
                                            torch::Tensor mod_vec) {
  int num_tensors = inputs.size();
  std::vector<torch::Tensor> outputs(num_tensors);
  for (int i = 0; i < num_tensors; i++) {
    outputs[i] = torch::empty_like(inputs[i]);
  }
  if (inputs[0].device().is_cuda()) {
    fused_uint64_mod_cuda(inputs, outputs, mod_vec);
  } else {
    fused_uint64_mod_cpu(inputs, outputs, mod_vec);
  }
  return outputs;
}
}  // namespace functional
}  // namespace recis
