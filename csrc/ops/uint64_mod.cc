#include "uint64_mod.h"
namespace recis {
namespace functional {
void uint64_mod_cpu_kernel(const int64_t* inputs, int64_t* output,
                           const int64_t input_size, const int64_t num) {
  for (int64_t i = 0; i < input_size; ++i) {
    uint64_t out = static_cast<uint64_t>(inputs[i]) % num;
    output[i] = static_cast<int64_t>(out);
  }
}

torch::Tensor uint64_mod(torch::Tensor inputs, torch::Scalar num) {
  auto input_size = inputs.numel();
  auto output = torch::empty_like(inputs);
  if (!input_size) {
    return output;
  }

  if (inputs.device().is_cuda()) {
    uint64_mod_cuda(inputs, output, input_size, num);
  } else {
    uint64_mod_cpu_kernel(inputs.data_ptr<int64_t>(),
                          output.data_ptr<int64_t>(), input_size,
                          num.to<int64_t>());
  }

  return output;
}
}  // namespace functional
}  // namespace recis
