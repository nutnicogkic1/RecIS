#include "ops/fused_bucketized.h"
namespace recis {
namespace functional {

int32_t bucketize(float value, float* boundaries, int32_t boundary_len) {
  int32_t bucket = 0;
  int32_t count = boundary_len;
  while (count > 0) {
    int32_t left = bucket;
    int32_t step = count / 2;
    left += step;
    if (!(value < boundaries[left])) {
      bucket = ++left;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return bucket;
}
void fused_bucketized_cpu(std::vector<torch::Tensor>& inputs,
                          std::vector<torch::Tensor>& outputs,
                          std::vector<torch::Tensor>& boundaries) {
  // use torch::parallel_for
  int64_t num_tensors = inputs.size();
  for (int64_t i = 0; i < num_tensors; i++) {
    torch::parallel_for(
        0, inputs[i].size(0), 1, [&](int64_t begin, int64_t end) {
          for (int64_t j = begin; j < end; j++) {
            auto input_data = inputs[i].data_ptr<float>();
            auto output_data = outputs[i].data_ptr<int64_t>();
            output_data[j] =
                bucketize(input_data[j], boundaries[i].data_ptr<float>(),
                          boundaries[i].numel());
          }
        });
  }
}

std::vector<torch::Tensor> fused_bucketized(
    std::vector<torch::Tensor> inputs, std::vector<torch::Tensor> boundaries) {
  int64_t num_tensors = inputs.size();
  std::vector<int64_t> sizes(num_tensors);
  std::vector<torch::Tensor> outputs(num_tensors);
  for (int64_t i = 0; i < num_tensors; i++) {
    sizes[i] = inputs[i].numel();
  }
  int64_t total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
  torch::Tensor output = torch::empty(
      {total_size},
      torch::TensorOptions().dtype(torch::kInt64).device(inputs[0].device()));
  outputs = torch::split(output, sizes);
  if (inputs[0].device().is_cuda()) {
    fused_bucketized_cuda(inputs, outputs, boundaries);
  } else {
    fused_bucketized_cpu(inputs, outputs, boundaries);
  }
  return outputs;
}
}  // namespace functional
}  // namespace recis
