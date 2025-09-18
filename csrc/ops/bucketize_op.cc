#include "bucketize_op.h"
namespace recis {
namespace functional {
template <typename scalar_t>
void bucketize_cpu_kernel(const scalar_t* __restrict__ values,
                          const scalar_t* __restrict__ boundaries,
                          int64_t* __restrict__ output,
                          const int64_t input_size, const int boundary_len) {
  for (int64_t i = 0; i < input_size; ++i) {
    auto first_bigger_it =
        std::upper_bound(boundaries, boundaries + boundary_len, values[i]);
    output[i] = first_bigger_it - boundaries;
  }
}

torch::Tensor bucketize_op(torch::Tensor values, torch::Tensor boundaries) {
  int64_t input_size = values.numel();
  auto output = torch::empty_like(values, torch::kInt64);
  if (!input_size) {
    return output;
  }

  int boundary_len = boundaries.numel();
  if (values.device().is_cuda()) {
    bucketize_cuda_op(values, boundaries, output, input_size, boundary_len);
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        values.scalar_type(), "bucketize_cpu", ([&] {
          bucketize_cpu_kernel<scalar_t>(
              values.data_ptr<scalar_t>(), boundaries.data_ptr<scalar_t>(),
              output.data_ptr<int64_t>(), input_size, boundary_len);
        }));
  }

  return output;
}
}  // namespace functional
}  // namespace recis
