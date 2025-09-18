#ifndef COLUMN_IO_CC_COLUMN_IO_DATASET_IMPL_VEC_TENSOR_CONVERTER_H_
#define COLUMN_IO_CC_COLUMN_IO_DATASET_IMPL_VEC_TENSOR_CONVERTER_H_
#include <cstddef>
#include <vector>

#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"
namespace column {
namespace dataset {
namespace detail {
template <typename T> Tensor VecToTensor(const std::vector<T> &source) {
  Tensor out(DataTypeToEnum<T>::value, {source.size()});
  for (size_t i = 0; i < source.size(); i++) {
    out.Raw<T>()[i] = source[i];
  }
  return out;
}
template <typename T>
std::vector<Tensor> VecsToTensor(const std::vector<std::vector<T>> &source) {
  std::vector<Tensor> output;
  output.reserve(source.size());
  for (const auto &vec : source) {
    output.emplace_back(VecToTensor<T>(vec));
  }
  return output;
}
} // namespace detail
} // namespace dataset
} // namespace column
#endif