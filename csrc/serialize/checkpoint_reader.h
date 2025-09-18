#pragma once
#include <string>

#include "ATen/core/TensorBody.h"
#include "c10/core/ScalarType.h"
#include "c10/util/intrusive_ptr.h"
#include "serialize/load_bundle.h"
#include "torch/extension.h"
namespace recis {
namespace serialize {
class CheckpointReader : public torch::CustomClassHolder {
 public:
  CheckpointReader(const std::string &path);

  void Init();

  std::vector<std::string> ListTensors();
  at::Tensor LoadTensor(const std::string &tensor_name);
  std::vector<int64_t> TensorShape(const std::string &tensor_name);
  torch::Tensor TensorType(const std::string &tensor_name);

 private:
  std::string path_;
  at::intrusive_ptr<LoadBundle> load_bundle_;
};
}  // namespace serialize
}  // namespace recis