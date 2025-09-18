#pragma once
#include <string>
#include <vector>

#include "ATen/core/Generator.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/ivalue.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/extension.h"
#include "torch/nn/init.h"
#include "torch/types.h"
namespace recis {
namespace embedding {
class Generator : public torch::CustomClassHolder {
 public:
  Generator(const std::vector<int64_t> &shape, torch::Dtype dtype);
  virtual torch::Tensor Generate(const std::vector<int64_t> &shape = {}) = 0;
  virtual void Initialize(torch::Tensor input) = 0;
  virtual const std::vector<int64_t> &Shape() { return shape_; }
  virtual const c10::TensorOptions TensorOption() { return option_; }
  virtual void set_device(torch::Device device);
  // interface for python
  static torch::Tensor DoGenerator(torch::intrusive_ptr<Generator> genrate);
  virtual ~Generator() = default;

 protected:
  std::vector<int64_t> shape_;
  torch::TensorOptions option_;
};

torch::intrusive_ptr<Generator> MakeEmptyGenerator(
    const std::vector<int64_t> &shape, torch::Dtype dtype);

torch::intrusive_ptr<Generator> MakeConstantGenerator(
    const std::vector<int64_t> &shape, torch::Dtype dtype, double init_val);

torch::intrusive_ptr<Generator> MakeUniformGenerator(
    const std::vector<int64_t> &shape, torch::Dtype dtype, double a, double b,
    c10::optional<torch::Generator> generator);
torch::intrusive_ptr<Generator> MakeNormalGenerator(
    const std::vector<int64_t> &shape, torch::Dtype dtype, double mean,
    double std, c10::optional<torch::Generator> generator);

torch::intrusive_ptr<Generator> MakeXavierUniFormGenerator(
    const std::vector<int64_t> &shape, torch::Dtype dtype, double gain,
    c10::optional<torch::Generator> generator);

torch::intrusive_ptr<Generator> MakeXavierNormalGenerator(
    const std::vector<int64_t> &shape, torch::Dtype dtype, double gain,
    c10::optional<torch::Generator> generator);

torch::intrusive_ptr<Generator> MakeKaimingUniformGenerator(
    const std::vector<int64_t> &shape, torch::Dtype dtype, double a,
    const std::string &mode_s, const std::string &nonlinearity_s,
    c10::optional<torch::Generator> generator);

torch::intrusive_ptr<Generator> MakeKaimingNormalGenerator(
    const std::vector<int64_t> &shape, torch::Dtype dtype, double a,
    const std::string &mode_s, const std::string &nonlinearity_s,
    c10::optional<torch::Generator> generator);

torch::intrusive_ptr<Generator> MakeTruncNormalGenerator(
    const std::vector<int64_t> &shape, torch::Dtype type, double mean,
    double std, double a, double b, at::optional<torch::Generator> generator);
}  // namespace embedding
}  // namespace recis
