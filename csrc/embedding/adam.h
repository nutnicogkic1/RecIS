#pragma once
#include <cmath>
#include <string>
#include <unordered_map>

#include "ATen/core/Dict.h"
#include "ATen/core/TensorBody.h"
#include "c10/core/ScalarType.h"
#include "c10/util/flat_hash_map.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/irange.h"
#include "embedding/hashtable.h"
#include "embedding/optim.h"
#include "embedding/slot_group.h"
#include "torch/types.h"

namespace recis {
namespace optim {
struct SparseAdamOptions
    : public SparseOptimizerCloneableOptions<SparseAdamOptions> {
  SparseAdamOptions(double lr = 1e-3);
  TORCH_ARG(double, lr) = 1e-3;
  typedef std::tuple<double, double> betas_t;
  TORCH_ARG(betas_t, betas) = std::make_tuple(0.9, 0.999);
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(double, weight_decay) = 1e-2;
  TORCH_ARG(bool, amsgrad) = false;

 public:
  double get_lr() const override;
  void set_lr(const double lr) override;
};
struct TORCH_API SparseAdamParamState
    : public SparseOptimizerCloneableParamState<SparseAdamParamState> {
 public:
  SparseAdamParamState()
      : step_dtype_(torch::kInt64), beta_dtype_(torch::kDouble) {}
  using ParamContainer = at::intrusive_ptr<embedding::Slot>;
  torch::Dtype step_dtype() const { return step_dtype_; }
  torch::Dtype beta_dtype() const { return beta_dtype_; }

  const ParamContainer param() const { return param_; }
  void param(const ParamContainer param) { param_ = param; }
  const ParamContainer exp_avg() const { return exp_avg_; }
  void exp_avg(const ParamContainer param) { exp_avg_ = param; }
  const ParamContainer exp_avg_sq() const { return exp_avg_sq_; }
  void exp_avg_sq(const ParamContainer param) { exp_avg_sq_ = param; }
  const HashTablePtr hashtable() const { return hashtable_; }
  HashTablePtr hashtable(HashTablePtr hashtable) {
    hashtable_ = hashtable;
    return hashtable_;
  }

  TORCH_API friend bool operator==(const SparseAdamParamState &lhs,
                                   const SparseAdamParamState &rhs);
  TORCH_ARG(torch::Tensor, beta1);
  TORCH_ARG(torch::Tensor, beta2);
  TORCH_ARG(torch::Tensor, step);

 private:
  at::intrusive_ptr<embedding::Slot> param_;
  at::intrusive_ptr<embedding::Slot> exp_avg_;
  at::intrusive_ptr<embedding::Slot> exp_avg_sq_;
  torch::Dtype step_dtype_;
  torch::Dtype beta_dtype_;
  HashTablePtr hashtable_;
};

class SparseAdam : public SparseOptimizer {
 public:
  explicit SparseAdam(std::vector<SparseOptimizerParamGroup> param_groups,
                      SparseAdamOptions defaults = {})
      : SparseOptimizer(std::move(param_groups),
                        std::make_unique<SparseAdamOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
    auto betas = defaults.betas();
    TORCH_CHECK(0 <= std::get<0>(betas) && std::get<0>(betas) < 1.0,
                "Invalid beta parameter at index 0: ", std::get<0>(betas));
    TORCH_CHECK(0 <= std::get<1>(betas) && std::get<1>(betas) < 1.0,
                "Invalid beta parameter at index 1: ", std::get<1>(betas));
    TORCH_CHECK(defaults.weight_decay() >= 0,
                "Invalid weight_decay value: ", defaults.weight_decay());
    for (const auto &param_group : param_groups_) {
      for (const auto &param : param_group.params()) {
        InitParamState(param.first, param.second);
      }
    }
  }
  explicit SparseAdam(std::unordered_map<std::string, HashTablePtr> params,
                      SparseAdamOptions defaults = {})
      : SparseAdam({SparseOptimizerParamGroup(std::move(params))}, defaults) {}
  const std::tuple<std::unordered_map<std::string, HashTablePtr>,
                   std::unordered_map<std::string, torch::Tensor>>
  state_dict() override;
  void load_state_dict(torch::Dict<std::string, HashTablePtr> hashtables,
                       torch::Dict<std::string, torch::Tensor> steps) override;
  virtual void add_param_group(
      const SparseOptimizerParamGroup &param_group) override;
  virtual void add_parameters(
      const torch::Dict<std::string, HashTablePtr> &parameters) override;
  void InitParamState(const std::string &param_name, HashTablePtr param);
  void step() override;
  void zero_grad();
  void set_grad_accum_steps(const int64_t steps) override {
    grad_accum_steps_ = steps;
  }
  void set_lr(const double lr) override {
    for (auto &group : param_groups_) {
      group.options().set_lr(lr);
    }
  }
  static c10::intrusive_ptr<SparseAdam> Make(
      const torch::Dict<std::string, HashTablePtr> &hashtables, double lr,
      double beta1, double beta2, double eps, double weight_decay,
      bool amsgrad);
};
}  // namespace optim
}  // namespace recis
