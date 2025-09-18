#include "embedding/adamw.h"

#include <ATen/ParallelNative.h>
#include <math.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"
#include "ATen/ParallelFuture.h"
#include "ATen/SparseTensorImpl.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/function.h"
#include "ATen/core/ivalue.h"
#include "ATen/core/ivalue_inl.h"
#include "ATen/core/jit_type.h"
#include "ATen/ops/ones.h"
#include "ATen/ops/unique_consecutive.h"
#include "ATen/ops/zeros.h"
#include "ATen/record_function.h"
#include "c10/core/DeviceType.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/Exception.h"
#include "c10/util/StringUtil.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/irange.h"
#include "embedding/hashtable.h"
#include "embedding/initializer.h"
#include "embedding/optim.h"
#include "embedding/parallel_util.h"
#include "ops/block_apply_adamw_op.h"
#include "torch/types.h"

namespace recis {
namespace optim {
namespace {
at::SparseTensorImpl *get_sparse_impl(const at::Tensor &self) {
  return static_cast<at::SparseTensorImpl *>(self.unsafeGetTensorImpl());
}

void FusedSparseAdamW(torch::Tensor grad, SparseAdamWOptions &options,
                      SparseAdamWParamState &state) {
  auto table = state.hashtable();
  auto param = state.param();
  grad = grad.to(param->TensorOptions().device());
  auto block_size = param->BlockSize();
  auto exp_avg = state.exp_avg();
  auto exp_avg_sq = state.exp_avg_sq();
  auto beta1 = std::get<0>(options.betas());
  auto beta2 = std::get<1>(options.betas());

  auto weight_decay = 1 - options.lr() * options.weight_decay();
  auto eps = options.eps();

  auto index = get_sparse_impl(grad)->indices();
  auto grad_emb = get_sparse_impl(grad)->values();

  recis::functional::block_apply_adamw(
      index, grad_emb, (*param->Values()), state.beta1(), state.beta2(),
      state.step(), (*exp_avg->Values()), (*exp_avg_sq->Values()), options.lr(),
      beta1, beta2, weight_decay, eps, block_size);
}

std::string kParamNameSep{"sparse_adamw_"};
std::string ExpAvgName() {
  static std::string local_var = torch::str(kParamNameSep, "exp_avg");
  return local_var;
}
std::string ExpAvgSqName() {
  static std::string local_var = torch::str(kParamNameSep, "exp_avg_sq");
  return local_var;
}
const std::string StepName(const std::string &tensor_name) {
  return torch::str(kParamNameSep, "step");
}
const std::string BetaOneName(const std::string &tensor_name) {
  return torch::str(kParamNameSep, "beta1");
}
std::string BetaTwoName(const std::string &tensor_name) {
  return torch::str(kParamNameSep, "beta2");
}
}  // namespace

SparseAdamWOptions::SparseAdamWOptions(double lr) : lr_(lr) {}
double SparseAdamWOptions::get_lr() const { return lr_; }

void SparseAdamWOptions::set_lr(const double lr) { lr_ = lr; }

void SparseAdamW::step() {
  torch::NoGradGuard no_grad;
  std::unordered_map<std::string, std::pair<HashTablePtr, torch::Tensor>>
      var_grads;
  for (auto &group : param_groups_) {
    for (auto &it : group.params()) {
      auto &p = it.second;
      if (!p.defined()) {
        continue;
      }
      if (!p->has_grad()) {
        continue;
      }
      const auto &grad = p->grad(grad_accum_steps_);
      if (!grad.defined()) {
        continue;
      }
      TORCH_CHECK(grad.is_sparse(), "SparseAdamW only support sparse gradients" /*, please consider SparseAdamW instead*/);
      var_grads[it.first] = std::make_pair(p, grad);
    }
  }
  for (auto &group : param_groups_) {
    for (auto &it : group.params()) {
      auto var_grad = var_grads.find(it.first);
      if (var_grad == var_grads.end()) {
        continue;
      }
      auto &options = static_cast<SparseAdamWOptions &>(group.options());
      auto &state = static_cast<SparseAdamWParamState &>(
          *state_[var_grad->second.first.get()]);
      int64_t param_size = state.param()->Values()->size();
      TORCH_CHECK(
          param_size == state.exp_avg()->Values()->size() &&
              param_size == state.exp_avg_sq()->Values()->size(),
          "param size and adamw param size mismatch",
          ", param_size: ", param_size,
          ", adamw exp_avg size: ", state.exp_avg()->Values()->size(),
          ", adamw exp_avg_sq size: ", state.exp_avg_sq()->Values()->size());
      {
        RECORD_FUNCTION(
            torch::str("FusedSparseAdamW", "/", it.first, "/", "Update"),
            std::vector<c10::IValue>());
        FusedSparseAdamW(var_grad->second.second, options, state);
      }
    }
  }
}

void SparseAdamW::zero_grad() { SparseOptimizer::zero_grad(); }

void SparseAdamW::add_param_group(
    const SparseOptimizerParamGroup &param_group) {
  SparseOptimizerParamGroup param_group_(param_group.params());
  if (!param_group.has_options()) {
    param_group_.set_options(defaults_->clone());
  } else {
    param_group_.set_options(param_group.options().clone());
  }
  for (const auto &param : param_group_.params()) {
    InitParamState(param.first, param.second);
  }
  param_groups_.emplace_back(param_group_);
}

void SparseAdamW::add_parameters(
    const torch::Dict<std::string, HashTablePtr> &parameters) {
  auto &parameters_ = param_groups_[0].params();
  for (auto it = parameters.begin(); it != parameters.end(); it++) {
    InitParamState(it->key(), it->value());
    parameters_[it->key()] = it->value();
  }
}

void SparseAdamW::InitParamState(const std::string &param_name,
                                 HashTablePtr param) {
  TORCH_CHECK(state_.count(param.get()) == 0,
              "some parameters appear in more than one parameter group.");
  auto state = std::make_unique<SparseAdamWParamState>();
  auto slot_group = param->SlotGroup();
  auto emb_slot = slot_group->EmbSlot();
  // Exponential moving average of gradient values
  state->beta1(
      torch::ones({}, at::TensorOptions()
                          .dtype(state->beta_dtype())
                          .device(emb_slot->TensorOptions().device())));
  state->beta2(
      torch::ones({}, at::TensorOptions()
                          .dtype(state->beta_dtype())
                          .device(emb_slot->TensorOptions().device())));
  state->step(
      torch::zeros({1}, at::TensorOptions()
                            .dtype(state->step_dtype())
                            .device(emb_slot->TensorOptions().device())));

  state->param(emb_slot);
  state->exp_avg(slot_group->AppendSlot(ExpAvgName(), emb_slot->Dtype(),
                                        emb_slot->FullShape(1)));
  state->exp_avg_sq(slot_group->AppendSlot(ExpAvgSqName(), emb_slot->Dtype(),
                                           emb_slot->FullShape(1)));
  state->hashtable(param);
  state_[param.get()] = std::move(state);
}

const std::tuple<std::unordered_map<std::string, HashTablePtr>,
                 std::unordered_map<std::string, torch::Tensor>>
SparseAdamW::state_dict() {
  std::unordered_map<std::string, HashTablePtr> ret;
  std::unordered_map<std::string, torch::Tensor> steps;
  for (auto &group : param_groups_) {
    for (auto &it : group.params()) {
      auto &p = it.second;
      if (!p.defined()) {
        continue;
      }
      auto &param_name = it.first;
      auto &state = static_cast<SparseAdamWParamState &>(*state_[p.get()]);
      steps[StepName(param_name)] = state.step();
      steps[BetaOneName(param_name)] = state.beta1();
      steps[BetaTwoName(param_name)] = state.beta2();
    }
  }
  return std::make_tuple(ret, steps);
}

void SparseAdamW::load_state_dict(
    torch::Dict<std::string, HashTablePtr> hashtables,
    torch::Dict<std::string, torch::Tensor> steps) {}

c10::intrusive_ptr<SparseAdamW> SparseAdamW::Make(
    const torch::Dict<std::string, HashTablePtr> &hashtables, double lr,
    double beta1, double beta2, double eps, double weight_decay,
    bool use_nesterov) {
  LOG(WARNING) << at::get_parallel_info();
  SparseAdamWOptions option;
  option.lr(lr);
  option.betas(std::make_pair(beta1, beta2));
  option.eps(eps);
  option.weight_decay(weight_decay);
  option.use_nesterov(use_nesterov);
  std::unordered_map<std::string, HashTablePtr> input;
  for (auto it = hashtables.begin(); it != hashtables.end(); it++) {
    input[it->key()] = it->value();
  }
  return c10::make_intrusive<SparseAdamW>(input, option);
}
}  // namespace optim
}  // namespace recis
