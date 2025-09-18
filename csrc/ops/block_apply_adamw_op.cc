#include "block_apply_adamw_op.h"

#include <cmath>

namespace recis {
namespace functional {

template <class TEmb, class TBeta>
struct BlocksApplyAdamwTFFunctor {
  BlocksApplyAdamwTFFunctor(int64_t embedding_dim, int64_t block_size,
                            const int64_t *index_vec,
                            std::vector<torch::Tensor> &emb_blocks, TEmb *grad,
                            TBeta *alpha, std::vector<torch::Tensor> &exp_avg,
                            std::vector<torch::Tensor> &exp_avg_sq,
                            double beta1, double beta2, double weight_decay,
                            double eps)
      : embedding_dim_(embedding_dim),
        block_size_(block_size),
        index_vec_(index_vec),
        emb_blocks_(emb_blocks),
        grad_(grad),
        alpha_(alpha),
        exp_avg_(exp_avg),
        exp_avg_sq_(exp_avg_sq),
        beta1_(beta1),
        one_min_beta1_(1 - beta1),
        one_min_beta2_(1 - beta2),
        beta2_(beta2),
        weight_decay_(weight_decay),
        eps_(eps) {}
  void operator()(const int64_t beg, const int64_t end) const {
    for (auto i : c10::irange(beg, end)) {
      auto index = index_vec_[i];
      auto block_index = index / block_size_;
      auto row_index = index % block_size_;
      auto offset = row_index * embedding_dim_;

      auto emb_vec = emb_blocks_[block_index].data_ptr<TEmb>() + offset;
      auto exp_avg_sq_vec = exp_avg_sq_[block_index].data_ptr<TEmb>() + offset;
      auto exp_avg_vec = exp_avg_[block_index].data_ptr<TEmb>() + offset;
      auto grad_vec = grad_ + i * embedding_dim_;
      for (auto element_index : c10::irange(embedding_dim_)) {
        auto &emb_elem = emb_vec[element_index];
        // weight decay
        emb_elem = emb_elem * weight_decay_;
        auto grad_elem = grad_vec[element_index];

        auto &exp_avg_elem = exp_avg_vec[element_index];
        exp_avg_elem = exp_avg_elem * beta1_ + grad_elem * one_min_beta1_;

        auto &exp_avg_sq_elem = exp_avg_sq_vec[element_index];
        exp_avg_sq_elem =
            exp_avg_sq_elem * beta2_ + one_min_beta2_ * grad_elem * grad_elem;
        emb_elem +=
            alpha_[0] * (exp_avg_elem / (sqrtf(exp_avg_sq_elem) + eps_));
      }
    }
  }

 private:
  const int64_t embedding_dim_;
  const int64_t block_size_;
  const int64_t *index_vec_;
  std::vector<torch::Tensor> &emb_blocks_;
  TEmb *grad_;
  TBeta *alpha_;
  std::vector<torch::Tensor> &exp_avg_;
  std::vector<torch::Tensor> &exp_avg_sq_;
  const double beta1_;
  const double beta2_;
  const double one_min_beta1_;
  const double one_min_beta2_;
  const double weight_decay_;
  const double eps_;
};

template <typename beta_t>
void cal_alpha_cpu_kernel(beta_t *beta1_t, beta_t *beta2_t, beta_t *alpha,
                          double lr, double beta1, double beta2) {
  beta1_t[0] *= beta1;
  beta2_t[0] *= beta2;
  auto bias_correction1 = 1 - beta1_t[0];
  auto bias_correction2 = 1 - beta2_t[0];
  alpha[0] = -lr / bias_correction1 * sqrtf(bias_correction2);
}

void block_apply_adamw_cpu(const torch::Tensor index, const torch::Tensor grad,
                           std::vector<torch::Tensor> emb_blocks,
                           torch::Tensor beta1_t, torch::Tensor beta2_t,
                           torch::Tensor alpha_t, torch::Tensor step,
                           std::vector<torch::Tensor> exp_avg,
                           std::vector<torch::Tensor> exp_avg_sq, double lr,
                           double beta1, double beta2, double weight_decay,
                           double eps, int64_t block_size) {
  int64_t embedding_dim = emb_blocks[0].size(1);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      beta1_t.scalar_type(), "calculate_adamw_alpha_cpu_impl", ([&] {
        using beta_t = scalar_t;
        cal_alpha_cpu_kernel<beta_t>(
            beta1_t.data_ptr<beta_t>(), beta2_t.data_ptr<beta_t>(),
            alpha_t.data_ptr<beta_t>(), lr, beta1, beta2);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            grad.scalar_type(), "apply_adamw_cpu_impl", ([&] {
              BlocksApplyAdamwTFFunctor<scalar_t, beta_t> apply_functor(
                  embedding_dim, block_size, index.data_ptr<int64_t>(),
                  emb_blocks, grad.data_ptr<scalar_t>(),
                  alpha_t.data_ptr<beta_t>(), exp_avg, exp_avg_sq, beta1, beta2,
                  weight_decay, eps);
              at::parallel_for(0, index.numel(), 0, apply_functor);
            }));
      }));
}

void block_apply_adamw(const torch::Tensor index, const torch::Tensor grad,
                       std::vector<torch::Tensor> emb_blocks,
                       torch::Tensor beta1_t, torch::Tensor beta2_t,
                       torch::Tensor step, std::vector<torch::Tensor> exp_avg,
                       std::vector<torch::Tensor> exp_avg_sq, double lr,
                       double beta1, double beta2, double weight_decay,
                       double eps, int64_t block_size) {
  step.add(1);
  auto alpha_t = torch::empty_like(beta1_t);
  if (index.device().type() == torch::kCUDA) {
    block_apply_adamw_gpu(index, grad, emb_blocks, beta1_t, beta2_t, alpha_t,
                          step, exp_avg, exp_avg_sq, lr, beta1, beta2,
                          weight_decay, eps, block_size);
  } else {
    block_apply_adamw_cpu(index, grad, emb_blocks, beta1_t, beta2_t, alpha_t,
                          step, exp_avg, exp_avg_sq, lr, beta1, beta2,
                          weight_decay, eps, block_size);
  }
}

}  // namespace functional
}  // namespace recis
