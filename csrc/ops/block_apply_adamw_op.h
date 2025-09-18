#include <torch/extension.h>

namespace recis {
namespace functional {

void block_apply_adamw(const torch::Tensor index, const torch::Tensor grad,
                       std::vector<torch::Tensor> emb_blocks,
                       torch::Tensor beta1_t, torch::Tensor beta2_t,
                       torch::Tensor step, std::vector<torch::Tensor> exp_avg,
                       std::vector<torch::Tensor> exp_avg_sq, double lr,
                       double beta1, double beta2, double weight_decay,
                       double eps, int64_t block_size);

void block_apply_adamw_gpu(const torch::Tensor index, const torch::Tensor grad,
                           std::vector<torch::Tensor> emb_blocks,
                           torch::Tensor beta1_t, torch::Tensor beta2_t,
                           torch::Tensor alpha_t, torch::Tensor step,
                           std::vector<torch::Tensor> exp_avg,
                           std::vector<torch::Tensor> exp_avg_sq, double lr,
                           double beta1, double beta2, double weight_decay,
                           double eps, int64_t block_size);

}  // namespace functional
}  // namespace recis
