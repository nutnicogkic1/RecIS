#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>

#include "cuda/cuda_param.cuh"
#include "cuda/utils.cuh"

namespace recis {
namespace functional {

template <typename scalar_t>
void __device__ apply_adamw_kernel(scalar_t& emb, scalar_t& exp_avg,
                                   scalar_t& exp_avg_sq, scalar_t& grad,
                                   const scalar_t alpha, const scalar_t beta1,
                                   const scalar_t beta2,
                                   const scalar_t weight_decay,
                                   const scalar_t eps) {
  emb = emb * weight_decay;
  exp_avg = exp_avg * beta1 + grad * (1 - beta1);
  exp_avg_sq = exp_avg_sq * beta2 + (1 - beta2) * grad * grad;
  emb = emb + alpha * (exp_avg / (sqrtf(exp_avg_sq) + eps));
}

template <typename scalar_t, typename beta_t, typename pack_t>
__global__ void block_apply_adamw_cuda_kernel(
    const int64_t* index_vec, scalar_t* grad, scalar_t** emb_blocks,
    scalar_t** exp_avg, scalar_t** exp_avg_sq, beta_t* alpha, int64_t num_ids,
    int64_t embedding_dim, int64_t block_size, scalar_t beta1, scalar_t beta2,
    scalar_t weight_decay, scalar_t eps, int64_t id_tile_size,
    int64_t emb_tile_size) {
  int64_t block_idx = blockIdx.x * id_tile_size;
  int64_t emb_idx = threadIdx.x * emb_tile_size;
  int64_t idx = block_idx + threadIdx.y;
  if (idx >= num_ids || emb_idx >= embedding_dim) return;

  auto index = index_vec[idx];
  auto block_index = index / block_size;
  auto row_offset = index % block_size * embedding_dim;
  if (emb_idx + emb_tile_size <= embedding_dim) {
    pack_t pack_emb =
        *(pack_t*)(*(emb_blocks + block_index) + row_offset + emb_idx);
    pack_t pack_avg =
        *(pack_t*)(*(exp_avg + block_index) + row_offset + emb_idx);
    pack_t pack_avg_sq =
        *(pack_t*)(*(exp_avg_sq + block_index) + row_offset + emb_idx);
    pack_t pack_g = *(pack_t*)(grad + idx * embedding_dim + emb_idx);
    for (auto i = 0; i < emb_tile_size; ++i) {
      apply_adamw_kernel(
          *((scalar_t*)(&pack_emb) + i), *((scalar_t*)(&pack_avg) + i),
          *((scalar_t*)(&pack_avg_sq) + i), *((scalar_t*)(&pack_g) + i),
          static_cast<scalar_t>(alpha[0]), beta1, beta2, weight_decay, eps);
    }
    *(pack_t*)(*(emb_blocks + block_index) + row_offset + emb_idx) = pack_emb;
    *(pack_t*)(*(exp_avg + block_index) + row_offset + emb_idx) = pack_avg;
    *(pack_t*)(*(exp_avg_sq + block_index) + row_offset + emb_idx) =
        pack_avg_sq;
  } else {
    for (auto i = 0; i < embedding_dim - emb_idx; ++i) {
      scalar_t emb = emb_blocks[block_index][row_offset + emb_idx + i];
      scalar_t avg = exp_avg[block_index][row_offset + emb_idx + i];
      scalar_t avg_sq = exp_avg_sq[block_index][row_offset + emb_idx + i];
      scalar_t g = grad[idx * embedding_dim + emb_idx + i];
      apply_adamw_kernel(emb, avg, avg_sq, g, static_cast<scalar_t>(alpha[0]),
                         beta1, beta2, weight_decay, eps);
      emb_blocks[block_index][row_offset + emb_idx + i] = emb;
      exp_avg[block_index][row_offset + emb_idx + i] = avg;
      exp_avg_sq[block_index][row_offset + emb_idx + i] = avg_sq;
    }
  }
}

#define BLOCK_APPLY_ADAMW_TF_LAUNCH_KERNEL(scalar_t, beta_t, pack_t)        \
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();                   \
  block_apply_adamw_cuda_kernel<scalar_t, beta_t, pack_t>                   \
      <<<grids, blocks, 0, at::cuda::getCurrentCUDAStream()>>>(             \
          index_vec, grad, emb_blocks, exp_avg, exp_avg_sq, alpha, num_ids, \
          embedding_dim, block_size, beta1, beta2, weight_decay, eps,       \
          id_tile_size, emb_tile_size);                                     \
  cudaStreamSynchronize(stream);

template <typename scalar_t, typename beta_t>
void block_apply_adamw_kernel_launcher(
    const int64_t* index_vec, scalar_t* grad, scalar_t** emb_blocks,
    scalar_t** exp_avg, scalar_t** exp_avg_sq, beta_t* alpha, int64_t num_ids,
    int64_t embedding_dim, int64_t block_size, scalar_t beta1, scalar_t beta2,
    scalar_t weight_decay, scalar_t eps) {
  int64_t emb_tile_size, emb_thread_size, id_tile_size, id_blocks,
      real_pack_size;
  recis::cuda::cal_pack_sizes<scalar_t>(num_ids, embedding_dim, emb_tile_size,
                                        emb_thread_size, id_tile_size,
                                        id_blocks, real_pack_size);
  dim3 grids(id_blocks);
  dim3 blocks(emb_thread_size, id_tile_size);
  if (real_pack_size == 2) {
    BLOCK_APPLY_ADAMW_TF_LAUNCH_KERNEL(scalar_t, beta_t, scalar_t);
  } else if (real_pack_size == 4) {
    BLOCK_APPLY_ADAMW_TF_LAUNCH_KERNEL(scalar_t, beta_t, float);
  } else if (real_pack_size == 8) {
    BLOCK_APPLY_ADAMW_TF_LAUNCH_KERNEL(scalar_t, beta_t, float2);
  } else if (real_pack_size == 16) {
    BLOCK_APPLY_ADAMW_TF_LAUNCH_KERNEL(scalar_t, beta_t, float4);
  } else {
    TORCH_CHECK(false, "block apply adamw cuda kernel error pack size");
  }
}

#undef BLOCK_APPLY_ADAMW_TF_LAUNCH_KERNEL

template <typename beta_t>
__global__ void cal_alpha_cuda_kernel(beta_t* beta1_t, beta_t* beta2_t,
                                      beta_t* alpha, double lr, double beta1,
                                      double beta2) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 1) return;
  beta1_t[0] *= beta1;
  beta2_t[0] *= beta2;
  auto bias_correction1 = 1 - beta1_t[0];
  auto bias_correction2 = 1 - beta2_t[0];
  alpha[0] = -lr / bias_correction1 * sqrtf(bias_correction2);
}

void block_apply_adamw_gpu(const torch::Tensor index, const torch::Tensor grad,
                           std::vector<torch::Tensor> emb_blocks,
                           torch::Tensor beta1_t, torch::Tensor beta2_t,
                           torch::Tensor alpha_t, torch::Tensor step,
                           std::vector<torch::Tensor> exp_avg,
                           std::vector<torch::Tensor> exp_avg_sq, double lr,
                           double beta1, double beta2, double weight_decay,
                           double eps, int64_t block_size) {
  TORCH_CHECK(index.device().type() == torch::kCUDA,
              "Input must be on CUDA device");
  int64_t num_ids = index.numel();
  int embedding_dim = emb_blocks[0].size(1);
  auto block_num = emb_blocks.size();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, beta1_t.scalar_type(),
      "calculate_adamw_alpha_cuda_impl", ([&] {
        using beta_t = scalar_t;
        cal_alpha_cuda_kernel<beta_t><<<1, 1, 0, stream>>>(
            beta1_t.data_ptr<beta_t>(), beta2_t.data_ptr<beta_t>(),
            alpha_t.data_ptr<beta_t>(), lr, beta1, beta2);
        if (num_ids == 0) return;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, grad.scalar_type(),
            "apply_adamw_cuda_impl", ([&] {
              recis::cuda::CudaVecParam<scalar_t*> emb_blocks_ptrs(block_num,
                                                                   stream);
              recis::cuda::CudaVecParam<scalar_t*> exp_avg_ptrs(block_num,
                                                                stream);
              recis::cuda::CudaVecParam<scalar_t*> exp_avg_sq_ptrs(block_num,
                                                                   stream);
              for (auto i = 0; i < block_num; ++i) {
                emb_blocks_ptrs[i] = emb_blocks[i].data_ptr<scalar_t>();
                exp_avg_ptrs[i] = exp_avg[i].data_ptr<scalar_t>();
                exp_avg_sq_ptrs[i] = exp_avg_sq[i].data_ptr<scalar_t>();
              }
              block_apply_adamw_kernel_launcher<scalar_t, beta_t>(
                  index.data_ptr<int64_t>(), grad.data_ptr<scalar_t>(),
                  (scalar_t**)(emb_blocks_ptrs.data()),
                  (scalar_t**)(exp_avg_ptrs.data()),
                  (scalar_t**)(exp_avg_sq_ptrs.data()),
                  alpha_t.data_ptr<beta_t>(), num_ids, embedding_dim,
                  block_size, static_cast<scalar_t>(beta1),
                  static_cast<scalar_t>(beta2),
                  static_cast<scalar_t>(weight_decay),
                  static_cast<scalar_t>(eps));
            }));
      }));
}

}  // namespace functional
}  // namespace recis
