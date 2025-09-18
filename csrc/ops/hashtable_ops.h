#include <torch/extension.h>

namespace recis {
namespace functional {

torch::Tensor boolean_mask_op(torch::Tensor output, torch::Tensor mask,
                              torch::Tensor select_index, torch::Tensor input);
void boolean_mask_cuda_op(torch::Tensor output, torch::Tensor mask,
                          torch::Tensor select_index, torch::Tensor input,
                          const int64_t output_size);

torch::Tensor generate_ids_op(const int64_t gen_num,
                              std::vector<torch::Tensor> free_blocks,
                              const int64_t free_count, const int64_t cur_count,
                              const int64_t free_block_size);
void generate_ids_cuda_op(torch::Tensor output, const int64_t gen_num,
                          std::vector<torch::Tensor>& free_blocks,
                          const int64_t free_count, const int64_t cur_count,
                          const int64_t free_block_size);

void free_ids_op(torch::Tensor free_ids, std::vector<torch::Tensor> free_blocks,
                 const int64_t free_count, const int64_t free_block_size);
void free_ids_cuda_op(torch::Tensor free_ids, const int64_t free_num,
                      std::vector<torch::Tensor>& free_blocks,
                      const int64_t free_count, const int64_t free_block_size);

std::tuple<torch::Tensor, torch::Tensor> mask_key_index_op(
    torch::Tensor in_keys, torch::Tensor mask, torch::Tensor out_index,
    int64_t out_size);
void mask_key_index_cuda_op(torch::Tensor in_keys, torch::Tensor mask,
                            torch::Tensor in_out_index, torch::Tensor out_keys,
                            torch::Tensor out_index, int64_t in_size);

torch::Tensor scatter_ids_with_mask_op(torch::Tensor out_ids,
                                       torch::Tensor in_ids,
                                       torch::Tensor mask_index);
void scatter_ids_with_mask_cuda_op(torch::Tensor out_ids, torch::Tensor in_ids,
                                   torch::Tensor mask_index,
                                   const int64_t in_size);

}  // namespace functional
}  // namespace recis
