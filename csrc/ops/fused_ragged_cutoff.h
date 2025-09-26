#pragma once
#include <tuple>

#include "ATen/core/TensorBody.h"
#include "c10/cuda/CUDAStream.h"
#include "torch/extension.h"

namespace recis {
namespace functional {

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor>
fused_ragged_cutoff_2D(std::vector<at::Tensor> value,
                       std::vector<at::Tensor> offset, at::Tensor keep_length,
                       at::Tensor drop_sides, at::Tensor pad_sides);

void fused_ragged_cutoff_2D_cuda_op(
    std::vector<at::Tensor> values, std::vector<at::Tensor> offsets,
    at::Tensor cutoff_values, at::Tensor cutoff_offsets, at::Tensor drop_nums,
    at::Tensor pad_nums, at::Tensor keep_lens, at::Tensor fea_offset,
    at::Tensor output_val_fea_offset, int32_t max_row_num, int32_t fea_num,
    at::Tensor cutoff_val_nums, at::Tensor sides, cudaStream_t stream);

std::tuple<at::Tensor, at::Tensor, at::Tensor> post_cutoff_lens_cuda_op(
    std::vector<at::Tensor> offsets, at::Tensor keep_lengths,
    at::Tensor fea_offset, int fea_num, int total_rows, int max_row_num,
    cudaStream_t stream);

std::tuple<at::Tensor, at::Tensor> seg_scan_cuda(at::Tensor fea_offset,
                                                 int num_fea, int total_rows,
                                                 at::Tensor lengths,
                                                 int max_row_num,
                                                 cudaStream_t stream);

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>>
fused_ragged_cutoff_3D(std::vector<at::Tensor> values,
                       std::vector<at::Tensor> outer_offsets,
                       std::vector<at::Tensor> inner_offsets,
                       at::Tensor keep_lengths, at::Tensor drop_sides,
                       at::Tensor pad_sides);

void fused_ragged_cutoff_3D_cuda_op(
    std::vector<at::Tensor> values, std::vector<at::Tensor> offsets,
    std::vector<at::Tensor> inner_offsets, at::Tensor cutoff_values,
    at::Tensor output_inner_offsets, at::Tensor fea_offset,
    at::Tensor output_inner_fea_offset, at::Tensor output_val_fea_offset,
    int32_t fea_num, at::Tensor drop_nums, at::Tensor drop_sides,
    at::Tensor pad_nums, at::Tensor pad_sides, at::Tensor keep_lens,
    cudaStream_t stream);

std::tuple<at::Tensor, at::Tensor, at::Tensor> seg_gen_offsets_cuda(
    at::Tensor fea_seq_offset, std::vector<at::Tensor> outer_offsets,
    std::vector<at::Tensor> inner_offsets, at::Tensor output_inner_fea_offset,
    at::Tensor drop_nums, at::Tensor drop_sides, at::Tensor pad_nums,
    at::Tensor pad_sides, at::Tensor keep_lengths, int fea_num, int total_seqs,
    int total_cutoff_rows, int max_row_num, cudaStream_t stream);

}  // namespace functional
}  // namespace recis
