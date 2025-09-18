#include "fused_ragged_cutoff.h"

#include <ATen/Parallel.h>

#include <algorithm>
#include <iostream>
#include <tuple>

#include "c10/core/DeviceType.h"
#include "c10/core/TensorImpl.h"
#include "c10/cuda/CUDAStream.h"
#include "c10/util/Exception.h"
#include "c10/util/irange.h"
#include "ragged_common.cuh"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/extension.h"
#include "torch/types.h"

namespace recis {
namespace functional {

// all the input tensor should be on gpu
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor>
fused_ragged_cutoff(std::vector<at::Tensor> values,
                    std::vector<at::Tensor> offsets, at::Tensor keep_lengths,
                    at::Tensor drop_sides, at::Tensor pad_sides) {
  TORCH_CHECK(all_cuda(values),
              "fused_ragged_cutoff: each value should be placed on gpu");
  TORCH_CHECK(all_cuda(offsets),
              "fused_ragged_cutoff: each offsets should be placed on gpu");
  TORCH_CHECK(all_cuda({keep_lengths}),
              "fused_ragged_cutoff: keep_lengths should be placed on gpu");
  TORCH_CHECK(all_cuda({drop_sides}),
              "fused_ragged_cutoff: drop_sides should be placed on gpu");
  TORCH_CHECK(all_cuda({pad_sides}),
              "fused_ragged_cutoff: pad_sides should be placed on gpu");
  TORCH_CHECK(all_same_type(values),
              "fused_ragged_cutoff: each value should be the same type");
  TORCH_CHECK(all_same_type(offsets, torch::kInt32) ||
                  all_same_type(offsets, torch::kInt64),
              "fused_ragged_cutoff: each offsets should be torch::kInt32 "
              "or torch::kInt64");
  TORCH_CHECK(
      all_same_type({drop_sides, pad_sides}, torch::kBool),
      "fused_ragged_cutoff: drop_sides and pad_sides should be torch::kBool");

  auto stream = c10::cuda::getCurrentCUDAStream();
  int fea_num = values.size();
  TORCH_CHECK(keep_lengths.numel() == fea_num,
              "keep_lengths.numel() = ", keep_lengths.numel(),
              " but is expected equals fea_num =", fea_num);
  TORCH_CHECK(offsets.size() == fea_num, "offsets.numel() = ", offsets.size(),
              " but is expected equals fea_num =", fea_num);

  int max_row_num = 0;
  // fea_offset is used both on cpu and gpu
  std::vector<int> fea_offset(fea_num + 1);
  for (int i = 1; i <= fea_num; ++i) {
    int row = offsets[i - 1].numel() - 1;
    max_row_num = std::max(max_row_num, row);
    fea_offset[i] = fea_offset[i - 1] + row;
  }
  int total_rows = fea_offset[fea_num];

  // aten::copy: H2D. 30us
  at::Tensor fea_offset_cuda =
      torch::from_blob(fea_offset.data(), fea_offset.size(),
                       torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::kCUDA);
  at::Tensor keep_lens_cuda = keep_lengths.to(torch::kCUDA);

  at::Tensor cutoff_lens, drop_nums, pad_nums;
  std::tie(cutoff_lens, drop_nums, pad_nums) =
      post_cutoff_lens_cuda_op(offsets, keep_lens_cuda, fea_offset_cuda,
                               fea_num, total_rows, max_row_num, stream);

  at::Tensor cutoff_offsets, cutoff_val_nums;
  std::tie(cutoff_offsets, cutoff_val_nums) = seg_scan_cuda(
      fea_offset_cuda, fea_num, total_rows, cutoff_lens, max_row_num, stream);
  // cutoff_offsets: offsets of each feature after cutoff
  // exclusive offsets + inslusive offsets
  // cutoff_val_nums: value nums of each feature after cutoff

  std::vector<at::Tensor> drop_nums_vec(fea_num);
  std::vector<at::Tensor> pad_nums_vec(fea_num);

  at::Tensor cutoff_val_nums_cpu = cutoff_val_nums.to(torch::kCPU).contiguous();
  auto cutoff_val_nums_cpu_acc =
      cutoff_val_nums_cpu.accessor<int, 1>();  // around 100
  int global_size =
      std::accumulate(cutoff_val_nums_cpu.data_ptr<int>(),
                      cutoff_val_nums_cpu.data_ptr<int>() + fea_num, 0);
  // output_val_fea_offset: Prefix sum of the number of values actually used for
  // each feature after cutoff, used to locate where to write the source data
  // into the output buffer later (used both on cpu and gpu)
  at::Tensor output_val_fea_offset =
      at::empty({fea_num + 1}, offsets.front().options().device(torch::kCPU));
  auto output_val_fea_offset_acc = output_val_fea_offset.accessor<int, 1>();
  for (int i = 0; i < fea_num + 1; ++i) {
    if (i == 0) {
      output_val_fea_offset_acc[i] = 0;
      continue;
    }
    output_val_fea_offset_acc[i] =
        output_val_fea_offset_acc[i - 1] + cutoff_val_nums_cpu_acc[i - 1];
  }
  at::Tensor output_val_fea_offset_cuda =
      output_val_fea_offset.to(torch::kCUDA);
  at::Tensor cutoff_values;

  if (!drop_nums.eq(0).all().item<bool>()) {
    cutoff_values =
        at::empty({global_size}, values.front().options().device(torch::kCUDA));
    fused_ragged_cutoff_2D_cuda_op(
        values, offsets, cutoff_values, cutoff_offsets, drop_nums, pad_nums,
        keep_lens_cuda, fea_offset_cuda, output_val_fea_offset_cuda,
        max_row_num, fea_num, cutoff_val_nums, drop_sides, stream);
  } else {
    cutoff_values = at::cat(values, 0);
  }

  std::vector<at::Tensor> ragged_val_vec(fea_num);
  std::vector<at::Tensor> ragged_offsets_vec(fea_num);

  at::parallel_for(0, fea_num, 1, [&](int64_t beg, int64_t end) {
    for (int i = beg; i < end; ++i) {
      drop_nums_vec[i] = drop_nums.slice(0, fea_offset[i], fea_offset[i + 1]);
      pad_nums_vec[i] = pad_nums.slice(0, fea_offset[i], fea_offset[i + 1]);
      ragged_offsets_vec[i] =
          cutoff_offsets.slice(0, fea_offset[i] + i, fea_offset[i + 1] + i + 1);
      ragged_val_vec[i] = cutoff_values.slice(0, output_val_fea_offset_acc[i],
                                              output_val_fea_offset_acc[i + 1]);
    }
  });

  return std::make_tuple(ragged_val_vec, ragged_offsets_vec, drop_nums_vec,
                         pad_nums_vec, drop_sides, pad_sides);
}

// all the input tensor should be on gpu
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>>
fused_ragged_cutoff_3D(std::vector<at::Tensor> values,
                       std::vector<at::Tensor> outer_offsets,
                       std::vector<at::Tensor> inner_offsets,
                       at::Tensor keep_lengths, at::Tensor drop_sides,
                       at::Tensor pad_sides) {
  TORCH_CHECK(all_cuda(values),
              "fused_ragged_cutoff_3D: each value should be placed on gpu");
  TORCH_CHECK(all_cuda(inner_offsets),
              "fused_ragged_cutoff_3D: each offsets should be placed on gpu");
  TORCH_CHECK(all_cuda(outer_offsets),
              "fused_ragged_cutoff_3D: each offsets should be placed on gpu");
  TORCH_CHECK(all_cuda({keep_lengths}),
              "fused_ragged_cutoff_3D: keep_lengths should be placed on gpu");
  TORCH_CHECK(all_cuda({drop_sides}),
              "fused_ragged_cutoff_3D: drop_sides should be placed on gpu");
  TORCH_CHECK(all_cuda({pad_sides}),
              "fused_ragged_cutoff_3D: pad_sides should be placed on gpu");
  TORCH_CHECK(all_same_type(values),
              "fused_ragged_cutoff_3D: each value should be the same type");
  TORCH_CHECK(
      all_same_type(inner_offsets, torch::kInt32) ||
          all_same_type(inner_offsets, torch::kInt64),
      "fused_ragged_cutoff_3D: each inner offsets should be torch::kInt32 "
      "or torch::kInt64");
  TORCH_CHECK(
      all_same_type(outer_offsets, torch::kInt32) ||
          all_same_type(outer_offsets, torch::kInt64),
      "fused_ragged_cutoff_3D: each outer offsets should be torch::kInt32 "
      "or torch::kInt64");
  TORCH_CHECK(all_same_type({drop_sides, pad_sides}, torch::kBool),
              "fused_ragged_cutoff_3D: drop_sides and pad_sides should be "
              "torch::kBool");

  auto stream = c10::cuda::getCurrentCUDAStream();
  int fea_num = values.size();
  TORCH_CHECK(keep_lengths.numel() == fea_num,
              "keep_lengths.numel() = ", keep_lengths.numel(),
              " but is expected equals fea_num =", fea_num);
  TORCH_CHECK(outer_offsets.size() == fea_num,
              "outer_offsets.numel() = ", outer_offsets.size(),
              " but is expected equals fea_num =", fea_num);

  int max_seq_num = 0;
  std::vector<int> fea_seq_offset(fea_num + 1);
  for (int i = 1; i <= fea_num; ++i) {
    int seq = outer_offsets[i - 1].numel() - 1;
    max_seq_num = std::max(max_seq_num, seq);
    fea_seq_offset[i] = fea_seq_offset[i - 1] + seq;
  }
  int total_seqs = fea_seq_offset[fea_num];
  at::Tensor fea_seq_offset_cuda =
      torch::from_blob(fea_seq_offset.data(), fea_seq_offset.size(),
                       torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::kCUDA);
  at::Tensor keep_lens_cuda = keep_lengths.to(torch::kCUDA);

  std::vector<int> output_inner_fea_offset(fea_num + 1);
  auto keep_lengths_cpu = keep_lengths.to(torch::kCPU);
  auto keep_length_cpu_acc = keep_lengths_cpu.accessor<int, 1>();
  int max_cutoff_rows = 0;
  for (int i = 1; i <= fea_num; ++i) {
    int seq = outer_offsets[i - 1].numel() - 1;
    int cutoff_row = seq * keep_length_cpu_acc[i - 1];
    max_cutoff_rows = std::max(max_cutoff_rows, cutoff_row);
    output_inner_fea_offset[i] = output_inner_fea_offset[i - 1] + cutoff_row;
  }
  int total_cutoff_rows = output_inner_fea_offset[fea_num];
  at::Tensor output_inner_fea_offset_cuda =
      torch::from_blob(output_inner_fea_offset.data(),
                       output_inner_fea_offset.size(),
                       torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::kCUDA, true);

  at::Tensor cutoff_lens, drop_nums, pad_nums;
  std::tie(cutoff_lens, drop_nums, pad_nums) = post_cutoff_lens_cuda_op(
      outer_offsets, keep_lens_cuda, fea_seq_offset_cuda, fea_num, total_seqs,
      max_seq_num, stream);

  at::Tensor output_outer_offsets, output_inner_offsets, max_offset_seg;
  std::tie(output_outer_offsets, output_inner_offsets, max_offset_seg) =
      seg_gen_offsets_cuda(fea_seq_offset_cuda, outer_offsets, inner_offsets,
                           output_inner_fea_offset_cuda, drop_nums, drop_sides,
                           pad_nums, pad_sides, keep_lengths, fea_num,
                           total_seqs, total_cutoff_rows, max_cutoff_rows,
                           stream);

  // TODO: seq_num * feature number: put on gpu
  at::Tensor cutoff_val_nums_cpu = max_offset_seg.to(torch::kCPU);
  auto cutoff_val_nums_cpu_acc = cutoff_val_nums_cpu.accessor<int, 1>();
  at::Tensor output_val_fea_offset = at::empty(
      {fea_num + 1}, inner_offsets.front().options().device(torch::kCPU));
  auto output_val_fea_offset_acc = output_val_fea_offset.accessor<int, 1>();
  for (int i = 0; i < fea_num + 1; ++i) {
    if (i == 0) {
      output_val_fea_offset_acc[i] = 0;
      continue;
    }
    output_val_fea_offset_acc[i] =
        output_val_fea_offset_acc[i - 1] + cutoff_val_nums_cpu_acc[i - 1];
  }
  int32_t global_size = output_val_fea_offset_acc[fea_num];
  at::Tensor output_val_fea_offset_cuda =
      output_val_fea_offset.to(torch::kCUDA);
  at::Tensor cutoff_values;

  if (!drop_nums.eq(0).all().item<bool>()) {
    cutoff_values =
        at::empty({global_size}, values.front().options().device(torch::kCUDA));
    fused_ragged_cutoff_3D_cuda_op(
        values, outer_offsets, inner_offsets, cutoff_values,
        output_inner_offsets, fea_seq_offset_cuda, output_inner_fea_offset_cuda,
        output_val_fea_offset_cuda, fea_num, drop_nums, drop_sides, pad_nums,
        pad_sides, keep_lens_cuda, stream);
  } else {
    cutoff_values = at::cat(values, 0);
  }

  std::vector<at::Tensor> ragged_val_vec(fea_num);
  std::vector<at::Tensor> ragged_outer_offsets_vec(fea_num);
  std::vector<at::Tensor> ragged_inner_offsets_vec(fea_num);

  at::parallel_for(
      /* begin */ 0,
      /* end   */ fea_num,
      /* grain_size */ 1, [&](int64_t beg, int64_t end) {
        for (int i = beg; i < end; ++i) {
          ragged_outer_offsets_vec[i] = output_outer_offsets.slice(
              0, fea_seq_offset[i] + i, fea_seq_offset[i + 1] + i + 1);
          ragged_inner_offsets_vec[i] = output_inner_offsets.slice(
              0, output_inner_fea_offset[i] + i,
              output_inner_fea_offset[i + 1] + i + 1);
          ragged_val_vec[i] =
              cutoff_values.slice(0, output_val_fea_offset_acc[i],
                                  output_val_fea_offset_acc[i + 1]);
        }
      });

  return std::make_tuple(ragged_val_vec, ragged_outer_offsets_vec,
                         ragged_inner_offsets_vec);
}

}  // namespace functional
}  // namespace recis
