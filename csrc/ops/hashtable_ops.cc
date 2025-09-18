#include "hashtable_ops.h"
namespace recis {
namespace functional {

struct IdGeneratorFunctor {
  IdGeneratorFunctor(int64_t gen_num, int64_t free_count, int64_t cur_count,
                     int64_t free_block_size,
                     std::vector<torch::Tensor>& free_blocks, int64_t* out_vec)
      : gen_num_(gen_num),
        free_count_(free_count),
        cur_count_(cur_count),
        free_block_size_(free_block_size),
        free_blocks_(free_blocks),
        out_vec_(out_vec) {}
  void operator()(const int64_t beg, const int64_t end) const {
    for (auto i : c10::irange(beg, end)) {
      auto free_index = free_count_ - gen_num_ + i;
      bool from_free = free_index >= 0;
      if (from_free) {
        auto block_index = free_index / free_block_size_;
        auto row_index = free_index % free_block_size_;
        out_vec_[i] = free_blocks_[block_index].data_ptr<int64_t>()[row_index];
      } else {
        out_vec_[i] = cur_count_ - (free_index + 1);
      }
    }
  }

 private:
  const int64_t gen_num_;
  const int64_t free_count_;
  const int64_t cur_count_;
  const int64_t free_block_size_;
  std::vector<torch::Tensor>& free_blocks_;
  int64_t* out_vec_;
};

struct IdFreeFunctor {
  IdFreeFunctor(int64_t free_count, int64_t free_block_size,
                std::vector<torch::Tensor>& free_blocks, int64_t* ids_vec)
      : free_count_(free_count),
        free_block_size_(free_block_size),
        free_blocks_(free_blocks),
        ids_vec_(ids_vec) {}
  void operator()(const int64_t beg, const int64_t end) const {
    for (auto i : c10::irange(beg, end)) {
      auto free_index = free_count_ + i;
      auto block_index = free_index / free_block_size_;
      auto row_index = free_index % free_block_size_;
      free_blocks_[block_index].data_ptr<int64_t>()[row_index] = ids_vec_[i];
    }
  }

 private:
  const int64_t free_count_;
  const int64_t free_block_size_;
  std::vector<torch::Tensor>& free_blocks_;
  int64_t* ids_vec_;
};

void boolean_mask_cpu_kernel(int64_t* output, const bool* mask,
                             const int64_t* select_index, const int64_t* input,
                             const int64_t output_size) {
  for (int64_t i = 0; i < output_size; ++i) {
    if (mask[i]) {
      output[i] = input[select_index[i] - 1];
    }
  }
}

torch::Tensor boolean_mask_op(torch::Tensor output, torch::Tensor mask,
                              torch::Tensor select_index, torch::Tensor input) {
  int64_t input_size = input.numel();
  int64_t output_size = output.numel();
  if (!input_size || !output_size) {
    return output;
  }

  if (output.device().is_cuda()) {
    boolean_mask_cuda_op(output, mask, select_index, input, output_size);

  } else {
    boolean_mask_cpu_kernel(output.data_ptr<int64_t>(), mask.data_ptr<bool>(),
                            select_index.data_ptr<int64_t>(),
                            input.data_ptr<int64_t>(), output_size);
  }

  return output;
}

void generate_ids_cpu_kernel(torch::Tensor output, const int64_t gen_num,
                             std::vector<torch::Tensor>& free_blocks,
                             const int64_t free_count, const int64_t cur_count,
                             const int64_t free_block_size) {
  IdGeneratorFunctor gen_functor(gen_num, free_count, cur_count,
                                 free_block_size, free_blocks,
                                 output.data_ptr<int64_t>());
  at::parallel_for(0, gen_num, 0, gen_functor);
}

torch::Tensor generate_ids_op(const int64_t gen_num,
                              std::vector<torch::Tensor> free_blocks,
                              const int64_t free_count, const int64_t cur_count,
                              const int64_t free_block_size) {
  auto output = torch::empty({gen_num}, torch::TensorOptions()
                                            .dtype(torch::kInt64)
                                            .device(free_blocks[0].device()));
  if (!gen_num) {
    return output;
  }

  if (output.device().is_cuda()) {
    generate_ids_cuda_op(output, gen_num, free_blocks, free_count, cur_count,
                         free_block_size);

  } else {
    generate_ids_cpu_kernel(output, gen_num, free_blocks, free_count, cur_count,
                            free_block_size);
  }

  return output;
}

void free_ids_cpu_kernel(torch::Tensor free_ids, const int64_t free_num,
                         std::vector<torch::Tensor>& free_blocks,
                         const int64_t free_count,
                         const int64_t free_block_size) {
  IdFreeFunctor free_functor(free_count, free_block_size, free_blocks,
                             free_ids.data_ptr<int64_t>());
  at::parallel_for(0, free_num, 0, free_functor);
}

void free_ids_op(torch::Tensor free_ids, std::vector<torch::Tensor> free_blocks,
                 const int64_t free_count, const int64_t free_block_size) {
  auto free_num = free_ids.numel();
  if (!free_num) {
    return;
  }
  if (free_ids.device().is_cuda()) {
    free_ids_cuda_op(free_ids, free_num, free_blocks, free_count,
                     free_block_size);

  } else {
    free_ids_cpu_kernel(free_ids, free_num, free_blocks, free_count,
                        free_block_size);
  }
  return;
}

void mask_key_index_cpu_kernel(const int64_t* in_keys, const bool* mask,
                               const int64_t* in_out_index, int64_t* out_keys,
                               int64_t* out_index, const int64_t in_size) {
  for (int64_t i = 0; i < in_size; ++i) {
    if (mask[i]) {
      out_keys[in_out_index[i] - 1] = in_keys[i];
      out_index[in_out_index[i] - 1] = i;
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor> mask_key_index_op(
    torch::Tensor in_keys, torch::Tensor mask, torch::Tensor in_out_index,
    int64_t out_size) {
  auto out_keys = torch::empty(
      {out_size},
      torch::TensorOptions().dtype(torch::kInt64).device(in_keys.device()));
  auto out_index = torch::empty_like(out_keys);
  int64_t input_size = in_keys.numel();
  if (!input_size || !out_size) {
    return std::make_tuple(out_keys, out_index);
  }

  if (in_keys.device().is_cuda()) {
    mask_key_index_cuda_op(in_keys, mask, in_out_index, out_keys, out_index,
                           input_size);

  } else {
    mask_key_index_cpu_kernel(
        in_keys.data_ptr<int64_t>(), mask.data_ptr<bool>(),
        in_out_index.data_ptr<int64_t>(), out_keys.data_ptr<int64_t>(),
        out_index.data_ptr<int64_t>(), input_size);
  }

  return std::make_tuple(out_keys, out_index);
}

void scatter_ids_with_mask_cpu_kernel(int64_t* out_ids, const int64_t* in_ids,
                                      const int64_t* mask_index,
                                      const int64_t in_size) {
  for (int64_t i = 0; i < in_size; ++i) {
    out_ids[mask_index[i]] = in_ids[i];
  }
}

torch::Tensor scatter_ids_with_mask_op(torch::Tensor out_ids,
                                       torch::Tensor in_ids,
                                       torch::Tensor mask_index) {
  int64_t input_size = in_ids.numel();
  int64_t output_size = out_ids.numel();
  if (!input_size || !output_size) {
    return out_ids;
  }

  if (out_ids.device().is_cuda()) {
    scatter_ids_with_mask_cuda_op(out_ids, in_ids, mask_index, input_size);

  } else {
    scatter_ids_with_mask_cpu_kernel(
        out_ids.data_ptr<int64_t>(), in_ids.data_ptr<int64_t>(),
        mask_index.data_ptr<int64_t>(), input_size);
  }

  return out_ids;
}
}  // namespace functional
}  // namespace recis
