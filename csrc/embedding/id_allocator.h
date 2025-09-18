#pragma once
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/script.h>

#include "ATen/core/TensorBody.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/string_view.h"
#include "torch/types.h"

namespace recis {
namespace embedding {

class IdAllocator : public torch::CustomClassHolder {
 public:
  IdAllocator(torch::Device id_device, int64_t init_size,
              size_t free_block_size = 10240);
  torch::Tensor GenIds(size_t num_ids);
  void FreeIds(torch::Tensor ids);
  int64_t GetSize() { return cur_size_; };
  int64_t GetActiveSize() {
    TORCH_CHECK(cur_size_ >= free_size_);
    return cur_size_ - free_size_;
  };
  void Clear();
  ~IdAllocator();

 private:
  void ReduceBlock();
  void IncreaseBlock(int64_t increase_num);
  std::mutex mu_;
  torch::Device id_device_;
  int64_t cur_size_;
  size_t free_block_size_;
  int64_t free_size_;
  std::vector<torch::Tensor> free_blocks_;
};

}  // namespace embedding
}  // namespace recis
