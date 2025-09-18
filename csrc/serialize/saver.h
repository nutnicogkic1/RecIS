#pragma once
#include "ATen/core/TensorBody.h"
#include "ATen/core/ivalue.h"
#include "c10/util/intrusive_ptr.h"
#include "embedding/hashtable.h"
#include "serialize/table_writer.h"
namespace recis {
namespace serialize {

class Saver : public torch::CustomClassHolder {
 public:
  Saver(int64_t shard_index, int64_t shard_num, int64_t parallel,
        const std::string &path);
  std::vector<at::intrusive_ptr<WriteBlock>> MakeWriteBlocks(
      const torch::Dict<std::string, HashTablePtr> &hashtables,
      const torch::Dict<std::string, torch::Tensor> &tensors);
  void Save(std::vector<at::intrusive_ptr<WriteBlock>> write_blocks);

 private:
  int64_t parallel_;
  int64_t shard_index_;
  int64_t shard_num_;
  const std::string path_;
};
}  // namespace serialize
}  // namespace recis