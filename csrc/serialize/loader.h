#pragma once
#include <memory>
#include <string>
#include <unordered_map>

#include "ATen/core/TensorBody.h"
#include "ATen/core/ivalue.h"
#include "c10/util/flat_hash_map.h"
#include "c10/util/intrusive_ptr.h"
#include "embedding/hashtable.h"
#include "serialize/read_block.h"
namespace recis {
namespace serialize {
class Loader : public torch::CustomClassHolder {
 public:
  Loader(const std::string &path, int64_t parallel,
         torch::Dict<std::string, HashTablePtr> hts_to_load,
         torch::Dict<std::string, at::Tensor> tensors_to_load);
  /*
  {
    "dst_tensor_name":{"src_tensor_name": ["id", "emb", "xxx",...], ...},
  #sparse "dst_tensor_name": {"src_tensor_name":[""]"} #dense
  }
  */
  std::string DefaultLoadInfo();
  void Load(std::string load_info);
  ~Loader();

 private:
  int64_t parallel_;
  std::string path_;
  std::unordered_map<std::string, HashTablePtr> hts_to_load_;
  std::unordered_map<std::string, at::Tensor> tensors_to_load_;
};
}  // namespace serialize
}  // namespace recis
