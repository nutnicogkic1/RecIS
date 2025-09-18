#pragma once
#include <string>

#include "c10/util/flat_hash_map.h"
#include "embedding/hashtable.h"
namespace recis {
namespace serialize {
class LoadInfo {
 public:
  void Append(HashTablePtr ht);
  void Append(const std::string &tensor_name, at::Tensor tensor);
  const ska::flat_hash_map<
      std::string, ska::flat_hash_map<std::string, std::vector<std::string>>> &
  Infos() const;
  std::string Serialize();
  void Deserialize(std::string load_info);

 private:
  ska::flat_hash_map<std::string,
                     ska::flat_hash_map<std::string, std::vector<std::string>>>
      load_infos_;
};
}  // namespace serialize
}  // namespace recis