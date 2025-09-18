#pragma once
#include <c10/util/intrusive_ptr.h>

#include <string>
#include <unordered_map>

#include "c10/core/ScalarType.h"
#include "nlohmann/json.hpp"
namespace recis {
namespace serialize {

class IndexInfo : public at::intrusive_ptr_target {
 public:
  static at::intrusive_ptr<IndexInfo> Make();
  void Append(const std::string &tensor_name, const std::string &slice_info,
              const std::string &sep_info, const std::string &file_name);
  void MergeFrom(IndexInfo &rhv);

  int64_t FileNum() const { return file_index_map_.size(); }
  std::vector<std::string> SliceInfoOfTensor(const std::string &tensor_name);
  bool HasBlock(const std::string &tensor_name, const std::string &slice_info);
  bool HashTensor(const std::string &tensor_name);
  std::vector<std::string> ListTensor();
  int64_t FileIndexOfBlock(const std::string &block_name);
  const std::string FileNameByIndex(int64_t index);

  std::string Serialize();
  void Deserialize(const std::string &info_buf);

  void SetTensorNameMap(
      const std::unordered_map<std::string, std::string> &tensor_name_map);

 private:
  std::unordered_map<std::string, int64_t> file_index_map_;
  std::unordered_map<int64_t, std::string> index_file_map_;
  std::unordered_map<std::string, int64_t> block_index_map_;
  const std::unordered_map<std::string, std::string> *tensor_name_map_ =
      nullptr;

  static std::string kBlockKey;
  static std::string KFileKey;
};

}  // namespace serialize
}  // namespace recis
