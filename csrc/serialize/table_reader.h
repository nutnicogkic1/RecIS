#pragma once
#include <memory>
#include <string>

#include "ATen/Utils.h"
#include "ATen/core/TensorBody.h"
#include "c10/util/flat_hash_map.h"
#include "c10/util/intrusive_ptr.h"
#include "embedding/hashtable.h"
#include "platform/filesystem.h"
#include "serialize/block_info.h"
namespace recis {
namespace serialize {
class TableReader : public at::intrusive_ptr_target {
 public:
  static at::intrusive_ptr<TableReader> Make(
      const std::string &dir_name, const std::string &base_name,
      const std::unordered_map<std::string, std::string> &tensor_name_map_);
  TableReader() = default;
  void LoadMeta();
  at::intrusive_ptr<BlockInfo> BlockInfoOfBlock(const std::string &block_info);
  std::unique_ptr<RandomAccessFile> File();
  void SetTensorNameMap(
      const std::unordered_map<std::string, std::string> &tensor_name_map_);

 private:
  AT_DISALLOW_COPY_AND_ASSIGN(TableReader);
  std::string path_;
  uint64_t offset_;
  ska::flat_hash_map<std::string, at::intrusive_ptr<BlockInfo>> block_infos_;
  const std::unordered_map<std::string, std::string> *tensor_name_map_ =
      nullptr;
};
}  // namespace serialize
}  // namespace recis
