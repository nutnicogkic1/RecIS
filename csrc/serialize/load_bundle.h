#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "ATen/PTThreadPool.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/ivalue.h"
#include "c10/util/flat_hash_map.h"
#include "c10/util/intrusive_ptr.h"
#include "embedding/hashtable.h"
#include "platform/filesystem.h"
#include "serialize/block_info.h"
#include "serialize/index_info.h"
#include "serialize/load_info.h"
#include "serialize/read_block.h"
#include "serialize/table_reader.h"

namespace recis {
namespace serialize {
class LoadBundle : public torch::CustomClassHolder {
 public:
  static at::intrusive_ptr<LoadBundle> Make(const std::string &path);
  void Build();
  bool HasTensor(const std::string &tensor_name);
  std::vector<std::string> ListTensor();
  std::vector<int64_t> TensorShape(const std::string &tensor_name);
  at::ScalarType TensorType(const std::string &tensor_name);
  const std::vector<std::string> SliceInfos(const std::string &tensor_name);
  at::intrusive_ptr<BlockInfo> GetBlockInfo(const std::string &block_name);
  at::intrusive_ptr<TableReader> BlockReadFile(const std::string &block_name);

 private:
  void BuildIndexInfo();
  void TryInitReader(int64_t file_index);
  void GetTensorNameMap();
  std::string path_;
  at::intrusive_ptr<IndexInfo> index_info_;
  std::vector<at::intrusive_ptr<TableReader>> readers_;
  std::unordered_map<std::string, std::string> tensor_name_map_;
};
}  // namespace serialize
}  // namespace recis
