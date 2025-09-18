#pragma once
#include <string>

#include "c10/util/intrusive_ptr.h"
#include "serialize/index_info.h"
#include "serialize/table_writer.h"
#include "serialize/write_block.h"
namespace recis {
namespace serialize {
class SaveBundle : public at::intrusive_ptr_target {
 public:
  static at::intrusive_ptr<SaveBundle> Make(const std::string &path,
                                            int64_t parallel,
                                            int64_t shard_index,
                                            int64_t shard_num);
  SaveBundle() = default;
  AT_DISALLOW_COPY_AND_ASSIGN(SaveBundle);
  void AddBlock(at::intrusive_ptr<WriteBlock> write_block);
  void Save();
  void MergeMetaInfo();

 private:
  void GetSubFileData(const std::vector<std::string> &json_files,
                      nlohmann::json &json_data);
  void DumpJsonFile(const nlohmann::json &json_data, const std::string &path);
  void SaveMeta();
  void SaveTables();
  void MergeTorchRankJson();
  void MergeTensorKeyJson();
  void MergeJsonInfo();
  void InitTypeMap();
  int GenJsonInfo(const std::string &json_file, nlohmann::json &combine_data);
  int64_t IncTableIndex();
  std::string path_;
  int64_t shard_index_;
  int64_t shard_num_;
  int64_t parallel_;
  int64_t table_index_;
  at::intrusive_ptr<IndexInfo> index_info_;
  std::vector<at::intrusive_ptr<TableWriter>> table_writers_;
  std::vector<std::string> table_file_names_;
  std::unordered_map<std::string, std::string> tensor_name_map_;
  std::unordered_map<std::string, std::string> type_map_;
};
}  // namespace serialize
}  // namespace recis
