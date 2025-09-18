#include "serialize/index_info.h"

#include <string>
#include <unordered_set>
#include <vector>

#include "c10/core/ScalarType.h"
#include "c10/util/Logging.h"
#include "c10/util/intrusive_ptr.h"
#include "serialize/name.h"
#include "utils/str_util.h"
namespace recis {
namespace serialize {

at::intrusive_ptr<IndexInfo> IndexInfo::Make() {
  return at::make_intrusive<IndexInfo>();
}

void IndexInfo::Append(const std::string &tensor_name,
                       const std::string &slice_info,
                       const std::string &sep_info,
                       const std::string &file_name) {
  if (file_index_map_.count(file_name) == 0) {
    file_index_map_[file_name] = file_index_map_.size();
  }
  block_index_map_[BlockNameEncode(tensor_name, slice_info, sep_info)] =
      file_index_map_[file_name];
}

void IndexInfo::MergeFrom(IndexInfo &rhv) {
  int64_t data_file_num = this->file_index_map_.size();
  for (auto &kv : rhv.file_index_map_) {
    file_index_map_[kv.first] = data_file_num + kv.second;
  }
  for (auto &kv : rhv.block_index_map_) {
    block_index_map_[kv.first] = kv.second + data_file_num;
  }
}

std::vector<std::string> IndexInfo::SliceInfoOfTensor(
    const std::string &tensor_name) {
  std::vector<std::string> slice_infos;
  auto empty_block_name = BlockNameEncode(tensor_name, "", "");
  for (const auto &kv : block_index_map_) {
    if (util::string::StartsWith(kv.first, empty_block_name)) {
      slice_infos.push_back(SliceInfoFromBlockName(kv.first));
    }
  }
  return slice_infos;
}

bool IndexInfo::HasBlock(const std::string &tensor_name,
                         const std::string &slice_info) {
  return block_index_map_.count(BlockNameEncode(tensor_name, slice_info)) > 0;
}

bool IndexInfo::HashTensor(const std::string &tensor_name) {
  auto empty_block_name = BlockNameEncode(tensor_name, "", "");
  for (const auto &kv : block_index_map_) {
    if (util::string::StartsWith(kv.first, empty_block_name)) {
      return true;
    }
  }
  return false;
}

std::vector<std::string> IndexInfo::ListTensor() {
  std::unordered_set<std::string> name_set;
  for (const auto &kv : block_index_map_) {
    name_set.insert(TensorNameFromBlockName(kv.first));
  }
  std::vector<std::string> ret(name_set.begin(), name_set.end());
  return ret;
}

int64_t IndexInfo::FileIndexOfBlock(const std::string &block_name) {
  return block_index_map_.at(block_name);
}

const std::string IndexInfo::FileNameByIndex(int64_t index) {
  if (index_file_map_.empty()) {
    for (const auto &kv : file_index_map_) {
      index_file_map_[kv.second] = kv.first;
    }
  }
  return index_file_map_.at(index);
}

std::string IndexInfo::Serialize() {
  nlohmann::json::object_t file_index_json;
  for (const auto &kv : file_index_map_) {
    file_index_json[kv.first] = kv.second;
  }
  nlohmann::json::object_t block_index_json;
  for (const auto &kv : block_index_map_) {
    block_index_json[kv.first] = kv.second;
  }
  nlohmann::json index;
  index[KFileKey] = file_index_json;
  index[kBlockKey] = block_index_json;
  return index.dump();
}

void IndexInfo::SetTensorNameMap(
    const std::unordered_map<std::string, std::string> &tensor_name_map) {
  tensor_name_map_ = &tensor_name_map;
}

void IndexInfo::Deserialize(const std::string &info_buf) {
  auto info_json = nlohmann::json::parse(info_buf);
  nlohmann::json::object_t file_index_json = info_json[KFileKey];
  nlohmann::json::object_t block_index_json = info_json[kBlockKey];
  for (const auto &kv : file_index_json) {
    file_index_map_[kv.first] = kv.second;
  }
  for (const auto &kv : block_index_json) {
    auto tensor_name = kv.first;
    if ((nullptr != tensor_name_map_) &&
        (tensor_name_map_->count(tensor_name))) {
      tensor_name = (*tensor_name_map_).at(tensor_name);
    }
    block_index_map_[tensor_name] = kv.second;
  }
}
std::string IndexInfo::kBlockKey = "block_index";
std::string IndexInfo::KFileKey = "file_index";

}  // namespace serialize
}  // namespace recis
