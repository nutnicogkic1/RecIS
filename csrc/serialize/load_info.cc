#include "serialize/load_info.h"

#include <string>

#include "c10/util/flat_hash_map.h"
#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"
#include "serialize/name.h"
namespace recis {
namespace serialize {
void LoadInfo::Append(HashTablePtr ht) {
  auto slots = ht->SlotGroup();
  auto children_info = ht->ChildrenInfo();
  for (const auto &child_name : children_info->Children()) {
    auto &load_entry = load_infos_[child_name];
    load_entry[child_name].push_back(HTIdSlotName());
    for (const auto &slot : slots->Slots()) {
      load_entry[child_name].push_back(slot->Name());
    }
  }
}

void LoadInfo::Append(const std::string &tensor_name, at::Tensor tensor) {
  auto &load_entry = load_infos_[tensor_name];
  load_entry[tensor_name].resize(0);
}

std::string LoadInfo::Serialize() {
  nlohmann::json json;
  for (const auto &kv : load_infos_) {
    auto &dst_tensor_name = kv.first;
    nlohmann::json::object_t load_entry_json;
    for (const auto &load_entry : kv.second) {
      auto &src_tensor_name = load_entry.first;
      nlohmann::json::array_t slot_names_json;
      for (const auto &slot_name : load_entry.second) {
        slot_names_json.push_back(slot_name);
      }
      load_entry_json[src_tensor_name] = slot_names_json;
    }
    json[dst_tensor_name] = load_entry_json;
  }
  return json.dump();
}

const ska::flat_hash_map<
    std::string, ska::flat_hash_map<std::string, std::vector<std::string>>> &
LoadInfo::Infos() const {
  return load_infos_;
}

void LoadInfo::Deserialize(std::string load_info) {
  nlohmann::json json = nlohmann::json::parse(load_info);
  for (const auto &kv : json.items()) {
    auto &dst_tensor_name = kv.key();
    auto &load_entry_json = kv.value();
    for (const auto &load_entry : load_entry_json.items()) {
      auto &src_tensor_name = load_entry.key();
      auto &slot_names_json = load_entry.value();
      load_infos_[dst_tensor_name][src_tensor_name].resize(0);
      for (const auto &slot_name : slot_names_json) {
        load_infos_[dst_tensor_name][src_tensor_name].push_back(slot_name);
      }
    }
  }
}
}  // namespace serialize
}  // namespace recis
