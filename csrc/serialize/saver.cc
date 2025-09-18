#include "serialize/saver.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>

#include "ATen/Parallel.h"
#include "ATen/core/interned_strings.h"
#include "c10/util/Exception.h"
#include "c10/util/StringUtil.h"
#include "c10/util/irange.h"
#include "c10/util/string_view.h"
#include "embedding/hashtable.h"
#include "nlohmann/json_fwd.hpp"
#include "platform/env.h"
#include "platform/fileoutput_buffer.h"
#include "platform/filesystem.h"
#include "platform/status.h"
#include "serialize/index_info.h"
#include "serialize/name.h"
#include "serialize/save_bundle.h"
#include "serialize/table_writer.h"
namespace recis {
namespace serialize {
Saver::Saver(int64_t shard_index, int64_t shard_num, int64_t parallel,
             const std::string &path)
    : parallel_(parallel),
      shard_index_(shard_index),
      shard_num_(shard_num),
      path_(path) {}

std::vector<at::intrusive_ptr<WriteBlock>> Saver::MakeWriteBlocks(
    const torch::Dict<std::string, HashTablePtr> &hashtables,
    const torch::Dict<std::string, torch::Tensor> &tensors) {
  std::vector<at::intrusive_ptr<WriteBlock>> write_blocks;
  for (const auto &kv : hashtables) {
    auto &&ht_blocks = WriteBlock::MakeHTWriteBlock(kv.value());
    write_blocks.insert(write_blocks.end(), ht_blocks.begin(), ht_blocks.end());
  }
  if (shard_index_ == 0) {
    for (const auto &kv : tensors) {
      write_blocks.push_back(
          WriteBlock::MakeTensorWriteBlock(kv.key(), kv.value()));
    }
  }
  return write_blocks;
}

void Saver::Save(std::vector<at::intrusive_ptr<WriteBlock>> write_blocks) {
  auto save_bundle =
      SaveBundle::Make(path_, parallel_, shard_index_, shard_num_);
  LOG(WARNING) << " Add Block";
  for (auto block : write_blocks) {
    save_bundle->AddBlock(block);
  }
  save_bundle->Save();
  save_bundle->MergeMetaInfo();
}

}  // namespace serialize
}  // namespace recis