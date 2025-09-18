#include "serialize/table_writer.h"

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>

#include "c10/util/Exception.h"
#include "c10/util/StringUtil.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/string_view.h"
#include "embedding/hashtable.h"
#include "nlohmann/json_fwd.hpp"
#include "platform/env.h"
#include "platform/fileoutput_buffer.h"
#include "platform/filesystem.h"
#include "platform/path.h"
#include "platform/status.h"
#include "serialize/block_info.h"
#include "serialize/name.h"

namespace recis {
namespace serialize {

TableWriter::TableWriter(const std::string &dir_name,
                         const std::string &base_name,
                         const std::string &json_name)
    : basename_(base_name),
      dirname_(dir_name),
      jsonname_(json_name),
      file_(nullptr),
      data_offset_(0) {}

at::intrusive_ptr<TableWriter> TableWriter::Make(const std::string &dir_name,
                                                 const std::string &base_name,
                                                 const std::string &json_name) {
  return at::make_intrusive<TableWriter>(dir_name, base_name, json_name);
}

void TableWriter::AppendWriteBlock(
    torch::intrusive_ptr<WriteBlock> write_block) {
  write_blocks_.push_back(write_block);
}

bool TableWriter::Empty() const { return write_blocks_.empty(); }

void TableWriter::SequentialWrite() {
  if (Empty()) {
    return;
  }
  OpenFile();
  SaveMeta();
  SaveData();
  file_->Close();
}

const std::string &TableWriter::FileName() const { return basename_; }

void TableWriter::SaveMeta() {
  nlohmann::ordered_json meta;
  for (auto block : write_blocks_) {
    data_offset_ = block->WriteMeta(meta, data_offset_);
  }
  // meta[BlockKeyMeta()] = nlohmann::json::object({});
  auto meta_str = meta.dump();

  json_file_->Append(meta_str);
  json_file_->Close();

  // delete is_dense key when save meta
  for (auto &kv : meta.items()) {
    kv.value().erase("dense");
  }

  meta_str = meta.dump();
  uint64_t append_size = (8 - (meta_str.size() % 8));
  meta_str.append(std::string(append_size, ' '));
  int64_t meta_size = meta_str.size();
  RECIS_THROW_IF_ERROR(
      file_->Append(torch::string_view((char *)&meta_size, sizeof(uint64_t))));
  file_->Append(meta_str);
}

void TableWriter::SaveData() {
  for (auto block : write_blocks_) {
    block->WriteData(file_.get());
  }
}

void TableWriter::OpenFile() {
  {
    std::unique_ptr<WritableFile> file;
    RECIS_THROW_IF_ERROR(Env::Default()->NewWritableFile(
        io::JoinPath(dirname_, basename_), &file));
    file_.reset(new FileOutputBuffer(file.release(), 8 << 20));
  }

  {
    std::unique_ptr<WritableFile> file;
    RECIS_THROW_IF_ERROR(Env::Default()->NewWritableFile(
        io::JoinPath(dirname_, jsonname_), &file));
    json_file_.reset(new FileOutputBuffer(file.release(), 8 << 20));
  }
}

}  // namespace serialize
}  // namespace recis
