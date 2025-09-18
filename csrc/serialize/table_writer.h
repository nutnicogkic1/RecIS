#pragma once
#include <memory>

#include "ATen/core/TensorBody.h"
#include "c10/util/intrusive_ptr.h"
#include "nlohmann/json_fwd.hpp"
#include "platform/fileoutput_buffer.h"
#include "serialize/block_info.h"
#include "serialize/write_block.h"
namespace recis {
namespace serialize {
class TableWriter : public at::intrusive_ptr_target {
 public:
  AT_DISALLOW_COPY_AND_ASSIGN(TableWriter);
  static at::intrusive_ptr<TableWriter> Make(const std::string &dir_name,
                                             const std::string &base_name,
                                             const std::string &json_name);
  void AppendWriteBlock(torch::intrusive_ptr<WriteBlock> write_block);
  void SequentialWrite();
  const std::string &FileName() const;
  bool Empty() const;
  TableWriter(const std::string &dir_name, const std::string &base_name,
              const std::string &json_name);

 private:
  void OpenFile();
  void SaveMeta();
  void SaveData();
  std::string basename_;
  std::string dirname_;
  std::unique_ptr<FileOutputBuffer> file_;
  std::string jsonname_;
  std::unique_ptr<FileOutputBuffer> json_file_;
  int64_t data_offset_;
  std::vector<torch::intrusive_ptr<WriteBlock>> write_blocks_;
};
using TableWriterPtr = std::unique_ptr<TableWriter>;
}  // namespace serialize
}  // namespace recis
