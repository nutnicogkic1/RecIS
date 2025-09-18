#pragma once
#include <memory>
#include <string>

#include "ATen/core/TensorBody.h"
#include "ATen/core/ivalue.h"
#include "c10/util/intrusive_ptr.h"
#include "embedding/hashtable.h"
#include "embedding/slot_group.h"
#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"
#include "platform/fileoutput_buffer.h"
#include "serialize/block_info.h"
namespace recis {
namespace serialize {
class WriteBlock : public torch::CustomClassHolder {
 public:
  static std::vector<torch::intrusive_ptr<WriteBlock>> MakeHTWriteBlock(
      HashTablePtr hashtable);
  static torch::intrusive_ptr<WriteBlock> MakeTensorWriteBlock(
      const std::string &block_name, const at::Tensor &tensor);

  WriteBlock() = default;

  const std::string &TensorName();
  const std::string &SliceInfo();
  const bool IsDense();

  virtual void WriteData(FileOutputBuffer *file) = 0;
  virtual int64_t WriteMeta(nlohmann::ordered_json &meta, int64_t offset) = 0;
  virtual ~WriteBlock() = default;

  void SetTensorName(const std::string &new_tensor_name);

 protected:
  std::string tensor_name_;
  std::string slice_info_;
  bool is_dense_;
};

class TensorWriteBlock : public WriteBlock {
 public:
  TensorWriteBlock() = default;
  static torch::intrusive_ptr<WriteBlock> Make(const std::string &block_name,
                                               const at::Tensor &tensor);
  void WriteData(FileOutputBuffer *file) override;
  int64_t WriteMeta(nlohmann::ordered_json &meta, int64_t offset) override;
  ~TensorWriteBlock() = default;

 private:
  torch::Tensor tensor_;
};

class HTIdWriteBlock : public WriteBlock {
 public:
  HTIdWriteBlock() = default;
  static torch::intrusive_ptr<WriteBlock> Make(
      const std::string &shared_name, const std::string &slice_info,
      const std::shared_ptr<std::vector<int64_t>> ids);
  void WriteData(FileOutputBuffer *file) override;
  int64_t WriteMeta(nlohmann::ordered_json &meta, int64_t offset) override;
  ~HTIdWriteBlock() = default;

 private:
  std::shared_ptr<std::vector<int64_t>> ids_;
};

class HTSlotWriteBlock : public WriteBlock {
 public:
  HTSlotWriteBlock() = default;
  static torch::intrusive_ptr<WriteBlock> Make(
      const std::string &shared_name, const std::string &slot_name,
      const std::string &slice_info,
      const std::shared_ptr<std::vector<int64_t>> index,
      at::intrusive_ptr<embedding::Slot> slot);
  void WriteData(FileOutputBuffer *file) override;
  int64_t WriteMeta(nlohmann::ordered_json &meta, int64_t offset) override;
  ~HTSlotWriteBlock() = default;

 private:
  at::intrusive_ptr<embedding::Slot> slot_;
  std::shared_ptr<std::vector<int64_t>> index_;
  int64_t write_size_;
};
}  // namespace serialize
}  // namespace recis