#include <stddef.h>

#include <functional>
#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "column-io/framework/refcount.h"
#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"
#include "column-io/framework/thread_pool.h"
#ifndef COLUMN_IO_CC_COLUMN_IO_DATASET_DATASET_H_
#define COLUMN_IO_CC_COLUMN_IO_DATASET_DATASET_H_
namespace column {
namespace dataset {
class IteratorStateWriter {
public:
  virtual Status WriteString(const std::string &key,
                             const std::string &val) = 0;
  virtual Status WriteScalar(const std::string &key, int64_t val) = 0;
  virtual Status WriteScalar(const std::string &key,
                             const std::string &val) = 0;
  virtual Status WriteInt(const std::string &key, int64_t val) = 0;
  virtual Status WriteFloat(const std::string &key, double val) = 0;
  virtual Status WriteTensor(const std::string &key, const Tensor tensor) = 0;
  virtual ~IteratorStateWriter() = default;
};

class IteratorStateReader {
public:
  virtual bool Contain(const std::string &key) = 0;
  virtual bool Contains(const std::string &key) = 0;
  virtual Status ReadString(const std::string &key, std::string &val) = 0;
  virtual Status ReadInt(const std::string &key, int64_t &val) = 0;
  virtual Status ReadFloat(const std::string &key, double &val) = 0;
  virtual Status ReadTensor(const std::string &key, Tensor &tensor) = 0;
  virtual Status ReadScalar(const std::string &key, int64_t *val) = 0;
  virtual Status ReadScalar(const std::string &key, std::string *val) = 0;
  virtual ~IteratorStateReader() = default;
};

class IteratorBase {
public:
  virtual ~IteratorBase() {}
  IteratorBase() {}
  virtual Status Initialize() { return Status::OK(); }
  virtual Status GetNext(std::vector<Tensor> *outputs,
                         bool *end_of_sequence,
                         std::vector<size_t> *outputs_row_spliter = nullptr) = 0;
  virtual Status Save(IteratorStateWriter *writer) = 0;
  virtual Status Restore(IteratorStateReader *reader) = 0;

protected:
  Status SaveInput(IteratorStateWriter *writer,
                   std::shared_ptr<IteratorBase> &input) {
    return input->Save(writer);
  }
  Status RestoreInput(IteratorStateReader *reader,
                      std::shared_ptr<IteratorBase> &input) {
    return input->Restore(reader);
  }
  virtual Status SaveInternal(IteratorStateWriter *writer) = 0;
  virtual Status RestoreInternal(IteratorStateReader *reader) = 0;
  virtual Status GetNextInternal(std::vector<Tensor> *outputs,
                                 bool *end_of_sequence,
                                 std::vector<size_t> *outputs_row_spliter = nullptr) {
    /* @outputs: next遍历得到的原始tensor数据
     * @end_of_sequence: EOF标志位
     * @outputs_row_spliter: 行存输出模式下 原始tensors数据中列间分割点. (列存输出无需本参数: python schema生成ragged_ranks_用于packer.cc分割)
    */
    return Status::Unimplemented(__FUNCTION__, "Not Implemented");
  }
};

class DatasetBase : public std::enable_shared_from_this<DatasetBase> {
public:
  explicit DatasetBase(const std::string &name);
  const std::string &name() const { return name_; }
  Status MakeIterator(const std::string &prefix,
                      std::shared_ptr<IteratorBase> *iterator) {
    *iterator = MakeIteratorInternal(prefix);
    return (*iterator)->Initialize();
  }
  virtual ~DatasetBase() {}

protected:
  virtual std::shared_ptr<IteratorBase>
  MakeIteratorInternal(const std::string &prefix) = 0;

private:
  const std::string name_;
};

class DatasetIteratorBase : public IteratorBase {
public:
  struct BaseParam {
    const std::shared_ptr<DatasetBase> dataset;
    const std::string prefix;
  };
  virtual Status GetNext(std::vector<Tensor> *outputs,
                         bool *end_of_sequence,
                         std::vector<size_t> *outputs_row_spliter = nullptr) final {
    /* @outputs: next遍历得到的原始tensor数据
     * @end_of_sequence: EOF标志位
     * @ragged_ranks: 行存输出模式下 原始tensors数据中列间分割点. 列存输出模式无需本参数, 由python schema生成ragged_ranks_交予packer分割
    */
    return GetNextInternal(outputs, end_of_sequence, outputs_row_spliter);
  }
  virtual Status Save(IteratorStateWriter *writer) final {
    return SaveInternal(writer);
  }
  virtual Status Restore(IteratorStateReader *reader) final {
    return RestoreInternal(reader);
  }
  virtual const std::string prefix() const { return params_.prefix; }
  DatasetIteratorBase(const BaseParam &param) : params_(param) {}
  const std::string fullname(const std::string &name) {
    return absl::StrCat(prefix(), ":", name);
  }
  ~DatasetIteratorBase() {}

private:
  BaseParam params_;
};

template <typename DatasetType>
class DatasetIterator : public DatasetIteratorBase {
public:
  struct Param {
    const std::shared_ptr<DatasetType> dataset;
    const std::string prefix;
  };
  const std::shared_ptr<DatasetType> dataset() { return typed_dataset_; }
  DatasetIterator(const Param &param)
      : DatasetIteratorBase({param.dataset, param.prefix}) {
    typed_dataset_ = param.dataset;
  };

private:
  std::shared_ptr<DatasetType> typed_dataset_;
};

class DatasetBuilder {
  using CallFunc = std::function<absl::StatusOr<std::shared_ptr<DatasetBase>>(
      const std::string &)>;
  using CallVectorFunc = std::function<absl::StatusOr<std::shared_ptr<DatasetBase>>(
      const std::vector<std::string> &)>;

public:
  static std::shared_ptr<DatasetBuilder> Make(CallFunc func) {
    return std::shared_ptr<DatasetBuilder>(new DatasetBuilder(func));
  }
  static std::shared_ptr<DatasetBuilder> Make(CallVectorFunc vec_func) {
    return std::shared_ptr<DatasetBuilder>(new DatasetBuilder(vec_func));
  }
  DatasetBuilder() {}
  DatasetBuilder(CallFunc func) : func_(func) {}
  DatasetBuilder(CallVectorFunc vec_func) : vec_func_(vec_func) {}
  ~DatasetBuilder() {}
  absl::StatusOr<std::shared_ptr<DatasetBase>>
  MakeDataset(const std::string &path) {
    return func_(path);
  }
  absl::StatusOr<std::shared_ptr<DatasetBase>>
  MakeDataset(const std::vector<std::string> &path) {
    return vec_func_(path);
  }

private:
  CallFunc func_;
  CallVectorFunc vec_func_;
};

} // namespace dataset
} // namespace column
#endif
