#include "column-io/dataset/repeat_dataset.h"
#include "absl/strings/str_cat.h"
#include "column-io/framework/error_code.h"
#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
namespace column {
namespace dataset {
namespace {
const std::string kDatasetName = "RepeatDataset";
class Dataset : public DatasetBase {
public:
  Dataset(const std::string &dataset_name,
          const std::shared_ptr<DatasetBase> &input, int64 take_size,
          int64 repeat)
      : DatasetBase(dataset_name), input_(input), take_size_(take_size),
        repeat_(repeat) {}

protected:
  std::shared_ptr<IteratorBase>
  MakeIteratorInternal(const std::string &prefix) {
    return std::make_shared<Iterator>(
        std::dynamic_pointer_cast<Dataset>(shared_from_this()), prefix);
  }

private:
  class Iterator : public DatasetIterator<Dataset> {
  public:
    Iterator(const std::shared_ptr<Dataset> &dataset, const std::string &prefix)
        : DatasetIterator<Dataset>({dataset, prefix}), left_repeat_(0),
          batch_index_(0) {}

  protected:
    Status SaveInternal(IteratorStateWriter *writer) override {
      std::lock_guard<std::mutex> l(mu_);
      if (input_impl_) {
        COLUMN_RETURN_NOT_OK(writer->WriteString(fullname("input"), ""));
        COLUMN_RETURN_NOT_OK(SaveInput(writer, input_impl_));
      }
      COLUMN_RETURN_NOT_OK(
          writer->WriteInt(fullname("left_repeat"), left_repeat_));
      COLUMN_RETURN_NOT_OK(
          writer->WriteInt(fullname("batch_index"), batch_index_));
      COLUMN_RETURN_NOT_OK(
          writer->WriteInt(fullname("cache_size"), cached_batches_.size()));
      for (size_t i = 0; i < cached_batches_.size(); i++) {
        COLUMN_RETURN_NOT_OK(
            writer->WriteInt(fullname(absl::StrCat("batch-", i, "-size")),
                             cached_batches_[i].size()));
        for (size_t j = 0; j < cached_batches_[i].size(); j++) {
          COLUMN_RETURN_NOT_OK(
              writer->WriteTensor(fullname(absl::StrCat("batch-", i, "-", j)),
                                  cached_batches_[i][j]));
        }
      }
      return Status::OK();
    }
    Status RestoreInternal(IteratorStateReader *reader) override {
      std::lock_guard<std::mutex> l(mu_);
      if (reader->Contain(fullname("input"))) {
        RestoreInput(reader, input_impl_);
      }
      COLUMN_RETURN_NOT_OK(
          reader->ReadInt(fullname("left_repeat"), left_repeat_));
      COLUMN_RETURN_NOT_OK(
          reader->ReadInt(fullname("batch_index"), batch_index_));
      int64_t cache_size;
      COLUMN_RETURN_NOT_OK(reader->ReadInt(fullname("cache_size"), cache_size));
      cached_batches_.reserve(cache_size);
      for (int64_t i = 0; i < cache_size; i++) {
        int64_t num_tensor;
        COLUMN_RETURN_NOT_OK(reader->ReadInt(
            fullname(absl::StrCat("batch-", i, "-size")), num_tensor));
        cached_batches_[i].resize(num_tensor);
        for (int64 j = 0; j < num_tensor; j++) {
          COLUMN_RETURN_NOT_OK(
              reader->ReadTensor(fullname(absl::StrCat("batch-", i, "-", j)),
                                 cached_batches_[i][j]));
        }
      }
      return Status::OK();
    }

    Status Initialize() override {
      if (dataset()->take_size_ <= 0) {
        return Status::InvalidArgument("take size must > 0, but get [",
                                       dataset()->take_size_, "]");
      }
      if (dataset()->repeat_ == 0 || dataset()->repeat_ < -1) {
        return Status::InvalidArgument("repeat must == -1 or > 0, but get [",
                                       dataset()->repeat_, "]");
      }
      return dataset()->input_->MakeIterator(
          fullname(absl::StrCat(prefix(), "::input")), &input_impl_);
    }

    Status GetNextInternal(std::vector<Tensor> *outputs,
                           bool *end_of_sequence,
                           std::vector<size_t> *outputs_row_spliter = nullptr) override {
      std::lock_guard<std::mutex> l(mu_);
      // update for finite mode
      // and first load for infinite mode.
      if (left_repeat_ == 0) {
        if (!input_impl_) {
          *end_of_sequence = true;
          return Status::OK();
        }
        left_repeat_ = dataset()->repeat_;
        int64 num_batch = dataset()->take_size_;
        batch_index_ = 0;
        Status st;
        cached_batches_.clear();
        while (num_batch--) {
          std::vector<Tensor> batch;
          st = input_impl_->GetNext(&batch, end_of_sequence);
          if (!st.ok() || *end_of_sequence) {
            break;
          }
          cached_batches_.emplace_back(batch);
        }
        // input no data or no enough data;
        if (*end_of_sequence) {
          input_impl_.reset();
          // no data
          if (cached_batches_.empty()) {
            *end_of_sequence = true;
            return Status::OK();
          }
          *end_of_sequence = false;
          LOG(INFO) << "Only take [" << cached_batches_.size()
                    << "] batchs from input "
                    << " expect to get [" << dataset()->take_size_ << "]";
        } else if (!st.ok() && st.code() != ErrorCode::OUT_OF_RANGE) {
          return st;
        }
      }
      *outputs = cached_batches_[batch_index_];
      batch_index_++;
      batch_index_ %= cached_batches_.size();
      if (left_repeat_ > 0) {
        left_repeat_--;
      }
      return Status::OK();
    }

  private:
    std::mutex mu_;
    std::shared_ptr<IteratorBase> input_impl_;
    std::vector<std::vector<Tensor>> cached_batches_;
    int64_t left_repeat_;
    int64_t batch_index_;
  };
  const std::shared_ptr<DatasetBase> input_;
  int64 take_size_;
  int64 repeat_;
};
} // namespace

std::shared_ptr<DatasetBase>
RepeatDataset::MakeDataset(const std::shared_ptr<DatasetBase> &input,
                           int64 take_size, int64 repeat) {
  return std::make_shared<Dataset>(kDatasetName, input, take_size, repeat);
}
} // namespace dataset
} // namespace column
