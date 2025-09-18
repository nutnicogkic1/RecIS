#include "column-io/dataset/prefetch_dataset.h"
#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "column-io/framework/error_code.h"
#include "column-io/framework/status.h"
#include "column-io/framework/types.h"
#include <memory>
#include <mutex>
#include <string>
namespace column {
namespace dataset {
const std::string kDatasetname = "PrefetchDataset";

namespace {

class Dataset : public DatasetBase {
public:
  Dataset(const std::string &name, const std::shared_ptr<DatasetBase> &input,
          int64 buffer_size)
      : DatasetBase(name), input_(input), buffer_size_(buffer_size) {}

  std::shared_ptr<IteratorBase>
  MakeIteratorInternal(const std::string &prefix) override {
    return std::make_shared<Iterator>(
        std::dynamic_pointer_cast<Dataset>(shared_from_this()),
        absl::StrCat(prefix, "::", name()));
  }

private:
  class Iterator : public DatasetIterator<Dataset> {
  public:
    explicit Iterator(const std::shared_ptr<Dataset> &dataset,
                      const std::string &prefix)
        : DatasetIterator<Dataset>({dataset, prefix}) {}

    ~Iterator() override {
      // Signal the prefetch thread to terminate it. We will then
      // join that thread when we delete `this->prefetch_thread_`.
      //
      // TODO(mrry): Replace this cancellation logic with a
      // CancellationManager. The syntax would be more heavyweight,
      // but it would be possible to thread a cancellation manager
      // through the IteratorContext to upstream,
      // potentially-blocking iterators, when we add these.
      {
        absl::MutexLock l(&mu_);
        cancelled_ = true;
        cond_var_.SignalAll();
      }
    }

    Status Initialize() override {
      return dataset()->input_->MakeIterator(prefix(), &input_impl_);
    }

    Status GetNextInternal(std::vector<Tensor> *out_tensors,
                           bool *end_of_sequence,
                           std::vector<size_t> *outputs_row_spliter = nullptr) override {
      {
        absl::MutexLock l(&mu_);
        COLUMN_RETURN_NOT_OK(EnsurePrefetchThreadStarted());
        // Wait until the next element in the buffer has been
        // produced, or we are shutting down.
        while (!cancelled_ && buffer_.empty() && !prefetch_thread_finished_) {
          cond_var_.Wait(&mu_);
        }

        if (cancelled_) {
          return Status::Cancelled(
              "PrefetchDatasetOp::Dataset::Iterator::GetNext");
        }

        if (!buffer_.empty()) {
          return Consume(out_tensors, end_of_sequence);
        }

        if (prefetch_thread_finished_) {
          *end_of_sequence = true;
          return Status::OK();
        }
      }

      absl::MutexLock parent_l(&parent_mu_);
      absl::MutexLock l(&mu_);
      return input_impl_->GetNext(out_tensors, end_of_sequence);
    }

  protected:
    Status SaveInternal(IteratorStateWriter *writer) override {
      // Acquire both locks to ensure that the prefetch thread and
      // all GetNext threads are blocked.
      absl::MutexLock parent_l(&parent_mu_);
      absl::MutexLock l(&mu_);
      COLUMN_RETURN_NOT_OK(SaveInput(writer, input_impl_));
      COLUMN_RETURN_NOT_OK(
          writer->WriteScalar(fullname("buffer_size"), 0));
      //for (size_t i = 0; i < buffer_.size(); i++) {
      //  auto &buffer_element = buffer_[i];
      //  COLUMN_RETURN_NOT_OK(WriteStatus(writer, i, buffer_element.status));
      //  if (buffer_element.status.ok()) {
      //    COLUMN_RETURN_NOT_OK(writer->WriteScalar(
      //        fullname(absl::StrCat("buffer[", i, "].size")),
      //        buffer_element.value.size()));
      //    for (size_t j = 0; j < buffer_element.value.size(); j++) {
      //      COLUMN_RETURN_NOT_OK(writer->WriteTensor(
      //          fullname(absl::StrCat("buffer[", i, "][", j, "]")),
      //          buffer_element.value[j]));
      //    }
      //  }
      //}
      return Status::OK();
    }

    Status RestoreInternal(IteratorStateReader *reader) override {
      absl::MutexLock parent_l(&parent_mu_);
      absl::MutexLock l(&mu_);
      buffer_.clear();
      COLUMN_RETURN_NOT_OK(RestoreInput(reader, input_impl_));
      size_t buffer_size;
      {
        int64_t temp;
        COLUMN_RETURN_NOT_OK(
            reader->ReadScalar(fullname("buffer_size"), &temp));
        buffer_size = static_cast<size_t>(temp);
      }
      for (size_t i = 0; i < buffer_size; i++) {
        buffer_.emplace_back();
        auto &buffer_element = buffer_.back();
        COLUMN_RETURN_NOT_OK(ReadStatus(reader, i, &buffer_element.status));
        if (buffer_element.status.ok()) {
          size_t value_size;
          {
            int64_t temp;
            COLUMN_RETURN_NOT_OK(reader->ReadScalar(
                fullname(absl::StrCat("buffer[", i, "].size")), &temp));
            value_size = static_cast<size_t>(temp);
          }
          buffer_element.value.reserve(value_size);
          for (size_t j = 0; j < value_size; j++) {
            buffer_element.value.emplace_back();
            COLUMN_RETURN_NOT_OK(reader->ReadTensor(
                fullname(absl::StrCat("buffer[", i, "][", j, "]")),
                buffer_element.value.back()));
          }
        }
      }
      return Status::OK();
    }

  private:
    // A buffer element comprises a status and (if that status is
    // OK) a vector of tensors, representing an element of the input dataset.
    struct BufferElement {
      // The producer sets `status` if getting the input element fails.
      Status status;
      // The buffered data element.
      std::vector<Tensor> value;
    };

    Status Consume(std::vector<Tensor> *out_tensors, bool *end_of_sequence) {
      // A new element is available. Forward the status from computing it, and
      // (if we successfully got an element) the output values.
      Status s = buffer_.front().status;
      if (s.ok()) {
        *out_tensors = std::move(buffer_.front().value);
      }
      buffer_.pop_front();
      *end_of_sequence = false;

      // Wake the prefetch thread, in case it has been waiting for space
      // in the buffer. Also wake up threads from other calls to GetNext.
      //
      // TODO(mrry): Consider using different condition variables for
      // GetNext and Prefetch.
      cond_var_.SignalAll();
      return s;
    }

    Status EnsurePrefetchThreadStarted() {
      if (!prefetch_thread_) {
        prefetch_thread_.reset(
            new framework::StdThread([this]() { PrefetchThread(); }));
      }
      return Status::OK();
    }

    // Prefetches elements of the input, storing results in an internal
    // buffer.
    //
    // It owns the iterator context passed to it.
    void PrefetchThread() {
      while (true) {
        std::vector<Tensor> value;

        // 1. Wait for a slot in the buffer.
        {
          absl::MutexLock l(&mu_);
          while (!cancelled_ && buffer_.size() >= dataset()->buffer_size_) {
            cond_var_.Wait(&mu_);
          }

          if (cancelled_) {
            return;
          }
        }

        // 2. Read the next element.
        // Acquire the parent lock since we will be reading an element
        // from the input iterator. Note that we do not wish to release
        // this lock till we have added the fetched element to the
        // `buffer_` else there will be local state that may be missed
        // by SaveInternal.
        absl::MutexLock parent_l(&parent_mu_);
        bool end_of_sequence;
        BufferElement buffer_element;
        buffer_element.status =
            input_impl_->GetNext(&buffer_element.value, &end_of_sequence);
        if (buffer_element.status.ok() && end_of_sequence) {
          absl::MutexLock l(&mu_);
          prefetch_thread_finished_ = true;
          cond_var_.SignalAll();
          return;
        }

        // 3. Signal that the element has been produced.
        {
          absl::MutexLock l(&mu_);
          buffer_.push_back(std::move(buffer_element));
          cond_var_.SignalAll();
        }
      }
    }

    Status WriteStatus(IteratorStateWriter *writer, size_t index,
                       const Status &status) {
      COLUMN_RETURN_NOT_OK(writer->WriteScalar(
          CodeKey(index), static_cast<int64>(status.code())));
      if (!status.ok()) {
        COLUMN_RETURN_NOT_OK(writer->WriteScalar(ErrorMessageKey(index),
                                                 status.error_message()));
      }
      return Status::OK();
    }

    Status ReadStatus(IteratorStateReader *reader, size_t index,
                      Status *status) {
      int64_t code_int;
      COLUMN_RETURN_NOT_OK(reader->ReadScalar(CodeKey(index), &code_int));
      ErrorCode code = static_cast<ErrorCode>(code_int);

      if (code != ErrorCode::OK) {
        std::string error_message;
        COLUMN_RETURN_NOT_OK(
            reader->ReadScalar(ErrorMessageKey(index), &error_message));
        *status = Status(code, error_message);
      } else {
        *status = Status::OK();
      }
      return Status::OK();
    }

    std::string CodeKey(size_t index) {
      return fullname(absl::StrCat("status[", index, "].code"));
    }

    std::string ErrorMessageKey(size_t index) {
      return fullname(absl::StrCat("status[", index, "].error_message"));
    }

    // This mutex is used to ensure exclusivity between multiple threads
    // reading/writing this iterator's local state.
    absl::Mutex mu_;
    // This mutex is used to ensure exclusivity between multiple threads
    // accessing the parent iterator. We keep this separate from `mu_` to
    // allow prefetching to run in parallel with GetNext calls.
    absl::Mutex parent_mu_;
    std::shared_ptr<IteratorBase> input_impl_;
    absl::CondVar cond_var_;
    std::deque<BufferElement> buffer_;
    std::unique_ptr<framework::StdThread> prefetch_thread_;
    bool cancelled_ = false;
    bool prefetch_thread_finished_ = false;
  };
  const std::shared_ptr<DatasetBase> input_;
  const int64 buffer_size_;
};
} // namespace

std::shared_ptr<DatasetBase>
PrefetchDataset::MakeDataset(const std::shared_ptr<DatasetBase> &input,
                             int64 prefetch_num) {
  return std::make_shared<Dataset>(kDatasetname, input, prefetch_num);
}
} // namespace dataset
} // namespace column
