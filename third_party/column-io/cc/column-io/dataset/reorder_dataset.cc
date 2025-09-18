#include "absl/strings/str_cat.h"
#include "column-io/dataset/packer.h"
#include "column-io/dataset/parallel_dataset.h"
#include "column-io/framework/error_code.h"
#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"
#include <cstddef>
#include <memory>
#include <mutex>
namespace column {
namespace dataset {
namespace {
const std::string kDatasetName = "ReorderDataset";
class Dataset : public DatasetBase {
public:
  Dataset(const std::string &name, const std::shared_ptr<DatasetBase> &input,
          const std::vector<int64> &new_order)
      : DatasetBase(name), input_(input), new_order_(new_order) {}

  std::shared_ptr<IteratorBase>
  MakeIteratorInternal(const std::string &prefix) override {
    return std::make_shared<Iterator>(
        std::dynamic_pointer_cast<Dataset>(shared_from_this()),
        absl::StrCat(prefix, "::", name()));
  }

private:
  class Iterator : public DatasetIterator<Dataset> {
  public:
    Iterator(const std::shared_ptr<Dataset> dataset, const std::string &prefix)
        : DatasetIterator<Dataset>({dataset, prefix}) {}

    Status Initialize() override {
      return dataset()->input_->MakeIterator(fullname("::input"), &input_impl_);
    }

    Status GetNextInternal(std::vector<Tensor> *outputs,
                           bool *end_of_sequence,
                           std::vector<size_t> *outputs_row_spliter = nullptr) override {
      std::lock_guard<std::mutex> l(mu_);
      if (!input_impl_) {
        *end_of_sequence = true;
        return Status::OK();
      }
      std::vector<Tensor> tmp_output;
      auto st = input_impl_->GetNext(&tmp_output, end_of_sequence);
      if (*end_of_sequence) {
        return Status::OK();
      }
      if (!st.ok()) {
        return st;
      }
      outputs->clear();
      outputs->resize(tmp_output.size());
      for (size_t index = 0; index < dataset()->new_order_.size(); index++) {
        (*outputs)[index] = tmp_output[(dataset()->new_order_)[index]];
      }
      return Status::OK();
    }

  protected:
    Status SaveInternal(IteratorStateWriter *writer) override {
      std::lock_guard<std::mutex> l(mu_);
      if (input_impl_) {
        RETURN_IF_ERROR(writer->WriteString(fullname("input"), ""));
        RETURN_IF_ERROR(SaveInput(writer, input_impl_));
      }
      return Status::OK();
    }
    Status RestoreInternal(IteratorStateReader *reader) override {
      std::lock_guard<std::mutex> l(mu_);
      if (reader->Contain(fullname("input"))) {
        RETURN_IF_ERROR(RestoreInput(reader, input_impl_));
      }
      return Status::OK();
    }

  private:
    std::mutex mu_;
    std::shared_ptr<IteratorBase> input_impl_;
  };
  const std::shared_ptr<DatasetBase> input_;
  std::vector<int64> new_order_;
};
} // namespace

std::shared_ptr<dataset::DatasetBase>
Packer::MakeReorderDataset(const std::shared_ptr<DatasetBase> input,
                           const std::vector<int64> &new_order) {
  return std::make_shared<Dataset>(kDatasetName, input, new_order);
}
} // namespace dataset
} // namespace column
