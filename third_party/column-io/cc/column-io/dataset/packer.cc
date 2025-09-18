#include "column-io/dataset/packer.h"
#include "absl/log/log.h"
#include "absl/synchronization/blocking_counter.h"
#include "column-io/dataset/dataset.h"
#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"
#include "column-io/framework/cuda_utils.h"
#include <cstddef>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <type_traits>

namespace column {
namespace dataset {
namespace {
const std::string &kDatasetName("PackerDataset");
template <typename T> struct is_simple_type {
  static constexpr bool value = std::is_trivial<T>::value;
};
// The macro CASES() expands to a switch statement conditioned on
// TYPE_ENUM. Each case expands the STMTS after a typedef for T.
#define SINGLE_ARG(...) __VA_ARGS__
#define CASE(TYPE, STMTS)                                                      \
  case DataTypeToEnum<TYPE>::value: {                                          \
    typedef TYPE T;                                                            \
    STMTS;                                                                     \
    break;                                                                     \
  }
#define CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, INVALID, DEFAULT)                 \
  switch (TYPE_ENUM) {                                                         \
    CASE(float, SINGLE_ARG(STMTS))                                             \
    CASE(double, SINGLE_ARG(STMTS))                                            \
    CASE(int32_t, SINGLE_ARG(STMTS))                                           \
    CASE(uint8_t, SINGLE_ARG(STMTS))                                           \
    CASE(uint16_t, SINGLE_ARG(STMTS))                                          \
    CASE(uint32_t, SINGLE_ARG(STMTS))                                          \
    CASE(uint64_t, SINGLE_ARG(STMTS))                                          \
    CASE(int16_t, SINGLE_ARG(STMTS))                                           \
    CASE(int8_t, SINGLE_ARG(STMTS))                                            \
    CASE(std::string, SINGLE_ARG(STMTS))                                       \
    CASE(int64_t, SINGLE_ARG(STMTS))                                           \
    CASE(bool, SINGLE_ARG(STMTS))                                              \
  default:                                                                     \
    DEFAULT;                                                                   \
    break;                                                                     \
  }

#define CASES(TYPE_ENUM, STMTS)                                                \
  CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, LOG(FATAL) << "Type not set";           \
                     , LOG(FATAL) << "Unexpected type: " << TYPE_ENUM;)

const int kPackTensorBlock = 16;
class Dataset : public DatasetBase {
public:
  Dataset(const std::string &name, size_t batch_size, bool drop_remainder,
          const std::vector<int> &pack_tables, int num_tables,
          const std::vector<int> &ragged_ranks,
          const std::shared_ptr<DatasetBase> input, int64 parallel,
	  bool pinned_result, bool gpu_result)
      : DatasetBase(name), batch_size_(batch_size),
        drop_remainder_(drop_remainder), pack_tables_(pack_tables),
        ragged_ranks_(ragged_ranks), input_(input), num_tables_(num_tables),
        splits_sub_idx_(ragged_ranks.size()),
        pool_(new framework::StdThreadPool("PackerDatasetPool", parallel)),
        pinned_result_(pinned_result), gpu_result_(gpu_result) {
    for (int i = 1; i < ragged_ranks.size(); ++i) {
      splits_sub_idx_[i] = splits_sub_idx_[i - 1] + ragged_ranks[i - 1];
    }
    if (gpu_result) {
      device_id_ = GetCudaDeviceId();
      GPU_CK(cudaSetDevice(device_id_));
    }
    GPU_CK(cudaStreamCreate(&stream_));
  }

  int32_t get_batch_size(std::unique_ptr<std::vector<Tensor>> &batch) {
    return get_batch_size(*(batch.get()));
  }
  int32_t get_batch_size(const std::vector<Tensor> &batch) {
    // take indicator size as batch size.
    if (num_tables_ > 1) {
      return batch[0].NumElements();
    }
    // take first indice tensor size as batch size.
    if (ragged_ranks_[0] == 0) {
      return batch[0].Shape()[0];
    } else {
      int split_index = ragged_ranks_.size() + ragged_ranks_[0] - 1;
      return batch[split_index].Shape()[0] - 1;
    }
  }
  ~Dataset() override {
    GPU_CK(cudaStreamSynchronize(stream_));
    GPU_CK(cudaStreamDestroy(stream_));
  }

  std::shared_ptr<IteratorBase>
  MakeIteratorInternal(const std::string &prefix) override {
    return std::unique_ptr<IteratorBase>(
        new Iterator(std::dynamic_pointer_cast<Dataset>(shared_from_this()),
                     absl::StrCat(prefix, "::PackV2"), batch_size_));
  }

private:
  class Iterator : public DatasetIterator<Dataset> {
  public:
    explicit Iterator(const std::shared_ptr<Dataset> dataset,
                      const std::string &prefix, size_t batch_size)
        : DatasetIterator<Dataset>({dataset, prefix}), batch_size_(batch_size) {
          allocator_ = GetAllocator(this->dataset()->pinned_result_ || this->dataset()->gpu_result_);
          cuda_allocator_ = GetCudaAllocator(this->dataset()->stream_, this->dataset()->device_id_);
    }

    Status Initialize() override {
      return this->dataset()->input_->MakeIterator(this->prefix(),
                                                   &input_impl_);
    }

    size_t get_batch_size() { return batch_size_; }

    Status GetNextInternal(std::vector<Tensor> *out_tensors,
                           bool *end_of_sequence,
                           std::vector<size_t> *outputs_row_spliter = nullptr) override {
      int num_tensors = this->dataset()->ragged_ranks_.size();
      int num_splits = this->dataset()->splits_sub_idx_.back() +
                       this->dataset()->ragged_ranks_.back();
      int indicators_idx = 0;
      int values_idx = indicators_idx + (this->dataset()->num_tables_ - 1);
      int splits_idx = values_idx + num_tensors;

      std::vector<std::vector<Tensor>> batch_elements;
      int total_size = 0, head_offset = 0;

      std::lock_guard<std::mutex> l(
          mu_); // Consider reduce the granularity mutex ?
      *end_of_sequence = false;

      auto cur_batch_size = get_batch_size();

      if (remain_) {
        // maybe modify cur_batch_size according to data in "remain_"
        head_offset = remain_offset_;
        total_size += dataset()->get_batch_size(remain_) - remain_offset_;
        batch_elements.emplace_back(std::move(*remain_));
        remain_.reset();
      }

      while (total_size < cur_batch_size && !*end_of_sequence && input_impl_) {
        std::vector<Tensor> batch_element;
        RETURN_IF_ERROR(input_impl_->GetNext(&batch_element, end_of_sequence));
        if (!*end_of_sequence) {
          total_size += dataset()->get_batch_size(batch_element);
          batch_elements.emplace_back(std::move(batch_element));
        } else {
          input_impl_.reset();
        }
      }

      if (total_size == 0 ||
          (this->dataset()->drop_remainder_ && total_size < cur_batch_size)) {
        *end_of_sequence = true;
        return Status::OK();
      }

      Tensor batch_size_t(kInt32, {});
      std::vector<Tensor> indicators_t(this->dataset()->num_tables_ - 1);
      std::vector<Tensor> values_t(num_tensors);
      std::vector<Tensor> splits_t(num_splits);

      std::vector<std::vector<std::pair<int, int>>> all_table_ranges;
      all_table_ranges.reserve(batch_elements.size());

      int batch_size = std::min<int64_t>(total_size, cur_batch_size);
      for (int i = 0; i < batch_elements.size(); ++i) {
        std::vector<Tensor> &batch_element = batch_elements[i];

        // std::vector<std::vector<int>> table_ranges;
        std::vector<std::pair<int, int>> table_ranges;
        int begin = 0, end = dataset()->get_batch_size(batch_element);
        if (i == 0) {
          begin = head_offset;
        }
        if (i == batch_elements.size() - 1) {
          end -= total_size - batch_size;
        }
        table_ranges.emplace_back(begin, end);
        for (int j = 0; j < this->dataset()->num_tables_ - 1; ++j) {
          Tensor &indicators_t = batch_element[indicators_idx + j];
          auto indicators = indicators_t.Raw<int64_t>();
          int common_begin = std::numeric_limits<int>::max(),
              common_end = std::numeric_limits<int>::min();
          for (int k = begin; k < end; ++k) {
            int refer = static_cast<int>(indicators[k]);
            common_begin = std::min(common_begin, refer);
            common_end = std::max(common_end, refer + 1);
          }
          table_ranges.emplace_back(common_begin, common_end);
        }
        all_table_ranges.emplace_back(table_ranges);
      }

      auto pack_tensor = [&](int j) {
        int table = this->dataset()->pack_tables_[j];
        int sub_idx = this->dataset()->splits_sub_idx_[j];

        // Do statistic.

        std::vector<int> ragged_size(this->dataset()->ragged_ranks_[j] + 1, 0);
        for (int i = 0; i < batch_elements.size(); ++i) {
          std::vector<Tensor> &batch_element = batch_elements[i];

          int begin, end;
          std::tie(begin, end) = all_table_ranges[i][table];
          for (int k = this->dataset()->ragged_ranks_[j] - 1; k >= 0; --k) {
#define DECLARE_HANDLE_FOR_TYPE(Type)                                          \
  auto handle_##Type = [&]() {                                                 \
    ragged_size[k + 1] += end - begin;                                         \
    auto current_splits = batch_element[splits_idx + sub_idx + k].Raw<Type>(); \
    begin = current_splits[begin];                                             \
    end = current_splits[end];                                                 \
  }
            DECLARE_HANDLE_FOR_TYPE(int32_t);
            DECLARE_HANDLE_FOR_TYPE(int64_t);
#undef DECLARE_HANDLE_FOR_TYPE
            if (batch_element[splits_idx + sub_idx + k].Type() == kInt64) {
              handle_int64_t();
            } else {
              handle_int32_t();
            }
          }
          ragged_size[0] += end - begin;
        }

        // Allocate tensors.
        for (int k = 0; k < this->dataset()->ragged_ranks_[j]; ++k) {
#define DECLARE_HANDLE_FOR_TYPE(Type, TfType)                                  \
  auto handle_##Type = [&]() {                                                 \
    splits_t[sub_idx + k] = Tensor(allocator_, {size_t(ragged_size[k + 1] + 1)}, TfType);  \
    splits_t[sub_idx + k].Raw<Type>()[0] = 0;                                  \
  }
          DECLARE_HANDLE_FOR_TYPE(int32_t, kInt32);
          DECLARE_HANDLE_FOR_TYPE(int64_t, kInt64);
#undef DECLARE_HANDLE_FOR_TYPE
          if (batch_elements[0][splits_idx + sub_idx + k].Type() == kInt64) {
            handle_int64_t();
          } else {
            handle_int32_t();
          }
        }

        DataType dtype = batch_elements[0][values_idx + j].Type();
        TensorShape shape = batch_elements[0][values_idx + j].Shape();
        TensorShape old = shape;
        shape.Set(0, ragged_size[0]);
        values_t[j] = Tensor(allocator_, shape, dtype);
        int dense_dim = 1;
        for (int i = 1; i < shape.Size(); ++i) {
          dense_dim *= shape.Dims()[i];
        }

        // Copy results.

        int next_offset = 0;
        for (int i = 0; i < batch_elements.size(); ++i) {
          std::vector<Tensor> &batch_element = batch_elements[i];

          int begin, end;
          std::tie(begin, end) = all_table_ranges[i][table];
          int offset = next_offset;
          next_offset += end - begin;

          for (int k = this->dataset()->ragged_ranks_[j] - 1; k >= 0; --k) {
#define DECLARE_HANDLE_FOR_TYPE(Type)                                          \
  auto handle_##Type = [&]() {                                                 \
    auto splits = splits_t[sub_idx + k].Raw<Type>();                           \
    auto current_splits = batch_element[splits_idx + sub_idx + k].Raw<Type>(); \
    int splits_offset = splits[offset];                                        \
    int splits_begin = current_splits[begin];                                  \
    int splits_end = current_splits[end];                                      \
    for (int l = 1; l <= end - begin; ++l) {                                   \
      splits[offset + l] =                                                     \
          current_splits[begin + l] + splits_offset - splits_begin;            \
    }                                                                          \
    begin = splits_begin;                                                      \
    end = splits_end;                                                          \
    offset = splits_offset;                                                    \
  }
            DECLARE_HANDLE_FOR_TYPE(int32_t);
            DECLARE_HANDLE_FOR_TYPE(int64_t);
#undef DECLARE_HANDLE_FOR_TYPE
            if (batch_element[splits_idx + sub_idx + k].Type() == kInt64) {
              handle_int64_t();
            } else {
              handle_int32_t();
            }
          }

          CASES(
              dtype, do {
                auto values = values_t[j].Raw<T>();
                auto current_values = batch_element[values_idx + j].Raw<T>();
                if (is_simple_type<T>::value) {
                  std::memcpy(&values[offset * dense_dim],
                              &current_values[begin * dense_dim],
                              sizeof(T) * (end - begin) * dense_dim);
                } else {
                  for (int l = 0; l < (end - begin) * dense_dim; ++l) {
                    values[offset + l] = current_values[begin + l];
                  }
                }
              } while (0));
        }
        return Status::OK();
      };

      absl::BlockingCounter counter((num_tensors - 1) / kPackTensorBlock + 1);
      Status status;
      std::mutex status_mu;
      for (int job_begin = 0; job_begin < num_tensors;
           job_begin += kPackTensorBlock) {
        int job_end = std::min(job_begin + kPackTensorBlock, num_tensors);
        dataset()->pool_->Schedule([job_begin, job_end, &status, &status_mu,
                                    &counter, &pack_tensor]() {
          for (int j = job_begin; j < job_end; ++j) {
            Status s = pack_tensor(j);
            {
              std::lock_guard<std::mutex> l(status_mu);
              status = s;
            }
          }
          counter.DecrementCount();
        });
      }

      batch_size_t.Raw<int32_t>()[0] = 0;
      for (int i = 0; i < batch_elements.size(); ++i) {
        int begin, end;
        std::tie(begin, end) = all_table_ranges[i][0];
        batch_size_t.Raw<int32_t>()[0] += end - begin;
      }

      for (int j = 0; j < this->dataset()->num_tables_ - 1; ++j) {
        indicators_t[j] =
            Tensor(allocator_, {(size_t)batch_size_t.Scalar<int32_t>()}, kInt64);

        int offset = 0, indicators_offset = 0;
        for (int i = 0; i < batch_elements.size(); ++i) {
          std::vector<Tensor> &batch_element = batch_elements[i];

          int begin, end, indicators_begin, indicators_end;
          std::tie(begin, end) = all_table_ranges[i][0];
          std::tie(indicators_begin, indicators_end) =
              all_table_ranges[i][j + 1];

          auto indicators = indicators_t[j].Raw<int64_t>();
          auto current_indicators =
              batch_element[indicators_idx + j].Raw<int64_t>();

          for (int l = 0; l < end - begin; ++l) {
            indicators[offset + l] = current_indicators[begin + l];
            if (indicators[offset + l] >= 0) { // for no refer
              indicators[offset + l] += indicators_offset - indicators_begin;
            }
          }

          offset += end - begin;
          indicators_offset += indicators_end - indicators_begin;
        }
      }

      counter.Wait();
      RETURN_IF_ERROR(status);

      remain_offset_ = all_table_ranges.back()[0].second;
      if (remain_offset_ < dataset()->get_batch_size(batch_elements.back())) {
        remain_.reset(
            new std::vector<Tensor>(std::move(batch_elements.back())));
      }

      if (this->dataset()->gpu_result_) {
        out_tensors->reserve(
            indicators_t.size() + values_t.size() + splits_t.size());
        auto copy_and_push = [&](std::vector<Tensor>& tensors) {
          for (auto& tensor : tensors) {
            if (tensor.Type() == kString) {
              out_tensors->emplace_back(std::move(tensor));
            } else {
              auto tmp_tensor = Tensor(
                  cuda_allocator_,
                  tensor.Shape(), tensor.Type(),
                  {kDLCUDA, this->dataset()->device_id_});
              GPU_CK(cudaMemcpyAsync(
                    tmp_tensor.mutable_data(),
                    tensor.data(), tensor.TotalBytes(),
                    cudaMemcpyDefault, this->dataset()->stream_));
              out_tensors->emplace_back(std::move(tmp_tensor));
            }
          }
        };
        copy_and_push(indicators_t);
        copy_and_push(values_t);
        copy_and_push(splits_t);
      } else {
        std::move(indicators_t.begin(), indicators_t.end(),
                  std::back_inserter(*out_tensors));
        std::move(values_t.begin(), values_t.end(),
                  std::back_inserter(*out_tensors));
        std::move(splits_t.begin(), splits_t.end(),
                  std::back_inserter(*out_tensors));
      }
      GPU_CK(cudaStreamSynchronize(this->dataset()->stream_));
      *end_of_sequence = false;
      return Status::OK();
    }

  protected:
    Status SaveInternal(IteratorStateWriter *writer) override {
      std::lock_guard<std::mutex> l(mu_);
      if (!input_impl_) {
        RETURN_IF_ERROR(writer->WriteScalar(fullname("input_impl_empty"), ""));
      } else {
        RETURN_IF_ERROR(SaveInput(writer, input_impl_));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorStateReader *reader) override {
      std::lock_guard<std::mutex> l(mu_);
      if (!reader->Contains(fullname("input_impl_empty"))) {
        RETURN_IF_ERROR(RestoreInput(reader, input_impl_));
      } else {
        input_impl_.reset();
      }
      return Status::OK();
    }

  private:
    std::mutex mu_;
    std::shared_ptr<IteratorBase> input_impl_;
    std::unique_ptr<std::vector<Tensor>> remain_;
    int remain_offset_;
    size_t batch_size_;
    std::queue<std::vector<Tensor>> output_elements_;
    int max_group_id_ = 0;
    Allocator* allocator_;
    Allocator* cuda_allocator_;
  };

  size_t batch_size_;
  const std::vector<int> pack_tables_;
  const std::vector<int> ragged_ranks_;

  const std::shared_ptr<DatasetBase> input_;

  framework::StdThreadPool *pool_;
  cudaStream_t stream_;
  int num_tables_;
  std::vector<int> splits_sub_idx_;
  bool do_classify_;
  bool drop_remainder_;
  bool pinned_result_;
  bool gpu_result_;
  int device_id_;
};
#undef CASES
#undef CASE

} // namespace

std::shared_ptr<DatasetBase>
Packer::MakeDataset(const std::shared_ptr<DatasetBase> &input,
                    size_t batch_size, bool drop_remainder,
                    const std::vector<int> &pack_tables, int num_tables,
                    const std::vector<int> &ragged_ranks, int64 parallel,
                    bool pinned_result, bool gpu_result) {
  return std::make_shared<Dataset>(kDatasetName, batch_size, drop_remainder,
                                   pack_tables, num_tables, ragged_ranks, input,
                                   parallel, pinned_result, gpu_result);
}
} // namespace dataset
} // namespace column
