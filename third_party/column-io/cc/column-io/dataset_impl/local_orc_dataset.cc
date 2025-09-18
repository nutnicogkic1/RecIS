#include "column-io/dataset_impl/local_orc_dataset.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "arrow/buffer.h"
#include "arrow/io/api.h"
#include <arrow/adapters/orc/adapter.h>
#include "arrow/record_batch.h"
#include "arrow/type.h"
#include "arrow/type_fwd.h"
#include "column-io/dataset/formater.h"
#include "column-io/dataset/vec_tensor_converter.h"
#include "column-io/dataset_impl/schema_parser.h"
#include "column-io/framework/error_code.h"
#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"
#include <cstddef>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
namespace column {
namespace dataset {
namespace {
class LocalFileReader {
public:
  static Status OpenFile(const std::string file_path,
                         const std::vector<std::string> &selected_columns,
                         std::unique_ptr<LocalFileReader> *out) {
    (*out).reset(new LocalFileReader(file_path, selected_columns));
    return (*out)->Init();
  }

  Status ReadRecordBatch(std::shared_ptr<arrow::RecordBatch> *out) {
    auto st = reader_->ReadNext(out);
    if (!st.ok()) {
      return Status::Internal(st.message());
    } else if (!(*out)) {
      return Status::OutOfRange();
    }
    return Status::OK();
  }

  Status Seek(int64_t index) {
    while (index != index_) {
      std::shared_ptr<arrow::RecordBatch> rb;
      auto st = ReadRecordBatch(&rb);
      if (!st.ok()) {
        return st;
      }
      index_++;
    }
    return Status::OK();
  }

  int64_t Tell() { return index_; }

private:
  LocalFileReader(const std::string file_path,
                  const std::vector<std::string> &selected_columns)
      : file_path_(file_path),
        selected_columns_(selected_columns),
        index_(0) {
  }

  Status Init() {
    auto file = arrow::io::ReadableFile::Open(file_path_);
    if (!file.ok()) {
      return Status::Internal(file.status().message());
    }
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    auto orc_reader = arrow::adapters::orc::ORCFileReader::Open(file.ValueOrDie(), pool);
    if (!orc_reader.ok()) {
      return Status::Internal(orc_reader.status().message());
    }
    orc_reader_ = std::move(orc_reader).ValueOrDie();
    InitSchema();
    auto reader = orc_reader_->GetRecordBatchReader(1024, selected_columns_);
    if (!reader.ok()) {
      return Status::Internal(reader.status().message());
    }
    reader_ = std::move(reader).ValueOrDie();
    return Status::OK();
  }

  Status InitSchema() {
    auto schema = orc_reader_->ReadSchema();
     if (!schema.ok()) {
      return Status::Internal(schema.status().message());
    }
    schema_ = schema.ValueOrDie();
    return Status::OK();
  }

  std::string file_path_;
  std::vector<std::string> selected_columns_;
  std::shared_ptr<arrow::adapters::orc::ORCFileReader> orc_reader_;
  std::shared_ptr<arrow::RecordBatchReader> reader_;
  std::shared_ptr<arrow::Schema> schema_;
  int64_t index_;
};

const std::string kDatasetName = "LocalOrcDataset";
class Dataset : public DatasetBase {
public:
  Dataset(const std::string &name, const std::vector<std::string> &paths,
          bool is_compressed, int64_t batch_size,
          const std::vector<std::string> &selected_columns,
          const std::vector<std::string> &input_columns,
          const std::vector<std::string> &hash_features,
          std::vector<std::string> hash_types,
          std::vector<int32_t> hash_buckets,
          const std::vector<std::string> &dense_columns,
          const std::vector<Tensor> &dense_defaults)
      : DatasetBase(name), paths_(paths), is_compressed_(is_compressed),
        batch_size_(batch_size), selected_columns_(selected_columns),
        input_columns_(input_columns), hash_features_(hash_features), hash_types_(hash_types), hash_buckets_(hash_buckets),
        dense_columns_(dense_columns), dense_defaults_(dense_defaults) {}

protected:
  std::shared_ptr<IteratorBase>
  MakeIteratorInternal(const std::string &prefix) override {
    return std::make_shared<Iterator>(
        std::dynamic_pointer_cast<Dataset>(shared_from_this()), prefix);
  }

private:
  class Iterator : public DatasetIterator<Dataset> {
  public:
    Iterator(const std::shared_ptr<Dataset> &dataset, const std::string &prefix)
        : DatasetIterator<Dataset>({dataset, prefix}), index_(0) {}

  protected:
    Status GetNextInternal(std::vector<Tensor> *outputs,
                           bool *end_of_sequence,
                           std::vector<size_t> *outputs_row_spliter = nullptr) {
      std::lock_guard<std::mutex> l(mu_);
      if (index_ >= dataset()->paths_.size()) {
        *end_of_sequence = true;
        return Status::OK();
      }
      while (true) {
        if (!reader_) {
          RETURN_IF_ERROR(
              LocalFileReader::OpenFile(dataset()->paths_[index_], dataset()->selected_columns_, &reader_));
        }
        std::shared_ptr<arrow::RecordBatch> rb;
        auto st = reader_->ReadRecordBatch(&rb);
        if (st.code() == ErrorCode::OUT_OF_RANGE) {
          if (index_ == dataset()->paths_.size() - 1) {
            *end_of_sequence = true;
            return Status::OutOfRange();
          } else {
            reader_.reset();
            index_++;
            continue;
          }
        } else if (!st.ok()) {
          return st;
        }
        st = (InitSchema(rb));
        outputs->clear();
        std::vector<std::shared_ptr<arrow::RecordBatch>> formated_data;
        st = (formater_->FormatSample(rb, &formated_data));
        st = (formater_->FlatConvert(formated_data, outputs));
        break;
      }
      return Status::OK();
    }

    Status SaveInternal(IteratorStateWriter *writer) {
      std::lock_guard<std::mutex> l(mu_);
      RETURN_IF_ERROR(writer->WriteInt(fullname("index"), index_));
      return writer->WriteInt(fullname("file_cur"), reader_->Tell());
    }

    Status RestoreInternal(IteratorStateReader *reader) {
      std::lock_guard<std::mutex> l(mu_);
      int64_t index;
      RETURN_IF_ERROR(reader->ReadInt(fullname("index"), index));
      int64_t file_cur;
      RETURN_IF_ERROR(reader->ReadInt(fullname("file_cur"), file_cur));
      index_ = index;
      RETURN_IF_ERROR(
          LocalFileReader::OpenFile(dataset()->paths_[index_], dataset()->selected_columns_, &reader_));
      RETURN_IF_ERROR(reader_->Seek(file_cur));
      return Status::OK();
    }

  private:
    Status InitSchema(std::shared_ptr<arrow::RecordBatch> &data) {
      if (formater_)
        return Status::OK();
      // init formater
      formater_ = ColumnDataFormater::GetColumnDataFormater(
          dataset()->is_compressed_, false);
      std::unordered_set<std::string> selected_columns;
      selected_columns.reserve(dataset()->selected_columns_.size());
      selected_columns.insert(dataset()->selected_columns_.begin(),
                              dataset()->selected_columns_.end());
      auto st = formater_->InitSchema(
          data->schema(), dataset()->hash_features_,dataset()->hash_types_, dataset()->hash_buckets_, dataset()->dense_columns_,
          dataset()->dense_defaults_, selected_columns);
      if (!st.ok()) {
        formater_.reset();
        return st;
      }
      return Status::OK();
    }
    std::unique_ptr<LocalFileReader> reader_;
    std::unique_ptr<ColumnDataFormater> formater_;
    std::mutex mu_;
    int64 index_;
  };
  const std::vector<std::string> paths_;
  const bool is_compressed_;
  const int64_t batch_size_;
  const std::vector<std::string> selected_columns_;
  const std::vector<std::string> input_columns_;
  const std::vector<std::string> hash_features_;
  const std::vector<std::string> hash_types_;
  const std::vector<int32_t> hash_buckets_;
  const std::vector<std::string> dense_columns_;
  const std::vector<Tensor> dense_defaults_;
};
const std::string kIndicator = "_indicator";

Status ReadRecordBatch(const std::string &path,
                       const std::unordered_set<std::string> &selected_columns,
                       const std::vector<std::string> &dense_features,
                       bool is_compressed,
                       std::shared_ptr<arrow::RecordBatch> *data) {
  // init reader
  std::vector<std::string> columns(selected_columns.begin(), selected_columns.end());
  std::unique_ptr<LocalFileReader> reader;
  RETURN_IF_ERROR(LocalFileReader::OpenFile(path, columns, &reader));
  auto st = reader->ReadRecordBatch(data);
  CHECK(st.ok()) << "Read batch failed at path [" << path << "]";
  LOG(INFO) << "read data, schema: " << (*(data))->schema()->ToString().c_str();
  return Status::OK();
}
} // namespace

absl::StatusOr<std::shared_ptr<DatasetBase>> LocalOrcDataset::MakeDataset(
    const std::vector<std::string> &paths, bool is_compressed,
    int64_t batch_size, const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int32_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults) {
  return std::shared_ptr<DatasetBase>(
      new Dataset(kDatasetName, paths, is_compressed, batch_size,
                  selected_columns, input_columns, hash_features,hash_types, hash_buckets, dense_columns,
                  detail::VecsToTensor(dense_defaults)));
}

std::shared_ptr<DatasetBuilder> LocalOrcDataset::MakeBuilder(
    bool is_compressed, int64_t batch_size,
    const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int32_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults) {
  return DatasetBuilder::Make(
      [=](const std::string &path)
          -> absl::StatusOr<std::shared_ptr<DatasetBase>> {
        return MakeDataset({path}, is_compressed, batch_size, selected_columns,
                           input_columns, hash_features, hash_types, hash_buckets, dense_columns,
                           dense_defaults);
      });
}

std::pair<
    std::vector<std::string>,
    std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>>
LocalOrcDataset::ParseSchema(
    const std::vector<std::string> &paths, bool is_compressed,
    const std::unordered_set<std::string> &selected_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int32_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults) {
  auto parser = SchemaParser::Make(ReadRecordBatch);
  return parser->ParseSchema(paths, is_compressed, selected_columns,
                             hash_features, hash_types, hash_buckets, dense_columns,
                             detail::VecsToTensor(dense_defaults));
}
} // namespace dataset
} // namespace column
