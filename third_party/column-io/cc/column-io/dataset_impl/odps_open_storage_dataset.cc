#if (_GLIBCXX_USE_CXX11_ABI != 0)
#include "column-io/dataset_impl/odps_open_storage_dataset.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "arrow/record_batch.h"
#include "arrow/type.h"
#include "column-io/dataset/formater.h"
#include "column-io/dataset/vec_tensor_converter.h"
#include "column-io/dataset_impl/schema_parser.h"
#include "column-io/framework/error_code.h"
#include "column-io/framework/status.h"
#include "column-io/open_storage/common-util/status.h"
#include "column-io/open_storage/wrapper/odps_open_storage_arrow_reader.h"
#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <tuple>
namespace column {
namespace dataset {
namespace {
column::Status StatusConvert(apsara::odps::algo::commonio::Status source) {
  ErrorCode code;
  switch(source.GetCode()) {
    case apsara::odps::algo::commonio::Status::kOk:                 code = ErrorCode::OK;                   break;
    case apsara::odps::algo::commonio::Status::kCancelled:          code = ErrorCode::CANCELLED;            break;
    case apsara::odps::algo::commonio::Status::kUnknown:            code = ErrorCode::UNKNOWN;              break;
    case apsara::odps::algo::commonio::Status::kInvalidArgument:    code = ErrorCode::INVALID_ARGUMENT;     break;
    case apsara::odps::algo::commonio::Status::kDeadlineExceeded:   code = ErrorCode::DEADLINE_EXCEEDED;    break;
    case apsara::odps::algo::commonio::Status::kNotFound:           code = ErrorCode::NOT_FOUND;            break;
    case apsara::odps::algo::commonio::Status::kAlreadyExists:      code = ErrorCode::ALREADY_EXISTS;       break;
    case apsara::odps::algo::commonio::Status::kPermissionDenied:   code = ErrorCode::PERMISSION_DENIED;    break;
    case apsara::odps::algo::commonio::Status::kResourceExhausted:  code = ErrorCode::RESOURCE_EXHAUSTED;   break;
    case apsara::odps::algo::commonio::Status::kFailedPrecondition: code = ErrorCode::FAILED_PRECONDITION;  break;
    case apsara::odps::algo::commonio::Status::kAborted:            code = ErrorCode::ABORTED;              break;
    case apsara::odps::algo::commonio::Status::kOutOfRange:         code = ErrorCode::OUT_OF_RANGE;         break;
    case apsara::odps::algo::commonio::Status::kUnimplemented:      code = ErrorCode::UNIMPLEMENTED;        break;
    case apsara::odps::algo::commonio::Status::kInternal:           code = ErrorCode::INTERNAL;             break;
    case apsara::odps::algo::commonio::Status::kUnavailable:        code = ErrorCode::UNAVAILABLE;          break;
    case apsara::odps::algo::commonio::Status::kDataLoss:           code = ErrorCode::DATA_LOSS;            break;
    case apsara::odps::algo::commonio::Status::kUnauthenticated:    code = ErrorCode::UNAUTHENTICATED;      break;
	default:	code = ErrorCode::UNKNOWN;              break;
  }
  absl::string_view view(source.GetMsg());
  column::Status target(code, view);
  return target; 
}
const std::string kDatasetName = "OdpsOpenStorage";
class Dataset : public DatasetBase {
public:
  Dataset(const std::string &name,
          const std::vector<std::string> &paths,
          bool is_compressed,
          int64_t batch_size,
          const std::vector<std::string> &selected_columns,
          const std::vector<std::string> &input_columns,
          const std::vector<std::string> &hash_features,
          const std::vector<std::string> &hash_types,
          const std::vector<int32_t> &hash_buckets,
          const std::vector<std::string> &dense_features,
          const std::vector<Tensor> &dense_defaults,
          bool use_xrec)
      : DatasetBase(name), paths_(std::move(paths)),
        input_columns_(input_columns), batch_size_(batch_size),
        selected_columns_(std::move(selected_columns)),
        hash_features_(hash_features),
        hash_types_(hash_types),
        hash_buckets_(hash_buckets),
        dense_features_(dense_features),
        dense_defaults_(dense_defaults),
        is_compressed_(is_compressed),
        ds_name_(name),
        use_xrec_(use_xrec) {
        // 行存输出格式下out_tensors顺序和selected_columns_需一致. dataset.py:GetNextFromIterator 也有这个参数的逻辑 
        std::string row_mode = std::getenv("ODPS_DATASET_ROW_MODE") ? std::getenv("ODPS_DATASET_ROW_MODE") : "0";
        row_mode_tensor_split_ = (row_mode == "1") ;
  }

  std::shared_ptr<IteratorBase>
  MakeIteratorInternal(const std::string &prefix) override {
    return std::shared_ptr<IteratorBase>(
        new Iterator(std::dynamic_pointer_cast<Dataset>(shared_from_this()),
                     absl::StrCat(prefix, "::OpenStorageDataset"), ds_name_));
  }

private:
  class Iterator : public DatasetIterator<Dataset> {
  public:
    explicit Iterator(const std::shared_ptr<Dataset> datataset,
                      const std::string &prefix, const std::string &ds_name)
        : DatasetIterator<Dataset>({datataset, prefix}) {
      reach_end_ = false;
    }

    Status GetNextInternal(std::vector<Tensor> *out_tensors,
                           bool *end_of_sequence,
                           std::vector<size_t> *outputs_row_spliter = nullptr) override {
      std::lock_guard<std::mutex> l(mu_);
      do {
        // Using TableDataConnector to process all files.
        // Multi-thread prefetched could be done in the Connector.
        if (reach_end_) {
          *end_of_sequence = true;
          return Status::OK();
        }
        if (!(status_.ok())) {
          return status_;
        }
        if (reader_) {
          *end_of_sequence = false;
          std::shared_ptr<arrow::RecordBatch> data;
		  //Status s;
          uint64_t rows_before_read = reader_->Tell();
          auto common_st = reader_->ReadBatch(data);
          uint64_t read_size = reader_->Tell() - rows_before_read;
          if (common_st.Ok()) {
            RETURN_IF_ERROR(InitSchema(data));
            std::vector<std::shared_ptr<arrow::RecordBatch>> formated_data;
            RETURN_IF_ERROR(formater_->FormatSample(data, &formated_data));
            RETURN_IF_ERROR(formater_->FlatConvert(formated_data, out_tensors));

            if (dataset()->row_mode_tensor_split_) {
                if( outputs_row_spliter == nullptr) {
                    throw std::runtime_error("outputs_row_spliter is nullptr when getting row mode batches");
                }
                std::vector<Tensor> temp_tensor_order;
                outputs_row_spliter->emplace_back(0);
                const auto& spliter_map = formater_->schema().flatconvert_tensor_spliter; // std::unordered_map<std::string, std::pair<size_t, size_t>>
                for (const std::string& col_name : dataset()->selected_columns_) {
                    size_t begin = 0, end = 0;
                    std::tie(begin, end) = spliter_map.at(col_name);
                    temp_tensor_order.insert(temp_tensor_order.end(), (*out_tensors).begin()+begin, (*out_tensors).begin()+end);
                    outputs_row_spliter->emplace_back(temp_tensor_order.size());
                }
                out_tensors->swap(temp_tensor_order);
            }
          } else {
            if (common_st.GetCode() ==
                apsara::odps::algo::commonio::Status::kOutOfRange) {
                if (dataset()->use_xrec_) {
                  LOG(INFO) << "OutOfRange: " << common_st.GetMsg()
                            << " of slice: " << dataset()->paths_[file_cur_];
                }
            } else {
              LOG(ERROR) << "errors::Internal: " << common_st.GetMsg();
            }
          }
          if (common_st.Ok()) {
            return StatusConvert(common_st);
          }
		  // deal with errors
          reader_.reset();
		  if (!common_st.GetCode() == apsara::odps::algo::commonio::Status::kOutOfRange) {
            return StatusConvert(common_st);
          }
          ++file_cur_;		  
        } else {
          if (file_cur_ >= dataset()->paths_.size()) {
            reach_end_ = true;
            continue;
          }
          // open reader
          int32_t batch_size = dataset()->batch_size_;
          if (dataset()->is_compressed_) {
            batch_size = std::max(1, dataset()->batch_size_ / 8);
          }
          // for match nagative samples table, and openstorage not allow exceed 20000
          batch_size = std::min(batch_size, 1024);// for better performance, changed from 20000 to 1024.
          if (dataset()->use_xrec_) {
            LOG(INFO) << "launch open file: " << dataset()->paths_[file_cur_]
                     << ", batch_row_num: " << std::to_string(batch_size);
          }
          auto algo_st = apsara::odps::tunnel::algo::tf::OdpsOpenStorageArrowReader::CreateReader(
                           dataset()->paths_[file_cur_], batch_size, dataset()->ds_name_, reader_);
          if (!algo_st.Ok()) {
            status_ = Status::InvalidArgument(
                "Create odps file reader failed: ", dataset()->paths_[file_cur_]);
            LOG(ERROR) << algo_st.GetMsg();
            return status_;
          }
		  if (begin_cur_ >= 0) {  // begin_cur_ is inited from RestoreInternal
            // validate begin_cur_
            size_t table_size;
            algo_st = apsara::odps::tunnel::algo::tf::OdpsOpenStorageArrowReader::GetTableSize(dataset()->paths_[file_cur_],
                table_size);
            if (!algo_st.Ok()) {
              status_ = Status::InvalidArgument(algo_st.GetMsg());
              reader_.reset();
              return status_;
            }
            if (begin_cur_ == table_size) {
              LOG(INFO) << "file: " << dataset()->paths_[file_cur_] <<
                  " reached end, skip. begin_cur_:  " << begin_cur_ <<
                  ", table_size: " << table_size;
              reader_.reset();
              ++file_cur_;
              begin_cur_ = -1;
              continue;
            }
            // seek
            LOG(INFO) << "seek file " << dataset()->paths_[file_cur_] <<
                " to " << begin_cur_;
            if (!reader_->Seek(begin_cur_).Ok()) {
              status_ = Status::InvalidArgument(
                  "Fail to seek path: ", dataset()->paths_[file_cur_],
                  ", to offset: ", begin_cur_);
              reader_.reset();
              return status_;
            }
            begin_cur_ = -1;
          }
        }
      } while (true);
    }

  protected:
    Status SaveInternal(IteratorStateWriter *writer) override {
      std::lock_guard<std::mutex> l(mu_);
      RETURN_IF_ERROR(writer->WriteInt(fullname("file_cur_"), file_cur_));
      LOG(INFO) << "save file_cur_: " << file_cur_;
      if (reader_) {
        RETURN_IF_ERROR(
            writer->WriteInt(fullname("begin_cur_"), reader_->Tell()));
        LOG(INFO) << "save begin_cur_: " << reader_->Tell();
      } else {
        RETURN_IF_ERROR(writer->WriteInt(fullname("begin_cur_"), begin_cur_));
        LOG(INFO) << "reader_ is null, save begin_cur_: " << begin_cur_;
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorStateReader *reader) override {
      std::lock_guard<std::mutex> l(mu_);
      RETURN_IF_ERROR(reader->ReadInt(fullname("file_cur_"), file_cur_));
      LOG(INFO) << "restore file_cur_: " << file_cur_;
      RETURN_IF_ERROR(reader->ReadInt(fullname("begin_cur_"), begin_cur_));
      LOG(INFO) << "restore begin_cur_: " << begin_cur_;
      return Status::OK();
    }

  private:
    Status InitSchema(std::shared_ptr<arrow::RecordBatch> &data) {
      if (formater_) {
        return Status::OK();
      }
      // init formater
      formater_ = ColumnDataFormater::GetColumnDataFormater(
          dataset()->is_compressed_, false, dataset()->row_mode_tensor_split_);
      std::unordered_set<std::string> selected_columns;
      selected_columns.reserve(dataset()->selected_columns_.size());
      selected_columns.insert(dataset()->selected_columns_.begin(),
                              dataset()->selected_columns_.end());
      auto st = formater_->InitSchema(
          data->schema(), dataset()->hash_features_, dataset()->hash_types_, dataset()->hash_buckets_,
          dataset()->dense_features_, dataset()->dense_defaults_, selected_columns);
      if (!st.ok()) {
        formater_.reset();
        return st;
      }
      return Status::OK();
    }
    std::mutex mu_;
    int64_t file_cur_{0};
    int64_t begin_cur_{-1};
    std::shared_ptr<apsara::odps::tunnel::algo::tf::OdpsOpenStorageArrowReader> reader_;
    std::unique_ptr<ColumnDataFormater> formater_;
    bool reach_end_;
    Status status_;
  };
  const std::vector<std::string> paths_;
  const std::vector<std::string> input_columns_;
  const std::vector<std::string> selected_columns_;
  const std::vector<std::string> hash_features_;
  const std::vector<std::string> hash_types_;
  const std::vector<int32_t> hash_buckets_;
  const std::vector<std::string> dense_features_;
  std::vector<Tensor> dense_defaults_;
  bool row_mode_tensor_split_;
  bool is_compressed_;
  int32_t batch_size_;
  std::string ds_name_;
  bool use_xrec_;
};

const std::string kIndicator = "_indicator";
Status GetInputColumnsFromOdpsOpenStorageSchema(const std::string& path,
    const std::unordered_set<std::string> &selected_columns,
    const std::vector<std::string> &dense_features,
    bool is_compressed,
    std::vector<std::string>* input_columns_from_schema) {
  // read schema
  std::unordered_map<std::string, std::string> schema;
  auto s = apsara::odps::tunnel::algo::tf::OdpsOpenStorageArrowReader::GetSchema(path, schema);
  CHECK(s.Ok())
      << "fail to GetSchema from path: " << path << ", error: " << s.GetMsg();

  std::unordered_set<std::string> useful_names;
  for (auto &feature: selected_columns) {
    useful_names.insert(feature);
  }
  for (auto &feature: dense_features) {
    useful_names.insert(feature);
  }
  if (is_compressed) useful_names.insert(kIndicator);

  for (auto &type_info: schema) {
    std::string column_name = type_info.first;
    if (is_compressed) {
      size_t pos = column_name.find_last_of("_");
      if (pos == std::string::npos) {
        LOG(INFO) << "compressed column name has no indicator suffix, skip: " << column_name;
        continue;
      }
      std::string alias = column_name.substr(0, pos);
      if (useful_names.count(alias) == 0) {
        LOG(INFO) << "compressed column not use, skip: " << column_name;
        continue;
      }
    } else {
      if (useful_names.count(column_name) == 0) {
        LOG(INFO) << "column not use, skip: " << column_name;
        continue;
      }
    }
    input_columns_from_schema->push_back(column_name);
  }
  return Status::OK();
}

Status ReadOdpsOpenStorageRecordBatch(
    const std::string& path,
    const std::unordered_set<std::string> &selected_columns,
    const std::vector<std::string> &dense_features,
    bool is_compressed,
    std::shared_ptr<arrow::RecordBatch>* data) {
  try {
    // init reader
    std::shared_ptr<apsara::odps::tunnel::algo::tf::OdpsOpenStorageArrowReader> reader;
    std::vector<std::string> input_columns_from_schema;
    GetInputColumnsFromOdpsOpenStorageSchema(path, selected_columns, dense_features, is_compressed, &input_columns_from_schema);
    auto s = apsara::odps::tunnel::algo::tf::OdpsOpenStorageArrowReader::CreateReader(
                path, 1, "SchemaReader", reader);
	CHECK(s.Ok() && reader != nullptr)
        << "failed to get reader from path: " << path << ", error:" << s.GetMsg();
    // read data
    std::shared_ptr<arrow::RecordBatch> rb;
    s = reader->ReadBatch(rb);
	CHECK(s.Ok())
        << "fail to read RecordBatch from path: " << path << ", error: " << s.GetMsg();
    //LOG(INFO) << "read data, schema: " << rb->schema()->ToString().c_str();  // FIXME: log to another file
    (*data) = rb;
    return Status::OK();
  } catch (const std::exception &ex) {
    CHECK(false) << "catch odps exception: " << ex.what();
  }
}
} // namespace

absl::StatusOr<std::shared_ptr<DatasetBase>>
OdpsOpenStorageDataset::MakeDataset(
    const std::vector<std::string> &paths,
    bool is_compressed,
    int64_t batch_size,
    const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int32_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults,
    bool use_xrec) {
  return std::shared_ptr<DatasetBase>(
      new Dataset(kDatasetName, paths, is_compressed, batch_size,
                  selected_columns, input_columns, hash_features,
                  hash_types, hash_buckets, dense_columns,
                  detail::VecsToTensor<float>(dense_defaults),
                  use_xrec));
}

std::shared_ptr<DatasetBuilder> OdpsOpenStorageDataset::MakeBuilder(
    bool is_compressed,
    int64_t batch_size,
    const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int32_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults,
    bool use_xrec) {
  return DatasetBuilder::Make(
      [=](const std::string &path)
          -> absl::StatusOr<std::shared_ptr<DatasetBase>> {
        return MakeDataset({path}, is_compressed, batch_size, selected_columns,
                           input_columns, hash_features, hash_types, hash_buckets,
                           dense_columns, dense_defaults, use_xrec);
      });
}

std::pair<
    std::vector<std::string>,
    std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>>
OdpsOpenStorageDataset::ParseSchema(
    const std::vector<std::string> &paths,
    bool is_compressed,
    const std::unordered_set<std::string> &selected_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int32_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults) {
  auto parser = SchemaParser::Make(ReadOdpsOpenStorageRecordBatch);
  return parser->ParseSchema(paths, is_compressed, selected_columns,
                             hash_features, hash_types, hash_buckets, dense_columns,
                             detail::VecsToTensor(dense_defaults));
}

Status OdpsOpenStorageDataset::GetTableSize(const std::string &path,
                                            size_t *ret) {
  return StatusConvert(apsara::odps::tunnel::algo::tf::OdpsOpenStorageArrowReader::GetTableSize(path, *ret)); 
}

int64_t OdpsOpenStorageDataset::GetSessionExpireTimestamp(const std::string &session_id) {
  return apsara::odps::tunnel::algo::tf::OdpsOpenStorageArrowReader::GetSessionExpireTimestamp(session_id);
}

} // namespace dataset
} // namespace column

#endif
