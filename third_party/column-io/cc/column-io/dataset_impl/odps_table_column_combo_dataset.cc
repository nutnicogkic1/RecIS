#if(_GLIBCXX_USE_CXX11_ABI == 0)

#include "column-io/dataset_impl/odps_table_column_combo_dataset.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "arrow/record_batch.h"
#include "arrow/type.h"
#include <arrow/array.h>
#include "column-io/dataset/formater.h"
#include "column-io/dataset/vec_tensor_converter.h"
#include "column-io/dataset_impl/schema_parser.h"
#include "column-io/framework/error_code.h"
#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"
#include "column-io/framework/types.h"
#include "column-io/odps/wrapper/odps_table_file_system.h"
#include "column-io/odps/wrapper/odps_table_reader.h"
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
namespace column{
namespace dataset{
namespace {
const std::string kDatasetName = "OdpsTableColumnCombo";
class Dataset : public DatasetBase {
public:
  Dataset(const std::string &name,
          const std::vector<std::vector<std::string>> &paths,
          bool is_compressed, int64_t batch_size,
          const std::vector<std::string> &selected_columns,
          const std::vector<std::vector<std::string>> &input_columns,
          const std::vector<std::string> &hash_features,
          const std::vector<std::string> &hash_types,
          const std::vector<int32_t> &hash_buckets,
          const std::vector<std::string> &dense_features,
          const std::vector<Tensor> &dense_defaults,
          bool check_data,
          std::string primary_key)
      : DatasetBase(name), paths_(std::move(paths)),
        input_columns_(input_columns),
        batch_size_(batch_size),
        selected_columns_(std::move(selected_columns)),
        hash_features_(hash_features),
        hash_types_(hash_types),
        hash_buckets_(hash_buckets),
        dense_features_(dense_features),
        dense_defaults_(dense_defaults),
        is_compressed_(is_compressed),
        ds_name_(name),
        check_data_(check_data),
        primary_key_(primary_key) {
    fs_ = odps::wrapper::OdpsTableFileSystem::Instance();
  }

  std::shared_ptr<IteratorBase>
  MakeIteratorInternal(const std::string &prefix) override {
    return std::shared_ptr<IteratorBase>(
        new Iterator(std::dynamic_pointer_cast<Dataset>(shared_from_this()),
                     absl::StrCat(prefix, "::TableColumnDataset"), ds_name_));
  }

private:
  class Iterator : public DatasetIterator<Dataset> {
  public:
    explicit Iterator(const std::shared_ptr<Dataset> datataset,
                      const std::string &prefix, const std::string &ds_name)
        : DatasetIterator<Dataset>({datataset, prefix}) {
      reach_end_ = false;
      table_num_ = dataset()->paths_[0].size();
      readers_.resize(table_num_);
      for(size_t i =0;i < table_num_; ++i){
        std::vector<std::string> tmp;
        // selected_table_columns_.resize(table_num_);
        selected_table_columns_.push_back(tmp);
      }
      column_formaters_.resize(table_num_);
    }

    Status InitSchema(std::shared_ptr<arrow::RecordBatch> &data, size_t table_idx) {
      if (column_formaters_[table_idx])
        return Status::OK();
      // init formater
      column_formaters_[table_idx] = ColumnDataFormater::GetColumnDataFormater(
          dataset()->is_compressed_, false);
      std::unordered_set<std::string> input_columns;
      const auto& cols = dataset()->input_columns_[table_idx];
      input_columns.reserve(cols.size());
      input_columns.insert(cols.begin(), cols.end());
      auto st = column_formaters_[table_idx]->InitSchema(
          data->schema(), dataset()->hash_features_,
          dataset()->hash_types_, dataset()->hash_buckets_,
          dataset()->dense_features_, dataset()->dense_defaults_,
          input_columns);
      if (!st.ok()) {
        column_formaters_[table_idx].reset();
        return st;
      }

      // Set selected_table_columns_ for following table readers
      auto schema = column_formaters_[table_idx]->schema();
      for (auto iter = schema.alias_map.begin(); iter != schema.alias_map.end(); ++iter) {
        selected_table_columns_[table_idx].push_back(iter->first);
      }

      return Status::OK();
    }

    Status ReadEachTable(std::vector<std::shared_ptr<arrow::RecordBatch>> &datas, 
                        size_t &min_batch_rows, size_t &min_reader_idx, bool &all_batch_size_same){
      auto s = Status::OK();
      for(size_t i = 0 ; i < table_num_; ++i){
          // uint64_t size_before_read = reader_->GetReadBytes();
          auto& reader_ = readers_.at(i);
          if (!reader_) {
            s = Status::InvalidArgument("reader ", i, " terminated unexpectedly.");
            return s;
          }
          auto& data = datas[i];
          auto s = reader_->ReadBatch(&data);
          // uint64_t read_size = reader_->GetReadBytes() - size_before_read;
          if (s.ok()) {
            if (data->num_rows() == 0) {
                s = Status::InvalidArgument("Get empty batch for reader ", i, ", please check data.");
                return s;
              }
            else if (min_reader_idx == -1) {
              min_batch_rows = data->num_rows();
              min_reader_idx = i;
            }
            else if (min_batch_rows != data->num_rows()) {
              all_batch_size_same = false;
              if (min_batch_rows > data->num_rows()) {
                min_batch_rows = data->num_rows();
                min_reader_idx = i;
              }
            }
            
          }else{
            // deal with errors
            clear_readers();
            if (s.code() != ErrorCode::OUT_OF_RANGE) {
              return s;
            }
          }
          
          // ++file_cur_;
        } // /end for read each table
      return s;
    }

    Status SeekEachTable(std::vector<std::shared_ptr<arrow::RecordBatch>> &datas,
                         size_t min_reader_idx, size_t min_batch_rows){
      auto seek_offset = readers_[min_reader_idx]->Tell();
      auto s = Status::OK();
        for (size_t i = 0; i < table_num_; ++i) {
          if (datas[i]->num_rows() > min_batch_rows) {
            auto& reader = readers_.at(i);
            auto sb_st = reader->Seek(seek_offset);
            if (!sb_st.ok()) {
              s = Status::InvalidArgument("Seek back failed for reader ", i);
              break;
            }
            datas[i] = datas[i]->Slice(0, min_batch_rows);
          }
        }
      return s;
    }

    Status CheckPrimaryKey( std::vector<std::shared_ptr<arrow::RecordBatch>> &datas){
      auto s = Status::OK();
      auto main_col = datas[0]->GetColumnByName(dataset()->primary_key_);
        if (!main_col) {
          s = Status::InvalidArgument("Primary key column `", dataset()->primary_key_,
                                "` not existed in the first input table.");
        }
        else {
          for (size_t i = 1; i < datas.size(); ++i) {
            auto col = datas[i]->GetColumnByName(dataset()->primary_key_);
            if(!main_col->Equals(col)) {
              s = Status::InvalidArgument("Primary key check failed between reader ",
                                    i, " and reader 0.");
              break;
            }
          }
        }
      return s;
    }

    Status FormatTableSample( std::vector<std::map<std::string, std::vector<std::vector<Tensor>>>> &merged_conv_output,
                              std::vector<std::shared_ptr<arrow::RecordBatch>> &datas){
      auto s = Status::OK();
      for(size_t i = 0 ;i < table_num_; ++i){
        auto& data = datas[i];
        if(!column_formaters_[i]){
          s = InitSchema(data, i);
        }
        if(!s.ok()){
          LOG(ERROR) <<"InitSchema error for reader "<<i;
          return s;
        }
        std::vector<std::shared_ptr<arrow::RecordBatch>> formated_data;
        s = column_formaters_[i]->FormatSample(data, &formated_data);
        if(!s.ok()){
          LOG(ERROR) <<"FormatSample error for reader "<<i;
          return s;
        }
        std::vector<std::map<std::string, std::vector<std::vector<Tensor>>>> conv_output;
        s = column_formaters_[i]->FlatConvert(formated_data, &conv_output);
        if(!s.ok()){
          LOG(ERROR) <<"Error when flattening data batch from reader "<<i;
          return s;
        }
        if(merged_conv_output.size() == 0) {
          merged_conv_output.resize(conv_output.size());
        } else {
          if (merged_conv_output.size() != conv_output.size()) {
            s = Status::InvalidArgument("Sample type is different among tables!");
            return s;
          }
        }

        for (size_t j = 0; j < merged_conv_output.size(); ++j) {
          merged_conv_output[j].insert(conv_output[j].begin(), conv_output[j].end());
        }
      } // end for format each table sample
      return s;
    }

    Status InitReaders(){
      auto status_ = Status::OK();
      size_t target_table_size_current_group = 0;
      for(size_t i = 0; i < table_num_; ++i){
        auto *fs = dataset()->fs_;
        column::framework::ColumnReader *raw_reader = nullptr;
        int32_t batch_size = dataset()->batch_size_;
        if (dataset()->is_compressed_)
          batch_size = std::max(1, dataset()->batch_size_ / 8);
        LOG(INFO) << "launch open file: " << dataset()->paths_[file_cur_][i];
        std::vector<std::string> selected_table_column = selected_table_columns_[i];
        auto algo_st =
            fs->CreateFileReader(dataset()->paths_[file_cur_][i], &raw_reader,
                                  batch_size, dataset()->input_columns_[i]);
        if (!algo_st.ok()) {
          status_ =
              Status::InvalidArgument("Create odps file reader failed: ",
                                      dataset()->paths_[file_cur_][i]);
          return status_;
        }
        auto odps_reader =
            dynamic_cast<column::odps::wrapper::OdpsTableReader *>(
                raw_reader);
        if (odps_reader == nullptr) {
          status_ = Status::InvalidArgument("Cast odps file reader failed: ",
                                            dataset()->paths_[file_cur_][i]);
          return status_;
        }
        readers_[i].reset(odps_reader);
        if (begin_cur_ >= 0) { // begin_cur_ is inited from RestoreInternal
          // validate begin_cur_
          size_t table_size;
          readers_[i]->CountRecords(&table_size);
          if (i == 0) {
            target_table_size_current_group = table_size;
          }
          else if (target_table_size_current_group != table_size){
            status_ = Status::InvalidArgument(
                "Table ", i, " has different size from ",
                "Table 0: ", table_size, " vs ",
                target_table_size_current_group);
            clear_readers();
            break;
          }
          if (begin_cur_ == table_size) {
            LOG(INFO) << "file: " << dataset()->paths_[file_cur_][i]
                      << " reached end, skip. begin_cur_:  " << begin_cur_
                      << ", table_size: " << table_size;
            clear_readers();
            ++file_cur_;
            begin_cur_ = -1;
            break;
          }
          // seek
          if (!readers_[i]->Seek(begin_cur_).ok()) {
            status_ = Status::InvalidArgument(
                "Fail to seek path: ", dataset()->paths_[file_cur_][i],
                ", to offset: ", begin_cur_);
            clear_readers();
            break;
          }
          begin_cur_ = -1;
        }  // end if (begin_cur_ >= 0)
      }  // end for (size_t i = 0; i < table_num_; ++i)
      return status_;
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
        if (readers_[0]) {
          Status s = Status::OK();
          *end_of_sequence = false;
          std::vector<std::shared_ptr<arrow::RecordBatch>> datas;
          datas.resize(table_num_);

          size_t min_batch_rows = 0, min_reader_idx = -1;
          bool all_batch_size_same = true;

          s = ReadEachTable(datas, min_batch_rows, min_reader_idx, all_batch_size_same);

          //Need to seek back reader
          if (s.ok() && !all_batch_size_same) {
            s = SeekEachTable(datas, min_reader_idx, min_batch_rows);
          }

          //Check row by primary key
          if (s.ok() && dataset()->check_data_) {
            s = CheckPrimaryKey(datas);
          }

          if (!s.ok() && s.code() != ErrorCode::OUT_OF_RANGE) {
              return s;
          }
          

          // Format Sample
          std::vector<std::map<std::string, std::vector<std::vector<Tensor>>>> merged_conv_output;
          s = FormatTableSample( merged_conv_output, datas);
          if (s.ok()) { // Format sample completed, return the sample and status OK
            for (auto& map : merged_conv_output) {
              for (auto& item : map) {
                for (auto& vec : item.second) {
                  out_tensors->insert(out_tensors->end(), vec.begin(), vec.end());
                }
              }
            }
            return s;
          }

          ++file_cur_;
          clear_readers();
          continue;
          
        } else {
          if (file_cur_ >= dataset()->paths_.size()) {
            reach_end_ = true;
            continue;
          }
          // open readers
          auto s = InitReaders();
          if(!s.ok()){
            LOG(ERROR) << "InitReaders failed !";
            return s;
          }
          
        }  // end GetNext else
      } while (true);
    }

    void clear_readers() {
      for (auto& reader: readers_) {
        if(reader) {
          reader.reset();
        }
      }
    }

  protected:
    Status SaveInternal(IteratorStateWriter *writer) override {
      std::lock_guard<std::mutex> l(mu_);
      RETURN_IF_ERROR(writer->WriteInt(fullname("file_cur_"), file_cur_));
      LOG(INFO) << "save file_cur_: " << file_cur_;
      if (readers_[0]) {
        RETURN_IF_ERROR(
            writer->WriteInt(fullname("begin_cur_"), readers_[0]->Tell()));
        LOG(INFO) << "save begin_cur_: " << readers_[0]->Tell();
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
    
    std::mutex mu_;
    int64_t file_cur_{0};
    int64_t begin_cur_{-1};
    // std::unique_ptr<column::odps::wrapper::OdpsTableReader> reader_;
    // std::unique_ptr<ColumnDataFormater> formater_;
    std::vector<std::unique_ptr<column::odps::wrapper::OdpsTableReader>> readers_;
    std::vector<std::unique_ptr<ColumnDataFormater>> column_formaters_;
    std::vector<std::vector<std::string>> selected_table_columns_;
    size_t table_num_;
    bool reach_end_;
    Status status_;
  };

  const std::vector<std::vector<std::string>> paths_;
  const std::vector<std::vector<std::string>> input_columns_;
  const std::vector<std::string> selected_columns_;
  const std::vector<std::string> hash_features_;
  const std::vector<std::string> hash_types_;
  const std::vector<int32_t> hash_buckets_;
  const std::vector<std::string> dense_features_;
  std::vector<Tensor> dense_defaults_;
  bool is_compressed_;
  int32_t batch_size_;
  column::odps::wrapper::OdpsTableFileSystem *fs_;
  std::string ds_name_;
  bool check_data_;
  std::string primary_key_;
};

Status ReadOdpsTableColumnBatch(const std::string &path,
                                std::shared_ptr<arrow::RecordBatch> &output) {
  column::odps::wrapper::OdpsTableFileSystem *fs =
      odps::wrapper::OdpsTableFileSystem::Instance();
  column::framework::ColumnReader *raw_reader = nullptr;
  auto algo_st = fs->CreateFileReader(path, &raw_reader, 1, {});
  if (!algo_st.ok()) {
    return algo_st;
  }
  auto odps_reader =
      dynamic_cast<column::odps::wrapper::OdpsTableReader *>(raw_reader);

  return Status::OK();
}

const std::string kIndicator = "_indicator";
Status GetInputColumnsFromOdpsSchema(
    odps::wrapper::OdpsTableFileSystem *fs, const std::string &path,
    const std::unordered_set<std::string> &selected_columns,
    const std::vector<std::string> &dense_features, bool is_compressed,
    std::vector<std::string> *input_columns_from_schema) {
  // init reader
  framework::ColumnReader *raw_reader = nullptr;
  std::vector<std::string> empty_vec;
  auto s = fs->CreateFileReader(path, &raw_reader, 1, empty_vec);
  CHECK(s.ok() && raw_reader != nullptr)
      << "Create table reader for path [" << path << "] failed";
  auto odps_reader = dynamic_cast<odps::wrapper::OdpsTableReader *>(raw_reader);
  CHECK(odps_reader != nullptr) << "fail to cast FileReader to OdpsTableReader";
  std::unique_ptr<odps::wrapper::OdpsTableReader> reader(odps_reader);
  // read schema
  std::unordered_map<std::string, std::string> schema;
  s = reader->GetSchema(&schema);
  CHECK(s.ok()) << "fail to GetSchema from path: " << path
                << ", error: " << s.error_message();

  std::unordered_set<std::string> useful_names;
  for (auto &feature : selected_columns) {
    useful_names.insert(feature);
  }
  for (auto &feature : dense_features) {
    useful_names.insert(feature);
  }
  if (is_compressed)
    useful_names.insert(kIndicator);

  for (auto &type_info : schema) {
    std::string column_name = type_info.first;
    std::string column_value = type_info.second;
    if (is_compressed) {
      size_t pos = column_name.find_last_of("_");
      if (pos == std::string::npos) {
        LOG(INFO) << "compressed column name has no indicator suffix, skip: "
                  << column_name;
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
Status
ReadOdpsRecordBatch(const std::string &path,
                    const std::unordered_set<std::string> &selected_columns,
                    const std::vector<std::string> &dense_features,
                    bool is_compressed,
                    std::shared_ptr<arrow::RecordBatch> *data) {
  // init fs
  auto fs = odps::wrapper::OdpsTableFileSystem::Instance();
  CHECK(fs) << "Init odps filesystem failed";
  // init reader
  std::vector<std::string> input_columns_from_schema;
  GetInputColumnsFromOdpsSchema(fs, path, selected_columns, dense_features,
                                is_compressed, &input_columns_from_schema);
  framework::ColumnReader *raw_reader = nullptr;
  auto s =
      fs->CreateFileReader(path, &raw_reader, 8, input_columns_from_schema);
  CHECK(s.ok() && raw_reader != nullptr)
      << "Create table reader for path [" << path << "] failed";
  auto odps_reader = dynamic_cast<odps::wrapper::OdpsTableReader *>(raw_reader);
  CHECK(odps_reader != nullptr) << "fail to cast FileReader to OdpsTableReader";
  std::unique_ptr<odps::wrapper::OdpsTableReader> reader(odps_reader);
  // read data
  std::shared_ptr<arrow::RecordBatch> rb;
  s = reader->ReadBatch(&rb);
  CHECK(s.ok()) << "fail to read RecordBatch from path: " << path
                << ", error: " << s.error_message();
  LOG(INFO) << "read data, schema: " << rb->schema()->ToString().c_str();
  (*data) = rb;
  return Status::OK();
}

}  //namespace


absl::StatusOr<std::shared_ptr<DatasetBase>>
OdpsTableColumnComboDataset::MakeDataset(
    const std::vector<std::vector<std::string>> &paths, bool is_compressed,
    int64_t batch_size, const std::vector<std::string> &selected_columns,
    const std::vector<std::vector<std::string>> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int32_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults,
    const bool &check_data,
    const std::string &primary_key) {
  return std::shared_ptr<DatasetBase>(
      new Dataset(kDatasetName, paths, is_compressed, batch_size,
                  selected_columns, input_columns, hash_features,
                  hash_types, hash_buckets, dense_columns,
                  detail::VecsToTensor<float>(dense_defaults),
                  check_data, primary_key));
}

std::shared_ptr<DatasetBuilder> OdpsTableColumnComboDataset::MakeBuilder(
    bool is_compressed, int64_t batch_size,
    const std::vector<std::string> &selected_columns,
    const std::vector<std::vector<std::string>> &input_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int32_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults,
    const bool &check_data,
    const std::string &primary_key) {
  return DatasetBuilder::Make(
      [=](const std::vector<std::string> &paths)
          -> absl::StatusOr<std::shared_ptr<DatasetBase>> {
        return MakeDataset({paths}, is_compressed, batch_size, selected_columns,
                           input_columns, hash_features, hash_types, hash_buckets,
                           dense_columns, dense_defaults, check_data, primary_key);
      });
}

std::pair<
    std::vector<std::string>,
    std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>>
OdpsTableColumnComboDataset::ParseSchema(
    const std::vector<std::string> &paths, bool is_compressed,
    const std::unordered_set<std::string> &selected_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int32_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<std::vector<float>> &dense_defaults) {
  auto parser = SchemaParser::Make(ReadOdpsRecordBatch);
  return parser->ParseSchema(paths, is_compressed, selected_columns,
                             hash_features, hash_types, hash_buckets, dense_columns,
                             detail::VecsToTensor(dense_defaults));
}

Status OdpsTableColumnComboDataset::GetTableSize(const std::string &path,
                                            size_t *ret) {
  auto fs = odps::wrapper::OdpsTableFileSystem::Instance();
  return fs->GetFileSize(path, ret);
}

}
}

#endif
