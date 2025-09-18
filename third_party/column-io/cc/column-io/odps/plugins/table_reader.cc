/*
 * Copyright 1999-2018 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string.h>
#include <unistd.h>
#include <unordered_set>

#include "algo/include/common/string_util.h"
#include "arrow/record_batch.h"
#include "arrow/type.h"
#include "column-io/odps/plugins/exception_to_ret.h"
#include "column-io/odps/plugins/logging.h"
#include "column-io/odps/plugins/table_reader.h"

namespace apsara {
namespace odps {
namespace algo {
namespace tf {
namespace plugins {
namespace odps {

static const char *kOdpsFieldPrefix = "__tf_0_";

TableReader::TableReader() : reader_(nullptr), record_(nullptr) {}

TableReader::~TableReader() {
  if (reader_ != nullptr) {
    delete reader_;
    reader_ = nullptr;
  }

  if (record_ != nullptr) {
    delete record_;
    record_ = nullptr;
  }
}

bool TableReader::Open(const std::string &config, const std::string &columns,
                       const std::string &delimiter) {
  SetDelimiter(delimiter);

  if (!SelectColumns(columns)) {
    return false;
  }

  conf_ = config;

  if (reader_ != nullptr) {
    delete reader_;
    reader_ = nullptr;
  }

  if (record_ != nullptr) {
    delete record_;
    record_ = nullptr;
  }

  reader_ = new ::apsara::odps::algo::TableReader(conf_);
  EXIT_IF_EXCEPTION(reader_->Open(columns_));

  if (!SelectColumns()) {
    delete reader_;
    reader_ = nullptr;
    return false;
  }

  const auto &schema = reader_->GetSchema();
  record_ = new ::apsara::odps::algo::TableRecord(schema);
  reader_->SetBufferRecord(record_);
  schema_ = schema;

  return true;
}

void TableReader::SetDelimiter(const std::string &delimiter) {
  delimiter_ = delimiter;
}

bool TableReader::SelectColumns(const std::string &columns) {
  if (!columns.empty()) {
    std::vector<std::string> vecs;
    ::apsara::odps::algo::SplitString(columns, ",", &vecs);
    for (auto &value : vecs) {
      std::string column = ::apsara::odps::algo::TrimString(value);
      if (!column.empty()) {
        columns_.emplace_back(column);
      }
    }
  }
  return true;
}

// Select common columns in the specified select columns and the
// table schema, ignore the unexisting columns in the specified
// select columns
//
// Return false if no columns have been selected, true or otherwise
bool TableReader::SelectColumns() {
  std::unordered_set<std::string> cols_set(columns_.begin(), columns_.end());
  columns_.clear();
  column_indexes_.clear();
  extractors_.clear();

  const auto &schema = reader_->GetSchema();
  int count = schema.Size();
  for (int i = 0; i < count; ++i) {
    auto col_name = schema.Name(i);
    if (cols_set.empty() || cols_set.find(col_name) != cols_set.end()) {
      columns_.emplace_back(col_name);
      column_indexes_.emplace_back(i);
      extractors_.emplace_back(GetExtrator(schema.Type(i)));
    }
  }

  if (columns_.empty()) {
    LOG(ERROR) << "Select columns for reader failed!";
    for (auto &column : cols_set) {
      LOG(ERROR) << "COLUMN: " << column;
    }
    return false;
  }

  return true;
}

void TableReader::Seek(size_t offset) {
  EXIT_IF_EXCEPTION(reader_->Seek(offset));
}

size_t TableReader::GetRowCount() {
  EXIT_IF_EXCEPTION(return reader_->GetRowCount());
}

bool TableReader::Read(std::string *line) {
  line->clear();

  bool read_status = false;
  EXIT_IF_EXCEPTION(read_status = reader_->Read());

  if (!read_status) {
    LOG(ERROR) << "Table read failed, config: " << conf_;
    return false;
  }

  for (size_t i = 0; i < extractors_.size(); ++i) {
    if (!delimiter_.empty() && i != 0) {
      line->append(delimiter_);
    }
    extractors_[i]->Extract(record_, column_indexes_[i], line);
  }

  return true;
}

std::shared_ptr<arrow::RecordBatch>
TableReader::GetRecordBatch(uint32_t count) {
  std::shared_ptr<arrow::RecordBatch> record_batch;
  EXIT_IF_EXCEPTION(record_batch = reader_->GetRecordBatch(count));
  while (record_batch != nullptr && record_batch->num_rows() == 0) {
    EXIT_IF_EXCEPTION(record_batch = reader_->GetRecordBatch(count));
  }
  return record_batch;
}

void TableReader::GetSchema(
    std::unordered_map<std::string, std::string> *schema) const {
  for (int32_t i = 0; i < schema_.Size(); ++i) {
    schema->insert({schema_.Name(i), schema_.TypeName(i)});
  }
}

size_t TableReader::GetReadBytes() {
  EXIT_IF_EXCEPTION(return reader_->GetReadBytes());
}

} // namespace odps
} // namespace plugins
} // namespace tf
} // namespace algo
} // namespace odps
} // namespace apsara
