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

#include "column-io/framework/status.h"
#ifdef TF_ENABLE_ODPS_COLUMN

#include "absl/log/log.h"
#include "column-io/odps/proxy/table_reader_proxy.h"
#include "column-io/odps/proxy/utils.h"

#include <functional>
#include <unordered_set>

namespace column {
namespace odps {
namespace proxy {

TableReaderProxy::TableReaderProxy() : reader_(nullptr) {}

TableReaderProxy::~TableReaderProxy() {
  if (reader_ != nullptr) {
    auto lib = LibOdps::Load();
    if (lib->status().ok()) {
      lib->TableClose(reader_);
      reader_ = nullptr;
    }
  }
}

Status TableReaderProxy::Open(const std::string &config,
                              const std::string &columns,
                              const std::string &delimiter) {
  auto lib = LibOdps::Load();
  if (!lib->status().ok()) {
    return lib->status();
  }

  reader_ = lib->TableOpen(config.c_str(), config.size(), columns.c_str(),
                           columns.size(), delimiter.c_str(), delimiter.size());
  if (reader_ == nullptr) {
    LOG(ERROR) << "pen table reader failed, check your config!";
    return Status::Internal();
  }

  return Status();
}

void TableReaderProxy::Seek(size_t offset) {
  auto lib = LibOdps::Load();
  if (!lib->status().ok()) {
    return;
  }
  lib->TableSeek(reader_, offset);
}

size_t TableReaderProxy::GetRowCount() {
  auto lib = LibOdps::Load();
  if (!lib->status().ok()) {
    return 0;
  }
  return lib->TableGetRowCount(reader_);
}

bool TableReaderProxy::Read(std::string *line) {
  line->clear();

  auto lib = LibOdps::Load();
  if (!lib->status().ok()) {
    return false;
  }

  int ret = lib->TableRead(reader_, line);
  if (ret < 0) {
    return false;
  }

  return true;
}

std::shared_ptr<arrow::RecordBatch> TableReaderProxy::ReadRows(size_t num) {
  std::shared_ptr<arrow::RecordBatch> record_batch(nullptr);

  auto lib = LibOdps::Load();
  if (lib->status().ok()) {
    lib->TableReadBatch(reader_, static_cast<int>(num), &record_batch);
  }

  return record_batch;
}

std::unordered_map<std::string, std::string>
TableReaderProxy::GetSchema() const {
  std::unordered_map<std::string, std::string> ret;
  auto lib = LibOdps::Load();

  if (lib->status().ok()) {
    lib->GetTableSchema(reader_, &ret);
  }

  return ret;
}

size_t TableReaderProxy::GetReadBytes() {
  auto lib = LibOdps::Load();

  if (lib->status().ok()) {
    return lib->TableGetReadBytes(reader_);
  }
  return 0;
}

} // namespace proxy
} // namespace odps
} // namespace column

#endif // TF_ENABLE_ODPS_COLUMN
