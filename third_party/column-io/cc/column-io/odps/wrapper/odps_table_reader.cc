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

#ifdef TF_ENABLE_ODPS_COLUMN
#include "column-io/framework/status.h"

#include "absl/log/log.h"
#include "column-io/odps/wrapper/odps_table_reader.h"

#include <algorithm>

#include "arrow/record_batch.h"
#include "arrow/type.h"
#include "rapidjson/document.h"

namespace column {
namespace odps {
namespace wrapper {
static const size_t kReadLines = 128;

OdpsTableReader::OdpsTableReader(const std::string &conf, size_t start,
                                 size_t end)
    : conf_(conf), start_(start), end_(end), cur_(0), batch_size_(kReadLines) {}

Status OdpsTableReader::Init(const std::string &delimiter,
                             const std::string &columns, size_t batch_size) {
  batch_size_ = batch_size;

  auto s = reader_.Open(conf_, columns, delimiter);
  if (!s.ok()) {
    LOG(ERROR) << "Open table failed, config:" << conf_;
  }
  reader_.Seek(start_);

  return s;
}

Status OdpsTableReader::Read(char *data, size_t buf_sz, size_t *ret) {
  *ret = 0;
  if (ReachEnd()) {
    return Status::OutOfRange();
  }

  std::string line;
  if (!reader_.Read(&line)) { // Read failed
    return Status::Internal();
  }

  if (line.size() > buf_sz) { // Limited buffer
    Seek(cur_);               // Rollback reading
    LOG(ERROR) << "Buffer size is not enough";
    return Status::Internal();
  }

  ++cur_;
  std::memcpy(data, line.data(), line.size());
  *ret = line.size();
  return Status();
}

Status OdpsTableReader::ReadLine(std::string *line) {
  if (ReachEnd()) {
    return Status::OutOfRange();
  }

  if (!reader_.Read(line)) {
    LOG(ERROR) << "Read line failed";
    return Status::Internal();
  }

  ++cur_;
  return Status();
}

Status OdpsTableReader::Seek(size_t offset) {
  if (start_ + offset >= end_) {
    LOG(ERROR) << "Seek out of range";
    return Status::OutOfRange();
  }

  cur_ = offset;
  reader_.Seek(start_ + offset);
  return Status();
}

int64_t OdpsTableReader::Tell() { return cur_; }

Status OdpsTableReader::CountRecords(size_t *psize) {
  *psize = end_ - start_;
  return Status();
}

Status OdpsTableReader::ReadBatch(std::shared_ptr<arrow::RecordBatch> *batch) {
  if (ReachEnd()) {
    return Status::OutOfRange();
  }

  size_t read_num = std::min(batch_size_, end_ - start_ - cur_);
  *batch = reader_.ReadRows(read_num);
  if ((*batch) == nullptr) {
    LOG(ERROR) << "Read failed";
    return Status::Internal();
  }

  cur_ += (*batch)->num_rows();
  return Status();
}

Status OdpsTableReader::GetSchema(
    std::unordered_map<std::string, std::string> *schema) const {
  *schema = std::move(reader_.GetSchema());
  return Status();
}

size_t OdpsTableReader::GetReadBytes() { return reader_.GetReadBytes(); }

bool OdpsTableReader::ReachEnd() { return start_ + cur_ >= end_; }

} // namespace wrapper
} // namespace odps
} // namespace column

#endif // TF_ENABLE_ODPS_COLUMN
