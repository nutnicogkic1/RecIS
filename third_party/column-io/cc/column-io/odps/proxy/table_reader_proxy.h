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

#ifndef PAIIO_CC_IO_ALGO_COLUMN_ODPS_TABLE_READER_PROXY_H_
#define PAIIO_CC_IO_ALGO_COLUMN_ODPS_TABLE_READER_PROXY_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "column-io/framework/status.h"
#include "column-io/odps/plugins/c_api.h"
#include "column-io/odps/proxy/lib_odps.h"

namespace arrow {

class RecordBatch;

} // namespace arrow

namespace column {
namespace odps {
namespace proxy {
// TableReaderProxy, a proxy to xdl::io::plugins::TableReader
class TableReaderProxy {
public:
  TableReaderProxy();

  TableReaderProxy(const TableReaderProxy &reader) = delete; // UnCopable

  ~TableReaderProxy();

  // Open table reader using specified configuration
  //
  // Arguments:
  //   config: config to open the table reader
  //   columns: selected columns, separate by ','
  //   delimiter: read values join delimiter
  //
  // Returns:
  //   Status::OK if success
  //   Status::Internal if some error happens
  Status Open(const std::string &config, const std::string &columns,
              const std::string &delimiter);

  // Seek to specified row offset, user should guarantee the offset is valid
  //
  // Arguments:
  //   offset: a valid row offset
  void Seek(size_t offset);

  // Get table row count
  //
  // Returns:
  //  Row number of the table
  size_t GetRowCount();

  // Read a row from the table with all selected columns extracted
  //
  // Arguments:
  //   line: a line to fill the the result
  //
  // Return:
  //   true: if read is successfully
  //   false: if some error happens
  bool Read(std::string *line);

  // Record num rows from the table with all selected columns
  // encoded in arrow::RecordBatch
  //
  // Arguments:
  //   num: number of rows to read
  //
  // Returns:
  //   nums rows of data encoded in arrow::RecordBatch
  std::shared_ptr<arrow::RecordBatch> ReadRows(size_t num);

  std::unordered_map<std::string, std::string> GetSchema() const;

  size_t GetReadBytes();

private:
  CAPI_TableReader *reader_;
};

} // namespace proxy
} // namespace odps
} // namespace column

#endif // PAIIO_CC_IO_ALGO_COLUMN_ODPS_TABLE_READER_PROXY_H_

#endif // TF_ENABLE_ODPS_COLUMN
