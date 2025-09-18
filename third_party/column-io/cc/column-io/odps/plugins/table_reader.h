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

#ifndef PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_TABLE_READER_H_
#define PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_TABLE_READER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "algo/include/data_io/table_reader.h"
#include "column-io/odps/plugins/extractor.h"

namespace arrow {

class Schema;

} // namespace arrow

namespace apsara {
namespace odps {
namespace algo {
namespace tf {
namespace plugins {
namespace odps {

class TableReader {
public:
  TableReader();

  TableReader(const TableReader &reader) = delete; // UnCopable

  ~TableReader();

  // Open table reader using specified configuration
  //
  // Arguments:
  //   config: config to open the table reader
  //
  // Returns:
  //   true if success
  //   false if some error happens
  bool Open(const std::string &config, const std::string &columns,
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

  /**
   * Read count rows from odps table
   *
   * Args:
   *  count: read number of rows
   *
   * Return:
   *  A RecordBatch which contains count rows or less if there
   *  is no enough row in current file
   */
  std::shared_ptr<arrow::RecordBatch> GetRecordBatch(uint32_t count);

  void GetSchema(std::unordered_map<std::string, std::string> *schema) const;

  size_t GetReadBytes();

private:
  void SetDelimiter(const std::string &delimiter);

  bool SelectColumns(const std::string &columns);

  bool SelectColumns();

private:
  std::string conf_;
  std::string delimiter_;
  ::apsara::odps::algo::TableReader *reader_;
  ::apsara::odps::algo::TableRecord *record_;
  apsara::odps::algo::TableSchema schema_;
  std::vector<int> column_indexes_;
  std::vector<Extractor *> extractors_;
  std::vector<std::string> columns_;
};

} // namespace odps
} // namespace plugins
} // namespace tf
} // namespace algo
} // namespace odps
} // namespace apsara

#endif // PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_TABLE_READER_H_
