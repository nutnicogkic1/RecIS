#include <string>
#include <unordered_map>

#include "arrow/record_batch.h"
#include "column-io/framework/status.h"
#ifndef _COMMON_IO_COLUMN_FRAMEWORK_COLUMN_READER_H_
#define _COMMON_IO_COLUMN_FRAMEWORK_COLUMN_READER_H_
namespace column {
namespace framework {
class ColumnReader {
public:
  virtual Status Seek(size_t offset) = 0;
  virtual int64_t Tell() = 0;
  virtual Status ReadBatch(std::shared_ptr<arrow::RecordBatch> *batch) = 0;
  virtual ~ColumnReader() = default;
};
} // namespace framework
} // namespace column
#endif