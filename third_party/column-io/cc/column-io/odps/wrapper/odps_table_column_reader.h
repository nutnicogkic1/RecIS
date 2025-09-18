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

#ifndef PAIIO_CC_IO_ALGO_COLUMN_COLUMN_READER_H_
#define PAIIO_CC_IO_ALGO_COLUMN_COLUMN_READER_H_

#include <memory>
#include <string>
#include <vector>

#include "arrow/type.h"
#include "column-io/odps/wrapper/odps_table_schema.h"

namespace column {
namespace odps {
namespace wrapper {
enum FeatureType {
  kDense = 0,
  kSparse,
  kWeightedSparse,
  kSeqSparse,
  kSeqWeightedSparse,
  kUnknow
};

class OdpsTableColumnReader {
public:
  virtual ~OdpsTableColumnReader(){};

  virtual bool ReadVec(const char **str, size_t *length) = 0;

  virtual bool ReadVec(const int64_t **data, size_t *length) = 0;
  virtual bool ReadVec(const float **data, size_t *length) = 0;
  virtual bool ReadVec(const double **data, size_t *length) = 0;

  virtual bool ReadMap(const int64_t **keys, const float **values,
                       size_t *length) = 0;
  virtual bool ReadMap(const int64_t **keys, const double **values,
                       size_t *length) = 0;

  virtual bool ReadMatrix(const char **data, size_t *length,
                          std::vector<size_t> *segments) = 0;
  virtual bool ReadMatrix(const int64_t **data, size_t *length,
                          std::vector<size_t> *segments) = 0;

  virtual bool HasIndicator() const = 0;
  virtual std::string indicator_name() const = 0;

  virtual bool GetIndicator(const int64_t **data, size_t *length) = 0;

  // Retrieve the innest value type for the Data Array
  // Eg: array<array<double>>             =>  arrow::Type::DOUBLE
  //     List<Struct<k:int64, v:float>>   =>  arrow::Type::FLOAT
  virtual arrow::Type::type value_type() const = 0;

  virtual FeatureType feature_type() const = 0;

  // Reset to begin pos
  virtual bool Reset() = 0;
};

std::unique_ptr<OdpsTableColumnReader>
NewColumnReader(const std::shared_ptr<arrow::RecordBatch> &record_batch,
                const OdpsTableSchema &schema, const std::string &column,
                bool compressed, bool is_large_list);

inline std::unique_ptr<OdpsTableColumnReader>
NewColumnReader(const std::shared_ptr<arrow::RecordBatch> &record_batch,
                const OdpsTableSchema &schema, const std::string &column,
                bool compressed) {
  return NewColumnReader(record_batch, schema, column, compressed, true);
}

} // namespace wrapper
} // namespace odps
} // namespace column

#endif // PAIIO_CC_IO_ALGO_COLUMN_COLUMN_READER_H_

#endif // TF_ENABLE_ODPS_COLUMN
