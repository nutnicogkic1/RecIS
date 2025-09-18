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

#ifndef PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_RECORD_WRAPPER_H_
#define PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_RECORD_WRAPPER_H_

#include "algo/include/data_io/table_record.h"

namespace apsara {
namespace odps {
namespace algo {
namespace tf {
namespace plugins {
namespace odps {

class RecordWrapper {
public:
  explicit RecordWrapper(apsara::odps::algo::TableRecord *record);

  const apsara::odps::algo::TableSchema &GetSchema();

  void *Get(int index, int32_t *len);

  void GetDoubleValue(int index, double **ret);

  void GetIntegerValue(int index, int64_t **ret);

  void GetTimeValue(int index, int64_t **ret);

  void GetBoolValue(int index, bool **ret);

  void GetStringValue(int index, char **ret, int32_t *len);

private:
  apsara::odps::algo::TableRecord *record_;
};

} // namespace odps
} // namespace plugins
} // namespace tf
} // namespace algo
} // namespace odps
} // namespace apsara

#endif // PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_RECORD_WRAPPER_H_
