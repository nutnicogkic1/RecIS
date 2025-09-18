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

#ifndef PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_EXTRACTOR_H_
#define PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_EXTRACTOR_H_

#include <string>

#include "algo/include/data_io/table_record.h"

namespace apsara {
namespace odps {
namespace algo {
namespace tf {
namespace plugins {
namespace odps {

class Extractor {
public:
  virtual ~Extractor() {}

  virtual int Extract(apsara::odps::algo::TableRecord *record, int index,
                      std::string *value) = 0;
};

Extractor *GetExtrator(apsara::odps::algo::ColumnType type);

} // namespace odps
} // namespace plugins
} // namespace tf
} // namespace algo
} // namespace odps
} // namespace apsara

#endif // PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_EXTRACTOR_H_
