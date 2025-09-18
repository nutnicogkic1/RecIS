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

#include <string>

#include "algo/include/common/data_limits.h"

#include "column-io/odps/plugins/logging.h"
#include "column-io/odps/plugins/record_wrapper.h"

namespace apsara {
namespace odps {
namespace algo {
namespace tf {
namespace plugins {
namespace odps {

RecordWrapper::RecordWrapper(apsara::odps::algo::TableRecord *record)
    : record_(record) {}

const apsara::odps::algo::TableSchema &RecordWrapper::GetSchema() {
  return record_->GetSchema();
}

void *RecordWrapper::Get(int index, int32_t *len) {
  return record_->Get(index, len);
}

void RecordWrapper::GetDoubleValue(int index, double **ret) {
  if (apsara::odps::algo::kDouble != GetSchema().Type(index)) {
    LOG(FATAL) << "Column " << index << " is not with type double!";
  }

  int32_t len;
  double *value = static_cast<double *>(record_->Get(index, &len));
  if (value != nullptr) {
    *ret = apsara::odps::algo::DataLimits<double>::IsNull(*value) ? nullptr
                                                                  : value;
  } else {
    *ret = nullptr;
  }
}

void RecordWrapper::GetIntegerValue(int index, int64_t **ret) {
  if (apsara::odps::algo::kInt != GetSchema().Type(index)) {
    LOG(FATAL) << "Column " << index << " is not with type int!";
  }

  int32_t len;
  int64_t *value = static_cast<int64_t *>(record_->Get(index, &len));
  if (value != nullptr) {
    *ret = apsara::odps::algo::DataLimits<int64_t>::IsNull(*value) ? nullptr
                                                                   : value;
  } else {
    *ret = nullptr;
  }
}

void RecordWrapper::GetTimeValue(int index, int64_t **ret) {
  if (apsara::odps::algo::kDateTime != GetSchema().Type(index)) {
    LOG(FATAL) << "Column " << index << " is not with type datetime!";
  }

  int32_t len;
  int64_t *value = static_cast<int64_t *>(record_->Get(index, &len));
  if (value != nullptr) {
    *ret = apsara::odps::algo::DataLimits<int64_t>::IsNull(*value) ? nullptr
                                                                   : value;
  } else {
    *ret = nullptr;
  }
}

void RecordWrapper::GetBoolValue(int index, bool **ret) {
  if (apsara::odps::algo::kBool != GetSchema().Type(index)) {
    LOG(FATAL) << "Column " << index << " is not with type bool!";
  }

  int32_t len;
  bool *value = static_cast<bool *>(record_->Get(index, &len));
  if (value != nullptr) {
    *ret = apsara::odps::algo::DataLimits<bool>::IsNull(
               *reinterpret_cast<uint8_t *>(value))
               ? nullptr
               : value;
  } else {
    *ret = nullptr;
  }
}

void RecordWrapper::GetStringValue(int index, char **ret, int32_t *len) {
  if (apsara::odps::algo::kString != GetSchema().Type(index)) {
    LOG(FATAL) << "Column " << index << " is not with type string!";
  }

  char *value = static_cast<char *>(record_->Get(index, len));
  if (value != nullptr) {
    *ret = apsara::odps::algo::DataLimits<std::string>::IsNull(*len) ? nullptr
                                                                     : value;
  } else {
    *ret = nullptr;
  }
}

} // namespace odps
} // namespace plugins
} // namespace tf
} // namespace algo
} // namespace odps
} // namespace apsara
