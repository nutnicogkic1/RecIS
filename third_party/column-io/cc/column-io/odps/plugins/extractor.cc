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

#include "column-io/odps/plugins/extractor.h"
#include "column-io/odps/plugins/logging.h"
#include "column-io/odps/plugins/record_wrapper.h"

#include <string.h>
#include <time.h>

namespace apsara {
namespace odps {
namespace algo {
namespace tf {
namespace plugins {
namespace odps {

namespace {

class BoolExtractor : public Extractor {
public:
  int Extract(apsara::odps::algo::TableRecord *record, int index,
              std::string *value) override;
};

int BoolExtractor::Extract(apsara::odps::algo::TableRecord *record, int index,
                           std::string *value) {
  RecordWrapper wrapper(record);
  bool *ret = nullptr;
  wrapper.GetBoolValue(index, &ret);
  if (ret == nullptr) {
    LOG(ERROR) << "Invalid data extract, got nullptr, index: " << index;
    return -1;
  }

  if (*ret) {
    value->append("true");
  } else {
    value->append("false");
  }

  return 0;
}

class StringExtractor : public Extractor {
public:
  int Extract(apsara::odps::algo::TableRecord *record, int index,
              std::string *value) override;
};

int StringExtractor::Extract(apsara::odps::algo::TableRecord *record, int index,
                             std::string *value) {
  RecordWrapper wrapper(record);
  char *ret = nullptr;
  int32_t len = 0;
  wrapper.GetStringValue(index, &ret, &len);
  if (ret == nullptr) {
    LOG(ERROR) << "Invalid data extract, got nullptr, index: " << index;
    return -1;
  }

  value->append(ret, len);
  return 0;
}

class DateTimeExtractor : public Extractor {
public:
  int Extract(apsara::odps::algo::TableRecord *record, int index,
              std::string *value) override;
};

int DateTimeExtractor::Extract(apsara::odps::algo::TableRecord *record,
                               int index, std::string *value) {
  RecordWrapper wrapper(record);
  int64_t *ret = nullptr;
  wrapper.GetTimeValue(index, &ret);
  if (ret == nullptr) {
    LOG(ERROR) << "Invalid data extract, got nullptr, index: " << index;
    return -1;
  }

  time_t t = static_cast<time_t>(*ret);
  struct tm gmt;
  gmtime_r(&t, &gmt);

  char buf[64];
  memset(buf, 0, 64);
  strftime(buf, 64, "%a, %d %b %Y %H:%M:%S GMT", &gmt);

  value->append(buf);
  return 0;
}

class IntExtractor : public Extractor {
public:
  int Extract(apsara::odps::algo::TableRecord *record, int index,
              std::string *value) override;
};

int IntExtractor::Extract(apsara::odps::algo::TableRecord *record, int index,
                          std::string *value) {
  RecordWrapper wrapper(record);
  int64_t *ret = nullptr;
  wrapper.GetIntegerValue(index, &ret);
  if (ret == nullptr) {
    LOG(ERROR) << "Invalid data extract, got nullptr, index: " << index;
    return -1;
  }

  value->append(std::to_string(*ret));
  return 0;
}

class DoubleExtractor : public Extractor {
  int Extract(apsara::odps::algo::TableRecord *record, int index,
              std::string *value) override;
};

int DoubleExtractor::Extract(apsara::odps::algo::TableRecord *record, int index,
                             std::string *value) {
  RecordWrapper wrapper(record);
  double *ret = nullptr;
  wrapper.GetDoubleValue(index, &ret);
  if (ret == nullptr) {
    LOG(ERROR) << "Invalid data extract, got nullptr, index: " << index;
    return -1;
  }

  value->append(std::to_string(*ret));
  return 0;
}

BoolExtractor bool_extractor;
StringExtractor string_extractor;
DateTimeExtractor datetime_extractor;
IntExtractor int_extractor;
DoubleExtractor double_extractor;

} // namespace

Extractor *GetExtrator(apsara::odps::algo::ColumnType type) {
  switch (type) {
  case apsara::odps::algo::kInt:
    return &int_extractor;
  case apsara::odps::algo::kBool:
    return &bool_extractor;
  case apsara::odps::algo::kString:
    return &string_extractor;
  case apsara::odps::algo::kDateTime:
    return &datetime_extractor;
  case apsara::odps::algo::kDouble:
    return &double_extractor;
  default:
    return nullptr;
  }
}

} // namespace odps
} // namespace plugins
} // namespace tf
} // namespace algo
} // namespace odps
} // namespace apsara
