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

#include "column-io/odps/wrapper/odps_table_column_reader.h"

#include <algorithm>
#include <memory>

#include "absl/log/log.h"
#include "arrow/array.h"
#include "arrow/record_batch.h"
#include "arrow/type.h"

namespace column {
namespace odps {
namespace wrapper {
namespace {

#define CHECK_NULL_1(ARR1)                                                     \
  if (ARR1->null_count() != 0) {                                               \
    LOG(ERROR) << "Not support column with null elements inside";              \
    return false;                                                              \
  }

#define CHECK_NULL_2(ARR1, ARR2)                                               \
  if (ARR1->null_count() != 0 || ARR2->null_count() != 0) {                    \
    LOG(ERROR) << "Not support column with null elements inside";              \
    return false;                                                              \
  }

#define CHECK_NULL_3(ARR1, ARR2, ARR3)                                         \
  if (ARR1->null_count() != 0 || ARR2->null_count() != 0 ||                    \
      ARR3->null_count() != 0) {                                               \
    LOG(ERROR) << "Not support column with null elements inside";              \
    return false;                                                              \
  }

#define CHECK_NULL_4(ARR1, ARR2, ARR3, ARR4)                                   \
  if (ARR1->null_count() != 0 || ARR2->null_count() != 0 ||                    \
      ARR3->null_count() != 0 || ARR4->null_count() != 0) {                    \
    LOG(ERROR) << "Not support column with null elements inside";              \
    return false;                                                              \
  }

const char *kStructKey = "k";
const char *kStructValue = "v";

arrow::Type::type
DetectValueType(const std::shared_ptr<arrow::DataType> &type) {
  switch (type->id()) {
  case arrow::Type::UINT8:
  case arrow::Type::INT8:
  case arrow::Type::UINT16:
  case arrow::Type::INT16:
  case arrow::Type::UINT32:
  case arrow::Type::INT32:
  case arrow::Type::UINT64:
  case arrow::Type::INT64:
  case arrow::Type::FLOAT:
  case arrow::Type::DOUBLE:
  case arrow::Type::STRING:
    return type->id();
  case arrow::Type::MAP:
    return DetectValueType(
        std::static_pointer_cast<arrow::MapType>(type)->item_type());
  case arrow::Type::LIST:
    return DetectValueType(
        std::static_pointer_cast<arrow::ListType>(type)->value_type());
  case arrow::Type::FIXED_SIZE_LIST:
    return DetectValueType(
        std::static_pointer_cast<arrow::FixedSizeListType>(type)->value_type());
  case arrow::Type::LARGE_LIST:
    return DetectValueType(
        std::static_pointer_cast<arrow::LargeListType>(type)->value_type());
  case arrow::Type::STRUCT:
    return DetectValueType(std::static_pointer_cast<arrow::StructType>(type)
                               ->GetFieldByName(kStructValue)
                               ->type());
  default:
    return arrow::Type::NA;
  }
}

template <typename ListType, arrow::Type::type list_type_id>
FeatureType
DetectFeatureType(bool compressed,
                  const std::shared_ptr<arrow::DataType> &array_type) {
  auto real_type = array_type;
  if (compressed) {
    if (real_type->id() == list_type_id) {
      real_type = std::static_pointer_cast<ListType>(real_type)->value_type();
    } else {
      // Error
    }
  }

  if (real_type->id() == arrow::Type::STRING) {
    return kSparse;
  }

  if (real_type->id() == list_type_id) {
    real_type = std::static_pointer_cast<ListType>(real_type)->value_type();
  }

  if (real_type->id() == arrow::Type::INT64) {
    return kSparse;
  }

  if (real_type->id() == arrow::Type::FLOAT) {
    return kSparse;
  }

  if (real_type->id() == arrow::Type::STRING) {
    return kSeqSparse;
  }

  if (real_type->id() == arrow::Type::STRUCT) {
    auto st = std::static_pointer_cast<arrow::StructType>(real_type);
    auto kt = st->GetFieldByName(kStructKey)->type();
    auto vt = st->GetFieldByName(kStructValue)->type();
    if (kt->id() == arrow::Type::INT64 &&
        (vt->id() == arrow::Type::FLOAT || vt->id() == arrow::Type::DOUBLE)) {
      return kWeightedSparse;
    }
  }

  if (real_type->id() == list_type_id) {
    real_type = std::static_pointer_cast<ListType>(real_type)->value_type();
    if (real_type->id() == arrow::Type::INT64) {
      return kSeqSparse;
    }

    if (real_type->id() == arrow::Type::STRUCT) {
      auto st = std::static_pointer_cast<arrow::StructType>(real_type);
      auto kt = st->GetFieldByName(kStructKey)->type();
      auto vt = st->GetFieldByName(kStructValue)->type();
      if (kt->id() == arrow::Type::INT64 &&
          (vt->id() == arrow::Type::FLOAT || vt->id() == arrow::Type::DOUBLE)) {
        return kSeqWeightedSparse;
      }
    }
  }

  return kUnknow;
}

///////////////////// ReReadableSGroupReader  /////////////////////

class BaseReader : public OdpsTableColumnReader {
public:
  BaseReader() : begin_(0), end_(0), index_(begin_) {}

  void Init(int begin, int end) {
    begin_ = begin;
    end_ = end;
    index_ = begin_;
  }

  bool Reset() override {
    index_ = begin_;
    return true;
  }

protected:
  bool Overflow() { return index_ >= end_; }

protected:
  int begin_;
  int end_;
  int index_;
};

///////////////////// FlatColumnReader /////////////////////

template <typename LIST_TYPE> class FlatColumnReader : public BaseReader {
public:
  FlatColumnReader(const std::shared_ptr<arrow::Array> &data,
                   FeatureType feature_type);

  bool ReadVec(const char **str, size_t *length) override;

  bool ReadVec(const int64_t **data, size_t *length) override;

  bool ReadVec(const float **data, size_t *length) override;

  bool ReadVec(const double **data, size_t *length) override;

  bool ReadMap(const int64_t **keys, const float **values,
               size_t *length) override;

  bool ReadMap(const int64_t **keys, const double **values,
               size_t *length) override;

  bool ReadMatrix(const char **data, size_t *length,
                  std::vector<size_t> *segments);
  bool ReadMatrix(const int64_t **data, size_t *length,
                  std::vector<size_t> *segments) override;

  bool HasIndicator() const override { return false; }

  std::string indicator_name() const override { return ""; }

  bool GetIndicator(const int64_t **data, size_t *length) override;

  arrow::Type::type value_type() const override { return value_type_; }

  FeatureType feature_type() const override { return feature_type_; }

private:
  template <typename ArrayType, typename CType>
  bool ReadVecInternal(const CType **data, size_t *length) {
    if (Overflow()) {
      return false;
    }

    auto l1 = std::static_pointer_cast<LIST_TYPE>(array_);
    auto l2 = std::static_pointer_cast<ArrayType>(l1->values());
    CHECK_NULL_2(l1, l2)
    *data = l2->data()->template GetValues<CType>(1) + l1->value_offset(index_);
    *length = l1->value_length(index_);
    ++index_;

    return true;
  }

  template <typename ArrayType, typename CType>
  bool ReadMapInternal(const int64_t **keys, const CType **values,
                       size_t *length) {
    if (Overflow()) {
      return false;
    }

    auto list_arr = std::static_pointer_cast<LIST_TYPE>(array_);
    auto struct_arr =
        std::static_pointer_cast<arrow::StructArray>(list_arr->values());
    auto key_arr = std::static_pointer_cast<arrow::Int64Array>(
        struct_arr->GetFieldByName(kStructKey));
    auto val_arr = std::static_pointer_cast<ArrayType>(
        struct_arr->GetFieldByName(kStructValue));
    CHECK_NULL_4(list_arr, struct_arr, key_arr, val_arr)

    int offset = list_arr->value_offset(index_);
    *keys = key_arr->data()->template GetValues<int64_t>(1) + offset;
    *values = val_arr->data()->template GetValues<CType>(1) + offset;
    *length = list_arr->value_length(index_);
    ++index_;

    return true;
  }

private:
  std::shared_ptr<arrow::Array> array_;
  FeatureType feature_type_;
  arrow::Type::type value_type_;
};

template <typename LIST_TYPE>
FlatColumnReader<LIST_TYPE>::FlatColumnReader(
    const std::shared_ptr<arrow::Array> &array, FeatureType feature_type)
    : array_(array), feature_type_(feature_type) {
  Init(0, array_->length());
  value_type_ = DetectValueType(array_->type());
}

// StringArray
template <typename LIST_TYPE>
bool FlatColumnReader<LIST_TYPE>::ReadVec(const char **str, size_t *length) {
  if (Overflow()) {
    return false;
  }
  CHECK_NULL_1(array_)

  auto string_array = std::static_pointer_cast<arrow::StringArray>(array_);
  auto view = string_array->GetView(index_);
  *str = view.data();
  *length = view.size();
  ++index_;

  return true;
}

// List<Int64Array>
template <typename LIST_TYPE>
bool FlatColumnReader<LIST_TYPE>::ReadVec(const int64_t **data,
                                          size_t *length) {
  return ReadVecInternal<arrow::Int64Array, int64_t>(data, length);
}

// List<FloatArray>
template <typename LIST_TYPE>
bool FlatColumnReader<LIST_TYPE>::ReadVec(const float **data, size_t *length) {
  return ReadVecInternal<arrow::FloatArray, float>(data, length);
}

// List<DoubleArray>
template <typename LIST_TYPE>
bool FlatColumnReader<LIST_TYPE>::ReadVec(const double **data, size_t *length) {
  return ReadVecInternal<arrow::DoubleArray, double>(data, length);
}

// List<Struct<k:int64, v:float>>
template <typename LIST_TYPE>
bool FlatColumnReader<LIST_TYPE>::ReadMap(const int64_t **keys,
                                          const float **values,
                                          size_t *length) {
  return ReadMapInternal<arrow::FloatArray, float>(keys, values, length);
}

// List<Struct<k:int64, v:double>>
template <typename LIST_TYPE>
bool FlatColumnReader<LIST_TYPE>::ReadMap(const int64_t **keys,
                                          const double **values,
                                          size_t *length) {
  return ReadMapInternal<arrow::DoubleArray, double>(keys, values, length);
}

template <typename LIST_TYPE>
bool FlatColumnReader<LIST_TYPE>::ReadMatrix(const char **data, size_t *length,
                                             std::vector<size_t> *segments) {
  if (Overflow()) {
    return false;
  }

  segments->clear();
  auto l1 = std::static_pointer_cast<LIST_TYPE>(array_);
  auto string_array =
      std::static_pointer_cast<arrow::StringArray>(l1->values());
  CHECK_NULL_2(l1, string_array)

  auto offset = l1->value_offset(index_);
  segments->resize(l1->value_length(index_));

  auto view = string_array->GetView(offset);
  *data = view.data();

  *length = 0;
  for (size_t i = 0; i < segments->size(); ++i) {
    (*segments)[i] = string_array->value_length(offset + i);
    *length += (*segments)[i];
  }
  ++index_;

  return true;
}

template <typename LIST_TYPE>
bool FlatColumnReader<LIST_TYPE>::ReadMatrix(const int64_t **data,
                                             size_t *length,
                                             std::vector<size_t> *segments) {
  if (Overflow()) {
    return false;
  }

  segments->clear();

  auto l1 = std::static_pointer_cast<LIST_TYPE>(array_);
  auto l2 = std::static_pointer_cast<LIST_TYPE>(l1->values());
  auto l3 = std::static_pointer_cast<arrow::Int64Array>(l2->values());

  CHECK_NULL_3(l1, l2, l3)

  auto offset = l1->value_offset(index_);
  segments->resize(l1->value_length(index_));
  *data = l3->data()->template GetValues<int64_t>(1) + l2->value_offset(offset);
  *length = 0;
  for (size_t i = 0; i < segments->size(); ++i) {
    (*segments)[i] = l2->value_length(offset + i);
    *length += (*segments)[i];
  }
  ++index_;

  return true;
}

template <typename LIST_TYPE>
bool FlatColumnReader<LIST_TYPE>::GetIndicator(const int64_t **data,
                                               size_t *length) {
  (void)data;
  (void)length;
  return false;
}

///////////////////// ListColumnReader /////////////////////

template <typename LIST_TYPE> class ListColumnReader : public BaseReader {
public:
  ListColumnReader(const std::shared_ptr<LIST_TYPE> &array,
                   const std::shared_ptr<LIST_TYPE> &indicator_array,
                   const std::string &indicator_name, FeatureType feature_type);

  bool ReadVec(const char **str, size_t *length) override;

  bool ReadVec(const int64_t **data, size_t *length) override;

  bool ReadVec(const float **data, size_t *length) override;

  bool ReadVec(const double **data, size_t *length) override;

  bool ReadMap(const int64_t **keys, const float **values,
               size_t *length) override;

  bool ReadMap(const int64_t **keys, const double **values,
               size_t *length) override;

  bool ReadMatrix(const char **data, size_t *length,
                  std::vector<size_t> *segments) override;
  bool ReadMatrix(const int64_t **data, size_t *length,
                  std::vector<size_t> *segments) override;

  bool HasIndicator() const override { return !indicator_name_.empty(); }

  std::string indicator_name() const override { return indicator_name_; }

  bool GetIndicator(const int64_t **data, size_t *length) override;

  arrow::Type::type value_type() const override { return value_type_; }

  FeatureType feature_type() const override { return feature_type_; }

private:
  void ReadIndicators();

  template <typename ArrayType, typename CType>
  bool ReadVecInternal(const CType **data, size_t *length) {
    if (!HasIndicator()) {
      if (Overflow()) {
        return false;
      }
    } else {
      if (!indicator_read_) {
        ReadIndicators();
      }

      if (index_ >= end_) {
        return false;
      }
    }

    auto l2 = std::static_pointer_cast<LIST_TYPE>(array_->values());
    auto l3 = std::static_pointer_cast<ArrayType>(l2->values());
    CHECK_NULL_3(array_, l2, l3)
    *data = l3->data()->template GetValues<CType>(1) + l2->value_offset(index_);
    *length = l2->value_length(index_);
    ++index_;

    return true;
  }

  template <typename ArrayType, typename CType>
  bool ReadMapInternal(const int64_t **keys, const CType **values,
                       size_t *length) {
    if (!HasIndicator()) {
      if (Overflow()) {
        return false;
      }
    } else {
      if (!indicator_read_) {
        ReadIndicators();
      }

      if (index_ >= end_) {
        return false;
      }
    }

    auto l2 = std::static_pointer_cast<LIST_TYPE>(array_->values());
    int offset = l2->value_offset(index_);

    auto struct_arr =
        std::static_pointer_cast<arrow::StructArray>(l2->values());
    auto key_arr = std::static_pointer_cast<arrow::Int64Array>(
        struct_arr->GetFieldByName(kStructKey));
    auto val_arr = std::static_pointer_cast<ArrayType>(
        struct_arr->GetFieldByName(kStructValue));
    CHECK_NULL_1(array_)
    CHECK_NULL_4(l2, struct_arr, key_arr, val_arr)

    *keys = key_arr->data()->template GetValues<int64_t>(1) + offset;
    *values = val_arr->data()->template GetValues<CType>(1) + offset;

    *length = l2->value_length(index_);
    ++index_;

    return true;
  }

private:
  std::shared_ptr<LIST_TYPE> array_;
  std::shared_ptr<LIST_TYPE> indicator_array_;
  std::string indicator_name_;
  FeatureType feature_type_;
  arrow::Type::type value_type_;
  std::vector<int64_t> indicators_;
  std::vector<int> row_offsets_;
  bool indicator_read_;
};

template <typename LIST_TYPE>
ListColumnReader<LIST_TYPE>::ListColumnReader(
    const std::shared_ptr<LIST_TYPE> &array,
    const std::shared_ptr<LIST_TYPE> &indicator_array,
    const std::string &indicator_name, FeatureType feature_type)
    : array_(array), indicator_array_(indicator_array),
      indicator_name_(indicator_name), feature_type_(feature_type),
      indicator_read_(false) {
  if (!HasIndicator()) {
    int row_count = 0;
    for (int i = 0; i < array_->length(); ++i) {
      row_count += array_->value_length(i);
    }
    Init(0, row_count);
  } else {
    Init(0, array_->length());
  }
  value_type_ = DetectValueType(array->type());
}

// List<StringArray>
template <typename LIST_TYPE>
bool ListColumnReader<LIST_TYPE>::ReadVec(const char **str, size_t *length) {
  if (Overflow()) {
    return false;
  }

  auto l2 = std::static_pointer_cast<arrow::StringArray>(array_->values());
  CHECK_NULL_2(array_, l2)
  auto view = l2->GetView(index_);
  *str = view.data();
  *length = view.size();
  ++index_;

  return true;
}

// List<List<Int64Array>>
template <typename LIST_TYPE>
bool ListColumnReader<LIST_TYPE>::ReadVec(const int64_t **data,
                                          size_t *length) {
  return ReadVecInternal<arrow::Int64Array, int64_t>(data, length);
}

// List<List<FloatArray>>
template <typename LIST_TYPE>
bool ListColumnReader<LIST_TYPE>::ReadVec(const float **data, size_t *length) {
  return ReadVecInternal<arrow::FloatArray, float>(data, length);
}

// List<List<DoubleArray>>
template <typename LIST_TYPE>
bool ListColumnReader<LIST_TYPE>::ReadVec(const double **data, size_t *length) {
  return ReadVecInternal<arrow::DoubleArray, double>(data, length);
}

// List<List<Struct<k:int64, v:float>>>
template <typename LIST_TYPE>
bool ListColumnReader<LIST_TYPE>::ReadMap(const int64_t **keys,
                                          const float **values,
                                          size_t *length) {
  return ReadMapInternal<arrow::FloatArray, float>(keys, values, length);
}

// List<List<Struct<k:int64, v:float>>>
template <typename LIST_TYPE>
bool ListColumnReader<LIST_TYPE>::ReadMap(const int64_t **keys,
                                          const double **values,
                                          size_t *length) {
  return ReadMapInternal<arrow::DoubleArray, double>(keys, values, length);
}

// array<array<string>>
template <typename LIST_TYPE>
bool ListColumnReader<LIST_TYPE>::ReadMatrix(const char **data, size_t *length,
                                             std::vector<size_t> *segments) {
  if (!HasIndicator()) {
    if (Overflow()) {
      return false;
    }
  } else {
    if (!indicator_read_) {
      ReadIndicators();
    }

    if (index_ >= end_) {
      return false;
    }
  }

  segments->clear();

  auto l2 = std::static_pointer_cast<LIST_TYPE>(array_->values());
  auto l3 = std::static_pointer_cast<arrow::StringArray>(l2->values());
  CHECK_NULL_3(array_, l2, l3)

  auto offset = l2->value_offset(index_);
  segments->resize(l2->value_length(index_));

  auto view = l3->GetView(offset);
  *data = view.data();
  *length = 0;
  for (size_t i = 0; i < segments->size(); ++i) {
    (*segments)[i] = l3->value_length(offset + i);
    *length += (*segments)[i];
  }
  ++index_;

  return true;
}

// List<List<List<Int64Array>>>
template <typename LIST_TYPE>
bool ListColumnReader<LIST_TYPE>::ReadMatrix(const int64_t **data,
                                             size_t *length,
                                             std::vector<size_t> *segments) {
  if (!HasIndicator()) {
    if (Overflow()) {
      return false;
    }
  } else {
    if (!indicator_read_) {
      ReadIndicators();
    }

    if (index_ >= end_) {
      return false;
    }
  }

  segments->clear();

  auto l2 = std::static_pointer_cast<LIST_TYPE>(array_->values());
  auto l3 = std::static_pointer_cast<LIST_TYPE>(l2->values());
  auto l4 = std::static_pointer_cast<arrow::Int64Array>(l3->values());
  CHECK_NULL_4(array_, l2, l3, l4)

  auto offset = l2->value_offset(index_);
  segments->resize(l2->value_length(index_));
  *data = l4->data()->template GetValues<int64_t>(1) + l3->value_offset(offset);
  *length = 0;
  for (size_t i = 0; i < segments->size(); ++i) {
    (*segments)[i] = l3->value_length(offset + i);
    *length += (*segments)[i];
  }
  ++index_;

  return true;
}

template <typename LIST_TYPE>
bool ListColumnReader<LIST_TYPE>::GetIndicator(const int64_t **data,
                                               size_t *length) {
  if (!HasIndicator()) {
    return false;
  }

  *data = indicators_.data();
  *length = indicators_.size();

  return true;
}

template <typename LIST_TYPE>
void ListColumnReader<LIST_TYPE>::ReadIndicators() {
  auto data =
      std::static_pointer_cast<arrow::Int64Array>(indicator_array_->values())
          ->data()
          ->template GetValues<int64_t>(1);
  int64_t offset = 0;
  for (int i = 0; i < end_; ++i) {
    auto begin = static_cast<int>(indicator_array_->value_offset(i));
    auto end = static_cast<int>(indicator_array_->value_offset(i + 1));
    for (; begin < end; ++begin) {
      indicators_.emplace_back(offset + *(data + begin));
    }
    offset += array_->value_length(i);
  }

  end_ = array_->value_offset(end_);
  Init(0, end_);

  indicator_read_ = true;
}

} // namespace

std::unique_ptr<OdpsTableColumnReader>
NewColumnReader(const std::shared_ptr<arrow::RecordBatch> &record_batch,
                const OdpsTableSchema &schema, const std::string &column,
                bool compressed, bool is_large_list) {
  std::unique_ptr<OdpsTableColumnReader> ret;

  const auto &alias_map = schema.alias_map();
  auto it = alias_map.find(column);
  if (it == alias_map.end()) {
    LOG(ERROR) << "No column found for " << column;
    return ret;
  }

  auto column_array = record_batch->GetColumnByName(it->second);
  if (column_array == nullptr) {
    LOG(ERROR) << "No column found in record batch " << column;
    return ret;
  }

  FeatureType data_type = kUnknow;
  if (is_large_list) {
    data_type =
        DetectFeatureType<arrow::LargeListType, arrow::Type::LARGE_LIST>(
            compressed, column_array->type());
  } else {
    data_type = DetectFeatureType<arrow::ListType, arrow::Type::LIST>(
        compressed, column_array->type());
  }

  if (!compressed) {
    if (is_large_list) {
      ret.reset(
          new FlatColumnReader<arrow::LargeListArray>(column_array, data_type));
    } else {
      ret.reset(
          new FlatColumnReader<arrow::ListArray>(column_array, data_type));
    }

    return ret;
  }

  std::string indicator = schema.GetIndicator(column);
  std::shared_ptr<arrow::Array> indicator_array = nullptr;
  if (!indicator.empty()) {
    indicator_array = record_batch->GetColumnByName(indicator);
  }

  if (is_large_list) {
    ret.reset(new ListColumnReader<arrow::LargeListArray>(
        std::static_pointer_cast<arrow::LargeListArray>(column_array),
        std::static_pointer_cast<arrow::LargeListArray>(indicator_array),
        indicator, data_type));
  } else {
    ret.reset(new ListColumnReader<arrow::ListArray>(
        std::static_pointer_cast<arrow::ListArray>(column_array),
        std::static_pointer_cast<arrow::ListArray>(indicator_array), indicator,
        data_type));
  }

  return ret;
}

} // namespace wrapper
} // namespace odps
} // namespace column

#endif // TF_ENABLE_ODPS_COLUMN
