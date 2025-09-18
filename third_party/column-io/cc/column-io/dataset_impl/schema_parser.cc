#include "column-io/dataset_impl/schema_parser.h"
namespace column {
namespace dataset {
std::pair<
    std::vector<std::string>,
    std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>>
SchemaParser::ParseSchema(
    const std::vector<std::string> &paths,
    bool is_compressed,
    const std::unordered_set<std::string> &selected_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int32_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<Tensor> &dense_defaults) {

  std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>
      output_schema;
  std::vector<std::string> input_columns;
  std::string row_mode = std::getenv("ODPS_DATASET_ROW_MODE") ? std::getenv("ODPS_DATASET_ROW_MODE") : "0";
  bool row_mode_with_null = (row_mode == "1") ;
  for (size_t i = 0; i < paths.size(); ++i) {
    std::shared_ptr<arrow::RecordBatch> data;
    auto st = rb_reader_(paths[i], selected_columns, dense_columns,
                         is_compressed, &data);
    CHECK(st.ok()) << st.error_message();
    std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>
        schema;
    std::vector<std::string> one_input_columns;
    st = ParseSchemaCommon(data, selected_columns, hash_features, hash_types, hash_buckets,
                           dense_columns, dense_defaults, is_compressed, false, &schema,
                           &one_input_columns, row_mode_with_null);
    CHECK(st.ok()) << st.error_message();
    if (!output_schema.empty()) {
      if (output_schema != schema) {
        CHECK(false) << "schema from path: " << paths[i]
                     << ", not the same as before";
      }
      if (input_columns != one_input_columns) {
        CHECK(false) << "input column from path: " << paths[i]
                     << ", not the same as before";
      }
    } else {
      output_schema = std::move(schema);
      input_columns = std::move(one_input_columns);
    }
  }
  return std::make_pair(input_columns, output_schema);
}


std::pair<
    std::vector<std::string>,
    std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>>
SchemaParser::ParseSchemaByRows(
    const std::vector<std::string> &paths,
    bool is_compressed,
    const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int32_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<Tensor> &dense_defaults) {
    
    std::vector<std::string> input_columns;
    std::vector<std::map<std::string, std::vector<std::vector<std::string>>>> output_schema;
    
    //std::string row_mode = std::getenv("ODPS_DATASET_ROW_MODE") ? std::getenv("ODPS_DATASET_ROW_MODE") : "0";
    //bool row_mode_with_null = (row_mode == "1") ;
    bool row_mode_with_null = true;
    std::shared_ptr<arrow::RecordBatch> data;
    const std::unordered_set<std::string> selected_columns_set(selected_columns.begin(), selected_columns.end());
    for (size_t i = 0; i < paths.size(); ++i) {
        data.reset();
        auto st = rb_reader_(paths[i], selected_columns_set, dense_columns, is_compressed, &data);
        CHECK(st.ok()) << st.error_message();
        std::vector<std::string> batch_columns;
        std::vector<std::map<std::string, std::vector<std::vector<std::string>>>> batch_schema;
        st = ParseSchemaCommon(data, selected_columns_set, hash_features, hash_types, hash_buckets,
                               dense_columns, dense_defaults, is_compressed, false, &batch_schema,
                               &batch_columns, row_mode_with_null);
        CHECK(st.ok()) << st.error_message();
        if (!output_schema.empty()) {
        if (output_schema != batch_schema) {
            CHECK(false) << "batch_schema from path: " << paths[i] << ", not the same as before";
        }
        if (input_columns != batch_columns) {
            CHECK(false) << "input_column from path: " << paths[i] << ", not the same as before";
        }
        } else {
        output_schema = std::move(batch_schema);
        input_columns = std::move(batch_columns);
        }
    }
    // Conver input_columns to original order
    // 目标: 以指定的selected_columns(P1)或数据源data->schema()中的列顺序(P2)对input_columns(P3)进行重排列
    std::unordered_map<std::string, size_t> orderMap;
    size_t order = 0;
    if( selected_columns.empty() ){ // sort input_columns by data->schema()->fields()
        for (const auto& field : data->schema()->fields())  orderMap[field->name()] = order++;
    }else{ // sort input_columns by selected_columns
        for (const auto& col_name : selected_columns)   orderMap[col_name] = order++;
    }
    data.reset();
    std::sort(input_columns.begin(), input_columns.end(), 
            [&orderMap](const std::string& a, const std::string& b) {
                return orderMap.at(a) < orderMap.at(b);
            }
    );
    return std::make_pair(input_columns, output_schema);
}


Status SchemaParser::ParseSchemaCommon(
    std::shared_ptr<arrow::RecordBatch> &data,
    const std::unordered_set<std::string> &selected_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int32_t> &hash_buckets,
    const std::vector<std::string> &dense_features,
    const std::vector<Tensor> &dense_defaults,
    bool is_compressed,
    bool is_large_list,
    std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>
        *output_schema,
    std::vector<std::string> *input_columns,
    bool with_null) {
  // parse schema
  auto formater =
      ColumnDataFormater::GetColumnDataFormater(is_compressed, is_large_list, with_null);
  auto st = formater->InitSchema(data->schema(), hash_features, hash_types, hash_buckets,
                                 dense_features, dense_defaults, selected_columns);
  if (!st.ok()) {
    return Status::InvalidArgument("fail to init formater, error info: ",
                                   st.error_message());
  }
  std::vector<std::shared_ptr<arrow::RecordBatch>> formated_data;
  st = formater->FormatSample(data, &formated_data);
  if (!st.ok()) {
    return Status::Internal("fail to format sample, error info: ",
                            st.error_message());
  }
  std::vector<std::map<std::string, std::vector<std::vector<Tensor>>>>
      out_tensors;
  st = formater->Convert(formated_data, &out_tensors);
  if (!st.ok()) {
    return Status::Internal("fail to convert sample, error info: ",
                            st.error_message());
  }
  // assemble tensor infos
  std::stringstream tensor_info;
  for (size_t i = 0; i < out_tensors.size(); ++i) {
    auto &map = out_tensors[i];
    output_schema->resize(i + 1);
    auto &out_map = (*output_schema)[i];
    for (auto &item : map) {
      auto &feature_tensors = out_map[item.first];
      for (size_t j = 0; j < item.second.size(); ++j) {
        feature_tensors.resize(j + 1);
        auto &one_vec = feature_tensors[j];
        for (auto &tensor : item.second[j]) {
          one_vec.emplace_back("Placeholder");
        }
      }
    }
  }
  formater->GetInputColumns(input_columns);
  return Status::OK();
}

/* Make: 创建一个解析列式结构的结构解析器
 * @fn: 读取一批列式结构的纯读函数
 * @return: 输出列式结构的解析器SchemaParser
*/ 
std::unique_ptr<SchemaParser> SchemaParser::Make(RecordBatchReaderFn fn) {
  return std::unique_ptr<SchemaParser>(new SchemaParser(fn));
}
/* MakeByRows: 创建一个解析输式结构的结构解析器
 * @fn: 读取一批列式结构的纯读函数 *注意* RecordBatch始终为列结构, 因此fn不存在天然的行结构输出
 * @return: 输出列式结构的解析器SchemaParser
*/ 
std::unique_ptr<SchemaParser> SchemaParser::MakeByRows(RecordBatchReaderFn fn) {
  return std::unique_ptr<SchemaParser>(new SchemaParser(fn));
}
} // namespace dataset
} // namespace column
