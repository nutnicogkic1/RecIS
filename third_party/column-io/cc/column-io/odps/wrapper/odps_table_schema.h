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

#ifndef PAIIO_CC_IO_ALGO_COLUMN_ODPS_TABLE_SCHEMA_H_
#define PAIIO_CC_IO_ALGO_COLUMN_ODPS_TABLE_SCHEMA_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "column-io/framework/status.h"
#include "column-io/odps/wrapper/odps_table_file_system.h"

namespace column {
namespace odps {
namespace wrapper {
extern const char *kNoIndicatorIdx;
extern const char *kIndicatorPre;

class OdpsTableSchema {
public:
  /*
  ** Sync table schema from odps tables
  *
  *Args:
  *  fs: odps table file system
  *  path: odps table paths
  *  compressed: whether the table is structure compressed
  *  allow_missing: whether alow specified column not exists in table
  *  selected_cols: specified selected table columns, containing the indicators
  *
  * Return:
  *   Status::OK() if all ok otherwise Error
  */
  Status Sync(OdpsTableFileSystem *fs, const std::vector<std::string> &paths,
              bool compressed, bool allow_missing,
              const std::vector<std::string> &selected_cols);

  const std::unordered_map<std::string, std::string> &alias_map() const {
    return alias_map_;
  }

  const std::unordered_map<std::string, std::string> &indicator_map() const {
    return indicator_map_;
  }

  const std::unordered_set<std::string> &indicators() const {
    return indicators_;
  }

  const std::vector<std::string> &select_columns() const {
    return select_columns_;
  }

  const std::unordered_map<std::string, std::string> &type_infos() const {
    return type_infos_;
  }

  std::string GetColumnName(const std::string &alias_name) const {
    // Find in alias map
    auto it = alias_map_.find(alias_name);
    if (it != alias_map_.end()) {
      return it->second;
    }

    // Not exists
    return "";
  }

  std::string GetIndicator(const std::string &column_name) const {
    auto it = indicator_map_.find(column_name);
    if (it != indicator_map_.end()) {
      return it->second;
    }

    // Not exists
    return "";
  }

  bool IsIndicator(const std::string &column_name) const {
    auto it = type_infos_.find(column_name);
    if (it != type_infos_.end()) {
      if (column_name.find(kIndicatorPre) == 0) {
        return true;
      }
    }

    return false;
  }

private:
  bool Reset(bool compressed,
             const std::unordered_set<std::string> &select_cols);

private:
  std::unordered_map<std::string, std::string> alias_map_;
  std::unordered_map<std::string, std::string> indicator_map_;
  std::unordered_set<std::string> indicators_;

  std::vector<std::string> select_columns_;
  std::unordered_map<std::string, std::string> type_infos_;
};

} // namespace wrapper
} // namespace odps
} // namespace column

#endif // PAIIO_CC_IO_ALGO_COLUMN_ODPS_TABLE_SCHEMA_H_

#endif // TF_ENABLE_ODPS_COLUMN
