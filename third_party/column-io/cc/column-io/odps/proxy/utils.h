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

#ifndef PAIIO_CC_IO_ALGO_COLUMN_ODPS_UTILS_H_
#define PAIIO_CC_IO_ALGO_COLUMN_ODPS_UTILS_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "column-io/framework/status.h"

namespace column {
namespace odps {
namespace proxy {
struct IoConfig {
  std::string capability;
  std::unordered_map<std::string, std::string> confs;
};

bool CheckOdpsIoConfigFile();

std::string GetOdpsIoConfigFile();

IoConfig &GetOdpsIoConfig();

bool GetConfigByName(const std::string &name, std::string *config);

Status LoadLibrary(const char *library_filename, void **handle);

Status GetSymbolFromLibrary(void *handle, const char *symbol_name,
                            void **symbol);
std::string RemovePrefix(const std::string &name, std::string prefix = "");

std::string GetObjectIdentifier(const std::string &name);

inline std::string JoinString(const std::vector<std::string> &parts,
                              const std::string &separator) {
  std::string result;
  int delim_length = separator.size();

  int length = 0;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      length += delim_length;
    }
    length += parts[i].size();
  }

  result.reserve(length);

  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      result.append(separator);
    }
    result.append(parts[i]);
  }

  return result;
}

std::string B64Decode(const std::string &in);

} // namespace proxy
} // namespace odps
} // namespace column

#endif // PAIIO_CC_IO_ALGO_COLUMN_ODPS_UTILS_H_

#endif // TF_ENABLE_ODPS_COLUMN
