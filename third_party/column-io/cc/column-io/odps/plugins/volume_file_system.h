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

#ifndef PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_VOLUME_FILE_SYSTEM_H_
#define PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_VOLUME_FILE_SYSTEM_H_

#include <memory>
#include <string>
#include <vector>

#include "algo/include/data_io/file_reader_base.h"
#include "algo/include/data_io/volume_file_system_v2.h"
#include "column-io/odps/plugins/extractor.h"

namespace apsara {
namespace odps {
namespace algo {
namespace tf {
namespace plugins {
namespace odps {

class VolumeFileSystem {
public:
  VolumeFileSystem();

  VolumeFileSystem(const VolumeFileSystem &fs) = delete; // UnCopable

  ~VolumeFileSystem();

  bool Init(const std::string &conf, const std::string &capability);
  bool IsFile(const std::string &path, bool *ret);
  bool IsDirectory(const std::string &path, bool *ret);
  bool GetFileSize(const std::string &path, size_t *ret);
  bool List(const std::string &path, std::vector<std::string> *ret);
  bool CreateFileReader(const std::string &path,
                        apsara::odps::algo::FileReaderBase **ret);

private:
  std::unique_ptr<apsara::odps::algo::VolumeFileSystemV2> fs_;
};

} // namespace odps
} // namespace plugins
} // namespace tf
} // namespace algo
} // namespace odps
} // namespace apsara

#endif // PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_VOLUME_FILE_SYSTEM_H_
