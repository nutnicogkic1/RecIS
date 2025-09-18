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

#ifndef PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_VOLUME_READER_H_
#define PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_VOLUME_READER_H_

#include <memory>
#include <string>
#include <vector>

#include "algo/include/data_io/file_reader_base.h"
#include "column-io/odps/plugins/extractor.h"

namespace apsara {
namespace odps {
namespace algo {
namespace tf {
namespace plugins {
namespace odps {

class VolumeReader {
public:
  VolumeReader(apsara::odps::algo::FileReaderBase *reader) : reader_(reader){};

  VolumeReader(const VolumeReader &reader) = delete; // UnCopable

  ~VolumeReader();

  // Seek to specified offset, user should guarantee the offset is valid
  //
  // Arguments:
  //   offset: a valid offset
  bool Seek(size_t offset);

  // Read buffer
  //
  // Arguments:
  //   buf: buffer to store result
  //   len: length of buffer
  //
  // Return:
  //   length of read bytes
  bool Read(char *buf, size_t len, ssize_t *read_len);

private:
  std::unique_ptr<apsara::odps::algo::FileReaderBase> reader_;
};

} // namespace odps
} // namespace plugins
} // namespace tf
} // namespace algo
} // namespace odps
} // namespace apsara

#endif // PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_VOLUME_READER_H_
