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

#include <unordered_set>

#include "algo/include/common/string_util.h"

#include "column-io/odps/plugins/exception_to_ret.h"
#include "column-io/odps/plugins/logging.h"
#include "column-io/odps/plugins/volume_reader.h"

namespace apsara {
namespace odps {
namespace algo {
namespace tf {
namespace plugins {
namespace odps {

VolumeReader::~VolumeReader() {}

bool VolumeReader::Seek(size_t offset) {
  bool flag = true;
  try {
    reader_->Seek(offset);
  }
  EXCEPTION_TO_RET(flag)

  return flag;
}

bool VolumeReader::Read(char *buf, size_t len, ssize_t *read_len) {
  bool flag = true;
  try {
    (*read_len) = reader_->Read(buf, len);
  }
  EXCEPTION_TO_RET(flag)

  return flag;
}

} // namespace odps
} // namespace plugins
} // namespace tf
} // namespace algo
} // namespace odps
} // namespace apsara
