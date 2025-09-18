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

#ifndef PAIIO_CC_IO_ALGO_COLUMN_ODPS_LIB_ODPS_H_
#define PAIIO_CC_IO_ALGO_COLUMN_ODPS_LIB_ODPS_H_

#include <functional>
#include <string>
#include <vector>

#include "column-io/framework/status.h"
#include "column-io/odps/plugins/c_api.h"

namespace column {
namespace odps {
namespace proxy {
class LibOdps {
public:
  static LibOdps *Load() {
    static LibOdps *lib = []() -> LibOdps * {
      LibOdps *lib = new LibOdps;
      lib->LoadAndBind();
      return lib;
    }();

    return lib;
  }
  static void LoadWrap() {
    [[maybe_unused]] bool succeed;
    succeed = Load() == nullptr;
  }

  Status status() { return status_; }

  std::function<CAPI_TableReader *(const char *, int, const char *, int,
                                   const char *, int)>
      TableOpen;
  std::function<CAPI_Offset(CAPI_TableReader *)> TableGetRowCount;
  std::function<void(CAPI_TableReader *, CAPI_Offset)> TableSeek;
  std::function<int(CAPI_TableReader *, void *)> TableRead;
  std::function<void(CAPI_TableReader *, int, void *)> TableReadBatch;
  std::function<void(CAPI_TableReader *, void *)> GetTableSchema;
  std::function<size_t(CAPI_TableReader *)> TableGetReadBytes;
  std::function<void(CAPI_TableReader *)> TableClose;

  std::function<CAPI_VolumeFileSystem *(const char *, int, const char *, int)>
      VolumeFileSystemOpen;
  std::function<bool(CAPI_VolumeFileSystem *, const char *, bool *)>
      VolumeFileSystemIsFile;
  std::function<bool(CAPI_VolumeFileSystem *, const char *, bool *)>
      VolumeFileSystemIsDirectory;
  std::function<bool(CAPI_VolumeFileSystem *, const char *, size_t *)>
      VolumeFileSystemGetSize;
  std::function<bool(CAPI_VolumeFileSystem *, const char *, void *)>
      VolumeFileSystemList;
  std::function<void(CAPI_VolumeFileSystem *)> VolumeFileSystemClose;
  std::function<CAPI_VolumeReader *(CAPI_VolumeFileSystem *, const char *)>
      VolumeOpen;
  std::function<bool(CAPI_VolumeReader *, char *, size_t, ssize_t *)>
      VolumeRead;
  std::function<bool(CAPI_VolumeReader *, CAPI_Offset)> VolumeSeek;
  std::function<void(CAPI_VolumeReader *)> VolumeClose;

private:
  void LoadAndBind();

  void *handle_;
  Status status_;
};

} // namespace proxy
} // namespace odps
} // namespace column

#endif // PAIIO_CC_IO_ALGO_COLUMN_ODPS_LIB_ODPS_H_

#endif // TF_ENABLE_ODPS_COLUMN
