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

#include "column-io/odps/proxy/lib_odps.h"
#include "absl/log/log.h"

#include <functional>
#include <unordered_set>

#include "column-io/odps/proxy/utils.h"

namespace column {
namespace odps {
namespace proxy {

const char *kLibOdpsDso =
    "/usr/lib/python2.7/site-packages/paiio/lib/libodps_plugin.so";

template <typename R, typename... Args>
Status BindFunc(void *handle, const char *name,
                std::function<R(Args...)> *func) {
  void *symbol_ptr = nullptr;
  auto s = GetSymbolFromLibrary(handle, name, &symbol_ptr);
  if (!s.ok()) {
    return s;
  }

  *func = reinterpret_cast<R (*)(Args...)>(symbol_ptr);
  return Status();
}

void LibOdps::LoadAndBind() {
  auto TryLoadAndBind = [this](const char *name, void **handle) -> Status {
    auto s = LoadLibrary(name, handle);
    if (!s.ok()) {
      return s;
    }

#define BIND_ODPS_FUNC(function)                                               \
  {                                                                            \
    auto s = BindFunc(*handle, #function, &function);                          \
    if (!s.ok()) {                                                             \
      return s;                                                                \
    }                                                                          \
  }

    BIND_ODPS_FUNC(TableOpen);
    BIND_ODPS_FUNC(TableGetRowCount);
    BIND_ODPS_FUNC(TableSeek);
    BIND_ODPS_FUNC(TableRead);
    BIND_ODPS_FUNC(TableReadBatch);
    BIND_ODPS_FUNC(GetTableSchema);
    BIND_ODPS_FUNC(TableGetReadBytes);
    BIND_ODPS_FUNC(TableClose);
    BIND_ODPS_FUNC(VolumeFileSystemOpen);
    BIND_ODPS_FUNC(VolumeFileSystemIsFile);
    BIND_ODPS_FUNC(VolumeFileSystemIsDirectory);
    BIND_ODPS_FUNC(VolumeFileSystemGetSize);
    BIND_ODPS_FUNC(VolumeFileSystemList);
    BIND_ODPS_FUNC(VolumeFileSystemClose);
    BIND_ODPS_FUNC(VolumeOpen);
    BIND_ODPS_FUNC(VolumeRead);
    BIND_ODPS_FUNC(VolumeSeek);
    BIND_ODPS_FUNC(VolumeClose);

#undef BIND_ODPS_FUNC
    return Status();
  };

  const char *lib_odps_path = getenv("LIB_ODPS_PLUGIN");
  if (lib_odps_path == nullptr) {
    lib_odps_path = kLibOdpsDso;
  }

  status_ = TryLoadAndBind(lib_odps_path, &handle_);
  if (status_.ok()) {
    LOG(INFO) << "Load odps so successfully, path: " << lib_odps_path;
  } else {
    LOG(ERROR) << "Load odps so failed, path: " << lib_odps_path;
  }
}

} // namespace proxy
} // namespace odps
} // namespace column

#endif // TF_ENABLE_ODPS_COLUMN
