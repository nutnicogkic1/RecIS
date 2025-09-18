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
#include <vector>

#include "algo/include/common/string_util.h"
#include "apsara/stone/encoding/base64/base64.h"

#include "column-io/odps/plugins/exception_to_ret.h"
#include "column-io/odps/plugins/logging.h"
#include "column-io/odps/plugins/volume_file_system.h"

namespace apsara {
namespace odps {
namespace algo {
namespace tf {
namespace plugins {
namespace odps {

VolumeFileSystem::VolumeFileSystem() {}

VolumeFileSystem::~VolumeFileSystem() {}

bool VolumeFileSystem::Init(const std::string &conf,
                            const std::string &capability) {
  std::string decodeConf;
  if (!apsara::stone::Base64Decode(conf, &decodeConf)) {
    LOG(ERROR) << "Invalid base64 conf" << conf;
    return false;
  }

  fs_.reset(new apsara::odps::algo::VolumeFileSystemV2(decodeConf, capability));
  return true;
}

bool VolumeFileSystem::IsFile(const std::string &path, bool *ret) {
  bool flag = true;
  try {
    *ret = fs_->IsFile(path);
  }
  EXCEPTION_TO_RET(flag)

  return flag;
}

bool VolumeFileSystem::IsDirectory(const std::string &path, bool *ret) {
  bool flag = true;
  try {
    *ret = fs_->IsDirectory(path);
  }
  EXCEPTION_TO_RET(flag)

  return flag;
}

bool VolumeFileSystem::GetFileSize(const std::string &path, size_t *ret) {
  bool flag = true;
  try {
    *ret = fs_->GetFileSize(path);
  }
  EXCEPTION_TO_RET(flag)

  return flag;
}

bool VolumeFileSystem::List(const std::string &path,
                            std::vector<std::string> *ret) {
  bool flag = true;
  try {
    *ret = fs_->List(path);
  }
  EXCEPTION_TO_RET(flag)

  return flag;
}

bool VolumeFileSystem::CreateFileReader(
    const std::string &path, apsara::odps::algo::FileReaderBase **ret) {
  bool flag = true;
  try {
    *ret = fs_->CreateFileReader(path);
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
