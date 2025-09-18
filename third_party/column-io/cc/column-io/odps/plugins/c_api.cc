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

#include <stdlib.h>
#include <string.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "column-io/odps/plugins/c_api.h"
#include "column-io/odps/plugins/logging.h"
#include "column-io/odps/plugins/table_reader.h"
#include "column-io/odps/plugins/volume_file_system.h"
#include "column-io/odps/plugins/volume_reader.h"

struct CAPI_TableReader {
  apsara::odps::algo::tf::plugins::odps::TableReader *reader;
};

struct CAPI_VolumeFileSystem {
  apsara::odps::algo::tf::plugins::odps::VolumeFileSystem *fs;
};

struct CAPI_VolumeReader {
  apsara::odps::algo::tf::plugins::odps::VolumeReader *reader;
};

extern "C" {

__attribute__((visibility("default"))) CAPI_TableReader *
TableOpen(const char *config, int l1, const char *columns, int l2,
          const char *delimiter, int l3) {
  CAPI_TableReader *proxy = new CAPI_TableReader;

  proxy->reader = new apsara::odps::algo::tf::plugins::odps::TableReader;
  if (!proxy->reader->Open(std::string(config, l1), std::string(columns, l2),
                           std::string(delimiter, l3))) {
    TableClose(proxy);
    return nullptr;
  }

  return proxy;
}

__attribute__((visibility("default"))) CAPI_Offset
TableGetRowCount(CAPI_TableReader *reader) {
  return reader->reader->GetRowCount();
}

__attribute__((visibility("default"))) void TableSeek(CAPI_TableReader *reader,
                                                      CAPI_Offset offset) {
  reader->reader->Seek(offset);
}

__attribute__((visibility("default"))) int TableRead(CAPI_TableReader *reader,
                                                     void *line) {
  auto read_line = reinterpret_cast<std::string *>(line);
  if (!reader->reader->Read(read_line)) {
    return -1;
  }

  return read_line->size();
}

__attribute__((visibility("default"))) void
TableReadBatch(CAPI_TableReader *reader, int count, void *batch) {
  auto record_batch =
      reinterpret_cast<std::shared_ptr<arrow::RecordBatch> *>(batch);
  *record_batch = reader->reader->GetRecordBatch(count);
}

__attribute__((visibility("default"))) void
GetTableSchema(CAPI_TableReader *reader, void *schema) {
  auto typed_schema =
      reinterpret_cast<std::unordered_map<std::string, std::string> *>(schema);
  reader->reader->GetSchema(typed_schema);
}

__attribute__((visibility("default"))) size_t
TableGetReadBytes(CAPI_TableReader *reader) {
  return reader->reader->GetReadBytes();
}

__attribute__((visibility("default"))) void
TableClose(CAPI_TableReader *reader) {
  delete reader->reader;
  delete reader;
}

__attribute__((visibility("default"))) CAPI_VolumeFileSystem *
VolumeFileSystemOpen(const char *conf, int conf_len, const char *capability,
                     int capability_len) {
  CAPI_VolumeFileSystem *fs = new CAPI_VolumeFileSystem;
  fs->fs = new apsara::odps::algo::tf::plugins::odps::VolumeFileSystem();
  if (!fs->fs->Init(std::string(conf, conf_len),
                    std::string(capability, capability_len))) {
    VolumeFileSystemClose(fs);
    return nullptr;
  }
  return fs;
}

__attribute__((visibility("default"))) bool
VolumeFileSystemIsFile(CAPI_VolumeFileSystem *fs, const char *path, bool *ret) {
  return fs->fs->IsFile(std::string(path), ret);
}

__attribute__((visibility("default"))) bool
VolumeFileSystemIsDirectory(CAPI_VolumeFileSystem *fs, const char *path,
                            bool *ret) {
  return fs->fs->IsDirectory(std::string(path), ret);
}

__attribute__((visibility("default"))) bool
VolumeFileSystemGetSize(CAPI_VolumeFileSystem *fs, const char *path,
                        size_t *ret) {
  return fs->fs->GetFileSize(std::string(path), ret);
}

__attribute__((visibility("default"))) bool
VolumeFileSystemList(CAPI_VolumeFileSystem *fs, const char *path, void *ret) {
  return fs->fs->List(std::string(path),
                      reinterpret_cast<std::vector<std::string> *>(ret));
}

__attribute__((visibility("default"))) void
VolumeFileSystemClose(CAPI_VolumeFileSystem *fs) {
  delete fs->fs;
  delete fs;
}

__attribute__((visibility("default"))) CAPI_VolumeReader *
VolumeOpen(CAPI_VolumeFileSystem *fs, const char *relative_path) {
  apsara::odps::algo::FileReaderBase *reader_base = nullptr;
  if (!fs->fs->CreateFileReader(std::string(relative_path), &reader_base)) {
    LOG(ERROR) << "create file reader failed for: " << relative_path;
    return nullptr;
  }

  CAPI_VolumeReader *reader = new CAPI_VolumeReader;
  reader->reader =
      new apsara::odps::algo::tf::plugins::odps::VolumeReader(reader_base);
  return reader;
}

__attribute__((visibility("default"))) bool
VolumeRead(CAPI_VolumeReader *reader, char *buffer, size_t len,
           ssize_t *read_len) {
  return reader->reader->Read(buffer, len, read_len);
}

__attribute__((visibility("default"))) bool
VolumeSeek(CAPI_VolumeReader *reader, CAPI_Offset offset) {
  return reader->reader->Seek(offset);
}

__attribute__((visibility("default"))) void
VolumeClose(CAPI_VolumeReader *reader) {
  delete reader->reader;
  delete reader;
}

} // endof extern "C"
