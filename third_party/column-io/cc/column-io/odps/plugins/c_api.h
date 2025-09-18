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

#ifndef PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_C_API_H_
#define PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_C_API_H_

#include <stddef.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef size_t CAPI_Offset;
typedef struct CAPI_TableReader CAPI_TableReader;
typedef struct CAPI_VolumeFileSystem CAPI_VolumeFileSystem;
typedef struct CAPI_VolumeReader CAPI_VolumeReader;

extern CAPI_TableReader *TableOpen(const char *config, int l1,
                                   const char *columns, int l2,
                                   const char *delimiter, int l3);

extern CAPI_Offset TableGetRowCount(CAPI_TableReader *reader);

extern void TableSeek(CAPI_TableReader *reader, CAPI_Offset offset);

extern int TableRead(CAPI_TableReader *reader, void *line);

extern void TableReadBatch(CAPI_TableReader *reader, int count, void *batch);

extern void GetTableSchema(CAPI_TableReader *reader, void *schema);

extern size_t TableGetReadBytes(CAPI_TableReader *reader);

extern void TableClose(CAPI_TableReader *reader);

extern CAPI_VolumeFileSystem *VolumeFileSystemOpen(const char *conf,
                                                   int conf_len,
                                                   const char *capability,
                                                   int capability_len);

extern void VolumeFileSystemClose(CAPI_VolumeFileSystem *fs);

extern bool VolumeFileSystemIsFile(CAPI_VolumeFileSystem *fs, const char *path,
                                   bool *ret);

extern bool VolumeFileSystemIsDirectory(CAPI_VolumeFileSystem *fs,
                                        const char *path, bool *ret);

extern bool VolumeFileSystemGetSize(CAPI_VolumeFileSystem *fs, const char *path,
                                    size_t *ret);

extern bool VolumeFileSystemList(CAPI_VolumeFileSystem *fs, const char *path,
                                 void *ret);

extern CAPI_VolumeReader *VolumeOpen(CAPI_VolumeFileSystem *fs,
                                     const char *relative_path);

extern bool VolumeRead(CAPI_VolumeReader *reader, char *buffer, size_t len,
                       ssize_t *read_len);

extern bool VolumeSeek(CAPI_VolumeReader *reader, CAPI_Offset offset);

extern void VolumeClose(CAPI_VolumeReader *reader);

#ifdef __cplusplus
} // end of extern "C"
#endif

#endif // PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_C_API_H_
