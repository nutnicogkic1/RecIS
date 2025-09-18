#ifndef TENSORFLOW_CORE_PLATFORM_ODPS_FILE_SYSTEM_H_
#define TENSORFLOW_CORE_PLATFORM_ODPS_FILE_SYSTEM_H_

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "column-io/framework/column_reader.h"
#include "column-io/framework/status.h"

namespace column {
namespace framework {

class FileSystem {
public:
  virtual ~FileSystem() {}

  virtual Status Exists(const std::string &path, bool *ret) = 0;
  virtual Status IsFile(const std::string &path, bool *ret) = 0;
  virtual Status IsDirectory(const std::string &path, bool *ret) = 0;
  virtual Status GetFileSize(const std::string &path, size_t *ret) = 0;
  virtual Status List(const std::string &path,
                      std::vector<std::string> *ret) = 0;
  virtual Status Move(const std::string &source, const std::string &target) = 0;
  virtual Status Delete(const std::string &path) = 0;
  virtual Status CreateDirectory(const std::string &path) = 0;
  virtual Status CreateFileReader(const std::string &path,
                                  ColumnReader **ret) = 0;
  virtual Status CreateFileWriter(const std::string &path,
                                  ColumnReader **ret) = 0;

  virtual Status CreateTransactionWriter(const std::string &path,
                                         ColumnReader **ret) {
    return Status::Unimplemented();
  }
  virtual Status TransactionMove(const std::string &source,
                                 const std::string &target) {
    return Status::Unimplemented();
  }
};

FileSystem *NewVolumeFileSystem(const std::string &conf,
                                const std::string &capability);
FileSystem *NewCacheVolumeFileSystem(const std::string &conf,
                                     const std::string &capability);
FileSystem *NewOssFileSystem(const std::string &conf);
FileSystem *NewCacheOssFileSystem(const std::string &conf);
FileSystem *NewTableFileSystem();

#ifdef TF_ENABLE_PANGU_TEMP
FileSystem *NewPanguTempFileSystem(const std::string &conf,
                                   const std::string &capability);
#endif // TF_ENABLE_PANGU_TEMP

void ReleaseFileSystem(FileSystem *fs);
void ReleaseFileReader(ColumnReader *reader);
void ReleaseFileWriter(ColumnReader *writer);

} // namespace framework
} // namespace column

#endif // TENSORFLOW_CORE_PLATFORM_ODPS_FILE_SYSTEM_H_
