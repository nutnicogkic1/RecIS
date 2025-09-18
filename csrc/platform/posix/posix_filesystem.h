#include "c10/util/string_view.h"
#include "platform/filesystem.h"
#include "platform/path.h"
#pragma once
namespace recis {
class PosixFileSystem : public FileSystem {
 public:
  PosixFileSystem() {}

  ~PosixFileSystem() {}

  Status NewRandomAccessFile(
      const std::string &filename,
      std::unique_ptr<RandomAccessFile> *result) override;

  Status NewWritableFile(const std::string &fname,
                         std::unique_ptr<WritableFile> *result) override;

  Status NewAppendableFile(const std::string &fname,
                           std::unique_ptr<WritableFile> *result) override;

  Status NewReadOnlyMemoryRegionFromFile(
      const std::string &filename,
      std::unique_ptr<ReadOnlyMemoryRegion> *result) override;

  Status FileExists(const std::string &fname) override;

  Status GetChildren(const std::string &dir,
                     std::vector<std::string> *result) override;

  Status Stat(const std::string &fname, FileStatistics *stats) override;

  Status GetMatchingPaths(const std::string &pattern,
                          std::vector<std::string> *results) override;

  Status DeleteFile(const std::string &fname) override;

  Status CreateDir(const std::string &name) override;

  Status DeleteDir(const std::string &name) override;

  Status GetFileSize(const std::string &fname, uint64_t *size) override;

  Status RenameFile(const std::string &src, const std::string &target) override;

  Status CopyFile(const std::string &src, const std::string &target) override;
};

Status IOError(const std::string &context, int err_number);

class LocalPosixFileSystem : public PosixFileSystem {
 public:
  std::string TranslateName(const std::string &name) const override {
    torch::string_view scheme, host, path;
    io::ParseURI(name, &scheme, &host, &path);
    return std::string(path);
  }
};
}  // namespace recis