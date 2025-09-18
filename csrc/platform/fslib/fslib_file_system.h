#include "platform/filesystem.h"
#pragma once
namespace recis {
class FslibTfFileSystem : public FileSystem {
 public:
  FslibTfFileSystem();
  virtual ~FslibTfFileSystem() = default;

 public:
  Status NewRandomAccessFile(
      const std::string &fname,
      std::unique_ptr<RandomAccessFile> *result) override;

  Status NewWritableFile(const std::string &fname,
                         std::unique_ptr<WritableFile> *result) override;

  Status NewAppendableFile(const std::string &fname,
                           std::unique_ptr<WritableFile> *result) override;

  Status NewReadOnlyMemoryRegionFromFile(
      const std::string &fname,
      std::unique_ptr<ReadOnlyMemoryRegion> *result) override;

  Status FileExists(const std::string &fname) override;

  Status GetChildren(const std::string &dir,
                     std::vector<std::string> *result) override;

  Status DeleteFile(const std::string &fname) override;

  Status CreateDir(const std::string &name) override;

  Status DeleteDir(const std::string &name) override;

  Status GetFileSize(const std::string &fname, uint64_t *size) override;

  Status RenameFile(const std::string &src, const std::string &target) override;

  Status Stat(const std::string &fname, FileStatistics *stat) override;

  Status GetMatchingPaths(const std::string &pattern,
                          std::vector<std::string> *results) override;

  std::string TranslateName(const std::string &name) const override;
};
}  // namespace recis