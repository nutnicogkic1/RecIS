#pragma once
#include "platform/env.h"

#include <chrono>
#include <mutex>

#include "c10/util/StringUtil.h"
#include "platform/filesystem.h"
#include "platform/path.h"
#include "platform/status.h"
#include "utils/str_util.h"
namespace recis {
// 128KB copy buffer
constexpr size_t kCopyFileBufferSize = 128 * 1024;

class FileSystemRegistryImpl : public FileSystemRegistry {
 public:
  Status Register(const std::string &scheme, Factory factory) override;
  FileSystem *Lookup(const std::string &scheme) override;
  Status GetRegisteredFileSystemSchemes(
      std::vector<std::string> *schemes) override;

 private:
  mutable std::mutex mu_;
  mutable std::unordered_map<std::string, std::unique_ptr<FileSystem>>
      registry_;
};

Status FileSystemRegistryImpl::Register(const std::string &scheme,
                                        FileSystemRegistry::Factory factory) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!registry_
           .emplace(std::string(scheme), std::unique_ptr<FileSystem>(factory()))
           .second) {
    return Status(Code::ALREADY_EXISTS, torch::str("File factory for ", scheme,
                                                   " already registered"));
  }
  return Status::OK();
}

FileSystem *FileSystemRegistryImpl::Lookup(const std::string &scheme) {
  std::lock_guard<std::mutex> lock(mu_);
  const auto found = registry_.find(scheme);
  if (found == registry_.end()) {
    return nullptr;
  }
  return found->second.get();
}

Status FileSystemRegistryImpl::GetRegisteredFileSystemSchemes(
    std::vector<std::string> *schemes) {
  std::lock_guard<std::mutex> lock(mu_);
  for (const auto &e : registry_) {
    schemes->push_back(e.first);
  }
  return Status::OK();
}

Env::Env() : file_system_registry_(new FileSystemRegistryImpl) {}

Status Env::GetFileSystemForFile(const std::string &fname,
                                 FileSystem **result) {
  torch::string_view scheme, host, path;
  io::ParseURI(fname, &scheme, &host, &path);
  FileSystem *file_system =
      file_system_registry_->Lookup(util::string::Lowercase(scheme));
  if (!file_system) {
    if (scheme.empty()) {
      scheme = "[local]";
    }

    return Status(Code::UNIMPLEMENTED,
                  torch::str("File system scheme ", scheme,
                             " not implemented (file: ", fname, ")"));
  }
  *result = file_system;
  return Status::OK();
}

Status Env::GetRegisteredFileSystemSchemes(std::vector<std::string> *schemes) {
  return file_system_registry_->GetRegisteredFileSystemSchemes(schemes);
}

Status Env::RegisterFileSystem(const std::string &scheme,
                               FileSystemRegistry::Factory factory) {
  return file_system_registry_->Register(scheme, std::move(factory));
}

Status Env::FlushFileSystemCaches() {
  std::vector<std::string> schemes;
  RECIS_RETURN_IF_ERROR(GetRegisteredFileSystemSchemes(&schemes));
  for (const std::string &scheme : schemes) {
    FileSystem *fs = nullptr;
    RECIS_RETURN_IF_ERROR(
        GetFileSystemForFile(io::CreateURI(scheme, "", ""), &fs));
    fs->FlushCaches();
  }
  return Status::OK();
}

Status Env::NewRandomAccessFile(const std::string &fname,
                                std::unique_ptr<RandomAccessFile> *result) {
  FileSystem *fs;
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewRandomAccessFile(fname, result);
}

Status Env::NewReadOnlyMemoryRegionFromFile(
    const std::string &fname, std::unique_ptr<ReadOnlyMemoryRegion> *result) {
  FileSystem *fs;
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewReadOnlyMemoryRegionFromFile(fname, result);
}

Status Env::NewWritableFile(const std::string &fname,
                            std::unique_ptr<WritableFile> *result) {
  FileSystem *fs;
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewWritableFile(fname, result);
}

Status Env::NewTransactionFile(const std::string &fname,
                               std::unique_ptr<WritableFile> *result) {
  FileSystem *fs;
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewTransactionFile(fname, result);
}

Status Env::NewAppendableFile(const std::string &fname,
                              std::unique_ptr<WritableFile> *result) {
  FileSystem *fs;
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewAppendableFile(fname, result);
}

Status Env::FileExists(const std::string &fname) {
  FileSystem *fs;
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->FileExists(fname);
}

bool Env::FilesExist(const std::vector<std::string> &files,
                     std::vector<Status> *status) {
  std::unordered_map<std::string, std::vector<std::string>> files_per_fs;
  for (const auto &file : files) {
    torch::string_view scheme, host, path;
    io::ParseURI(file, &scheme, &host, &path);
    files_per_fs[std::string(scheme)].push_back(file);
  }

  std::unordered_map<std::string, Status> per_file_status;
  bool result = true;
  for (auto itr : files_per_fs) {
    FileSystem *file_system = file_system_registry_->Lookup(itr.first);
    bool fs_result;
    std::vector<Status> local_status;
    std::vector<Status> *fs_status = status ? &local_status : nullptr;
    if (!file_system) {
      fs_result = false;
      if (fs_status) {
        Status s = Status(
            Code::UNIMPLEMENTED,
            torch::str("File system scheme '", itr.first, "' not implemented"));
        local_status.resize(itr.second.size(), s);
      }
    } else {
      fs_result = file_system->FilesExist(itr.second, fs_status);
    }
    if (fs_status) {
      result &= fs_result;
      for (int i = 0; i < itr.second.size(); ++i) {
        per_file_status[itr.second[i]] = fs_status->at(i);
      }
    } else if (!fs_result) {
      // Return early
      return false;
    }
  }

  if (status) {
    for (const auto &file : files) {
      status->push_back(per_file_status[file]);
    }
  }

  return result;
}

Status Env::GetChildren(const std::string &dir,
                        std::vector<std::string> *result) {
  FileSystem *fs;
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(dir, &fs));
  return fs->GetChildren(dir, result);
}

Status Env::GetMatchingPaths(const std::string &pattern,
                             std::vector<std::string> *results) {
  FileSystem *fs;
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(pattern, &fs));
  return fs->GetMatchingPaths(pattern, results);
}

Status Env::DeleteFile(const std::string &fname) {
  FileSystem *fs;
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->DeleteFile(fname);
}

Status Env::RecursivelyCreateDir(const std::string &dirname) {
  FileSystem *fs;
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->RecursivelyCreateDir(dirname);
}

Status Env::CreateDir(const std::string &dirname) {
  FileSystem *fs;
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->CreateDir(dirname);
}

Status Env::DeleteDir(const std::string &dirname) {
  FileSystem *fs;
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->DeleteDir(dirname);
}

Status Env::Stat(const std::string &fname, FileStatistics *stat) {
  FileSystem *fs;
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->Stat(fname, stat);
}

Status Env::IsDirectory(const std::string &fname) {
  FileSystem *fs;
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->IsDirectory(fname);
}

Status Env::DeleteRecursively(const std::string &dirname,
                              int64_t *undeleted_files,
                              int64_t *undeleted_dirs) {
  FileSystem *fs;
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->DeleteRecursively(dirname, undeleted_files, undeleted_dirs);
}

Status Env::GetFileSize(const std::string &fname, uint64_t *file_size) {
  FileSystem *fs;
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->GetFileSize(fname, file_size);
}

Status Env::RenameFile(const std::string &src, const std::string &target) {
  FileSystem *src_fs;
  FileSystem *target_fs;
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(src, &src_fs));
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(target, &target_fs));
  if (src_fs != target_fs) {
    return Status(Code::UNIMPLEMENTED, torch::str("Renaming ", src, " to ",
                                                  target, " not implemented"));
  }
  return src_fs->RenameFile(src, target);
}

Status Env::TransactionRenameFile(const std::string &src,
                                  const std::string &target) {
  FileSystem *src_fs;
  FileSystem *target_fs;
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(src, &src_fs));
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(target, &target_fs));
  if (src_fs != target_fs) {
    return Status(Code::UNIMPLEMENTED, torch::str("Renaming ", src, " to ",
                                                  target, " not implemented"));
  }
  return src_fs->TransactionRenameFile(src, target);
}

Status Env::CopyFile(const std::string &src, const std::string &target) {
  FileSystem *src_fs;
  FileSystem *target_fs;
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(src, &src_fs));
  RECIS_RETURN_IF_ERROR(GetFileSystemForFile(target, &target_fs));
  if (src_fs == target_fs) {
    return src_fs->CopyFile(src, target);
  }
  return FileSystemCopyFile(src_fs, src, target_fs, target);
}

Status FileSystemCopyFile(FileSystem *src_fs, const std::string &src,
                          FileSystem *target_fs, const std::string &target) {
  std::unique_ptr<RandomAccessFile> src_file;
  RECIS_RETURN_IF_ERROR(src_fs->NewRandomAccessFile(src, &src_file));

  std::unique_ptr<WritableFile> target_file;
  RECIS_RETURN_IF_ERROR(target_fs->NewWritableFile(target, &target_file));

  uint64_t offset = 0;
  std::unique_ptr<char[]> scratch(new char[kCopyFileBufferSize]);
  Status s = Status::OK();
  while (s.ok()) {
    torch::string_view result;
    s = src_file->Read(offset, kCopyFileBufferSize, &result, scratch.get());
    if (!(s.ok() || s.code() == Code::OUT_OF_RANGE)) {
      return s;
    }
    RECIS_RETURN_IF_ERROR(target_file->Append(result));
    offset += result.size();
  }
  return target_file->Close();
}
}  // namespace recis