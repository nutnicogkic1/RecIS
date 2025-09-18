#include "platform/filesystem.h"

#include "c10/util/Exception.h"
#include "c10/util/logging_is_not_google_glog.h"
#include "c10/util/string_view.h"
#include "platform/env.h"
#include "platform/path.h"
#include "platform/status.h"
#include "utils/str_util.h"
namespace recis {
FileSystem::~FileSystem() {}

std::string FileSystem::TranslateName(const std::string &name) const {
  // If the name is empty, CleanPath returns "." which is incorrect and
  // we should return the empty path instead.
  if (name.empty()) return name;
  return io::CleanPath(name);
}

Status FileSystem::IsDirectory(const std::string &name) {
  // Check if path exists.
  RECIS_RETURN_IF_ERROR(FileExists(name));
  FileStatistics stat;
  RECIS_RETURN_IF_ERROR(Stat(name, &stat));
  if (stat.is_directory) {
    return Status::OK();
  }
  return Status(Code::FAILED_PRECONDITION, "Not a directory");
}

void FileSystem::FlushCaches() {}

RandomAccessFile::~RandomAccessFile() {}

WritableFile::~WritableFile() {}

FileSystemRegistry::~FileSystemRegistry() {}

bool FileSystem::FilesExist(const std::vector<std::string> &files,
                            std::vector<Status> *status) {
  bool result = true;
  for (const auto &file : files) {
    Status s = FileExists(file);
    result &= s.ok();
    if (status != nullptr) {
      status->push_back(s);
    } else if (!result) {
      // Return early since there is no need to check other files.
      return false;
    }
  }
  return result;
}

Status FileSystem::DeleteRecursively(const std::string &dirname,
                                     int64_t *undeleted_files,
                                     int64_t *undeleted_dirs) {
  TORCH_CHECK_NOTNULL(undeleted_dirs);
  TORCH_CHECK_NOTNULL(undeleted_files);
  *undeleted_files = 0;
  *undeleted_dirs = 0;
  // Make sure that dirname exists;
  Status exists_status = FileExists(dirname);
  if (!exists_status.ok()) {
    (*undeleted_dirs)++;
    return exists_status;
  }
  std::deque<std::string> dir_q;      // Queue for the BFS
  std::vector<std::string> dir_list;  // List of all dirs discovered
  dir_q.push_back(dirname);
  Status ret;  // Status to be returned.
  // Do a BFS on the directory to discover all the sub-directories. Remove all
  // children that are files along the way. Then cleanup and remove the
  // directories in reverse order.;
  while (!dir_q.empty()) {
    std::string dir = dir_q.front();
    dir_q.pop_front();
    dir_list.push_back(dir);
    std::vector<std::string> children;
    // GetChildren might fail if we don't have appropriate permissions.
    Status s = GetChildren(dir, &children);
    ret.Update(s);
    if (!s.ok()) {
      (*undeleted_dirs)++;
      continue;
    }
    for (const std::string &child : children) {
      const std::string child_path = io::JoinPath(dir, child);
      // If the child is a directory add it to the queue, otherwise delete it.
      if (IsDirectory(child_path).ok()) {
        dir_q.push_back(child_path);
      } else {
        // Delete file might fail because of permissions issues or might be
        // unimplemented.
        Status del_status = DeleteFile(child_path);
        ret.Update(del_status);
        if (!del_status.ok()) {
          (*undeleted_files)++;
        }
      }
    }
  }
  // Now reverse the list of directories and delete them. The BFS ensures that
  // we can delete the directories in this order.
  std::reverse(dir_list.begin(), dir_list.end());
  for (const std::string &dir : dir_list) {
    // Delete dir might fail because of permissions issues or might be
    // unimplemented.
    Status s = DeleteDir(dir);
    ret.Update(s);
    if (!s.ok()) {
      (*undeleted_dirs)++;
    }
  }
  return ret;
}

Status FileSystem::RecursivelyCreateDir(const std::string &dirname) {
  torch::string_view scheme, host, remaining_dir;
  io::ParseURI(dirname, &scheme, &host, &remaining_dir);
  std::vector<torch::string_view> sub_dirs;
  while (!remaining_dir.empty()) {
    Status status = FileExists(io::CreateURI(scheme, host, remaining_dir));
    if (status.ok()) {
      break;
    }
    if (status.code() != Code::NOT_FOUND) {
      return status;
    }
    // Basename returns "" for / ending dirs.
    if (!util::string::EndsWith(remaining_dir, "/")) {
      sub_dirs.push_back(io::Basename(remaining_dir));
    }
    remaining_dir = io::Dirname(remaining_dir);
  }

  // sub_dirs contains all the dirs to be created but in reverse order.
  std::reverse(sub_dirs.begin(), sub_dirs.end());

  // Now create the directories.
  std::string built_path(remaining_dir);
  for (const torch::string_view &sub_dir : sub_dirs) {
    built_path = io::JoinPath(built_path, sub_dir);
    Status status = CreateDir(io::CreateURI(scheme, host, built_path));
    if (!status.ok() && status.code() != Code::ALREADY_EXISTS) {
      return status;
    }
  }
  return Status::OK();
}

Status FileSystem::NewTransactionFile(const std::string &fname,
                                      std::unique_ptr<WritableFile> *result) {
  return NewWritableFile(fname, result);
}

Status FileSystem::TransactionRenameFile(const std::string &src,
                                         const std::string &target) {
  return RenameFile(src, target);
}

Status FileSystem::CopyFile(const std::string &src, const std::string &target) {
  return FileSystemCopyFile(this, src, this, target);
}

}  // namespace recis