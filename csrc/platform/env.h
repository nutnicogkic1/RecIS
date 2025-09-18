#pragma once
#include "ATen/Utils.h"
#include "platform/filesystem.h"
namespace recis {
class Env {
 public:
  Env();
  virtual ~Env() = default;

  /// \brief Returns a default environment suitable for the current operating
  /// system.
  ///
  /// Sophisticated users may wish to provide their own Env
  /// implementation instead of relying on this default environment.
  ///
  /// The result of Default() belongs to this library and must never be deleted.
  static Env *Default();

  /// \brief Returns the FileSystem object to handle operations on the file
  /// specified by 'fname'. The FileSystem object is used as the implementation
  /// for the file system related (non-virtual) functions that follow.
  /// Returned FileSystem object is still owned by the Env object and will
  // (might) be destroyed when the environment is destroyed.
  virtual Status GetFileSystemForFile(const std::string &fname,
                                      FileSystem **result);

  /// \brief Returns the file system schemes registered for this Env.
  virtual Status GetRegisteredFileSystemSchemes(
      std::vector<std::string> *schemes);

  /// \brief Register a file system for a scheme.
  virtual Status RegisterFileSystem(const std::string &scheme,
                                    FileSystemRegistry::Factory factory);

  /// \brief Flush filesystem caches for all registered filesystems.
  Status FlushFileSystemCaches();

  /// \brief Creates a brand new random access read-only file with the
  /// specified name.

  /// On success, stores a pointer to the new file in
  /// *result and returns OK.  On failure stores NULL in *result and
  /// returns non-OK.  If the file does not exist, returns a non-OK
  /// status.
  ///
  /// The returned file may be concurrently accessed by multiple threads.
  ///
  /// The ownership of the returned RandomAccessFile is passed to the caller
  /// and the object should be deleted when is not used. The file object
  /// shouldn't live longer than the Env object.
  Status NewRandomAccessFile(const std::string &fname,
                             std::unique_ptr<RandomAccessFile> *result);

  /// \brief Creates an object that writes to a new file with the specified
  /// name.
  ///
  /// Deletes any existing file with the same name and creates a
  /// new file.  On success, stores a pointer to the new file in
  /// *result and returns OK.  On failure stores NULL in *result and
  /// returns non-OK.
  ///
  /// The returned file will only be accessed by one thread at a time.
  ///
  /// The ownership of the returned WritableFile is passed to the caller
  /// and the object should be deleted when is not used. The file object
  /// shouldn't live longer than the Env object.
  Status NewWritableFile(const std::string &fname,
                         std::unique_ptr<WritableFile> *result);

  /// \brief Create an object that writes to a new file with the specifed
  //  name.
  //  the function is added to speed up write performance in some fs (like oss).
  Status NewTransactionFile(const std::string &fname,
                            std::unique_ptr<WritableFile> *result);

  /// \brief Creates an object that either appends to an existing file, or
  /// writes to a new file (if the file does not exist to begin with).
  ///
  /// On success, stores a pointer to the new file in *result and
  /// returns OK.  On failure stores NULL in *result and returns
  /// non-OK.
  ///
  /// The returned file will only be accessed by one thread at a time.
  ///
  /// The ownership of the returned WritableFile is passed to the caller
  /// and the object should be deleted when is not used. The file object
  /// shouldn't live longer than the Env object.
  Status NewAppendableFile(const std::string &fname,
                           std::unique_ptr<WritableFile> *result);

  /// \brief Creates a readonly region of memory with the file context.
  ///
  /// On success, it returns a pointer to read-only memory region
  /// from the content of file fname. The ownership of the region is passed to
  /// the caller. On failure stores nullptr in *result and returns non-OK.
  ///
  /// The returned memory region can be accessed from many threads in parallel.
  ///
  /// The ownership of the returned ReadOnlyMemoryRegion is passed to the caller
  /// and the object should be deleted when is not used. The memory region
  /// object shouldn't live longer than the Env object.
  Status NewReadOnlyMemoryRegionFromFile(
      const std::string &fname, std::unique_ptr<ReadOnlyMemoryRegion> *result);

  /// Returns OK if the named path exists and NOT_FOUND otherwise.
  Status FileExists(const std::string &fname);

  /// Returns true if all the listed files exist, false otherwise.
  /// if status is not null, populate the vector with a detailed status
  /// for each file.
  bool FilesExist(const std::vector<std::string> &files,
                  std::vector<Status> *status);

  /// \brief Stores in *result the names of the children of the specified
  /// directory. The names are relative to "dir".
  ///
  /// Original contents of *results are dropped.
  Status GetChildren(const std::string &dir, std::vector<std::string> *result);

  /// \brief Returns true if the path matches the given pattern. The wildcards
  /// allowed in pattern are described in FileSystem::GetMatchingPaths.
  virtual bool MatchPath(const std::string &path,
                         const std::string &pattern) = 0;

  /// \brief Given a pattern, stores in *results the set of paths that matches
  /// that pattern. *results is cleared.
  ///
  /// More details about `pattern` in FileSystem::GetMatchingPaths.
  virtual Status GetMatchingPaths(const std::string &pattern,
                                  std::vector<std::string> *results);

  /// Deletes the named file.
  Status DeleteFile(const std::string &fname);

  /// \brief Deletes the specified directory and all subdirectories and files
  /// underneath it. undeleted_files and undeleted_dirs stores the number of
  /// files and directories that weren't deleted (unspecified if the return
  /// status is not OK).
  /// REQUIRES: undeleted_files, undeleted_dirs to be not null.
  /// Typical return codes
  ///  * OK - dirname exists and we were able to delete everything underneath.
  ///  * NOT_FOUND - dirname doesn't exist
  ///  * PERMISSION_DENIED - dirname or some descendant is not writable
  ///  * UNIMPLEMENTED - Some underlying functions (like Delete) are not
  ///                    implemented
  Status DeleteRecursively(const std::string &dirname, int64_t *undeleted_files,
                           int64_t *undeleted_dirs);

  /// \brief Creates the specified directory and all the necessary
  /// subdirectories. Typical return codes.
  ///  * OK - successfully created the directory and sub directories, even if
  ///         they were already created.
  ///  * PERMISSION_DENIED - dirname or some subdirectory is not writable.
  Status RecursivelyCreateDir(const std::string &dirname);

  /// \brief Creates the specified directory. Typical return codes
  ///  * OK - successfully created the directory.
  ///  * ALREADY_EXISTS - directory already exists.
  ///  * PERMISSION_DENIED - dirname is not writable.
  Status CreateDir(const std::string &dirname);

  /// Deletes the specified directory.
  Status DeleteDir(const std::string &dirname);

  /// Obtains statistics for the given path.
  Status Stat(const std::string &fname, FileStatistics *stat);

  /// \brief Returns whether the given path is a directory or not.
  /// Typical return codes (not guaranteed exhaustive):
  ///  * OK - The path exists and is a directory.
  ///  * FAILED_PRECONDITION - The path exists and is not a directory.
  ///  * NOT_FOUND - The path entry does not exist.
  ///  * PERMISSION_DENIED - Insufficient permissions.
  ///  * UNIMPLEMENTED - The file factory doesn't support directories.
  Status IsDirectory(const std::string &fname);

  /// Stores the size of `fname` in `*file_size`.
  Status GetFileSize(const std::string &fname, uint64_t *file_size);

  /// \brief Renames file src to target. If target already exists, it will be
  /// replaced.
  Status RenameFile(const std::string &src, const std::string &target);

  /// \brief Renames file src to target. If target already exists, it will be
  /// replaced.
  //  the function is added to speed up rename performance in some fs
  //  (like oss).
  Status TransactionRenameFile(const std::string &src,
                               const std::string &target);

  /// \brief Copy the src to target.
  Status CopyFile(const std::string &src, const std::string &target);

 private:
  std::unique_ptr<FileSystemRegistry> file_system_registry_;
  AT_DISALLOW_COPY_AND_ASSIGN(Env);
};

Status FileSystemCopyFile(FileSystem *src_fs, const std::string &src,
                          FileSystem *target_fs, const std::string &target);
namespace register_file_system {

template <typename Factory>
struct Register {
  Register(Env *env, const std::string &scheme) {
    // TODO(b/32704451): Don't just ignore the ::tensorflow::Status object!
    env->RegisterFileSystem(scheme,
                            []() -> FileSystem * { return new Factory; })
        .IgnoreError();
  }
};

}  // namespace register_file_system
}  // namespace recis
// Register a FileSystem implementation for a scheme. Files with names that have
// "scheme://" prefixes are routed to use this implementation.
#define REGISTER_FILE_SYSTEM_ENV(env, scheme, factory) \
  REGISTER_FILE_SYSTEM_UNIQ_HELPER(__COUNTER__, env, scheme, factory)
#define REGISTER_FILE_SYSTEM_UNIQ_HELPER(ctr, env, scheme, factory) \
  REGISTER_FILE_SYSTEM_UNIQ(ctr, env, scheme, factory)
#define REGISTER_FILE_SYSTEM_UNIQ(ctr, env, scheme, factory)               \
  static ::recis::register_file_system::Register<factory> register_ff##ctr \
      __attribute__((unused)) =                                            \
          ::recis::register_file_system::Register<factory>(env, scheme)

#define REGISTER_FILE_SYSTEM(scheme, factory) \
  REGISTER_FILE_SYSTEM_ENV(::recis::Env::Default(), scheme, factory);