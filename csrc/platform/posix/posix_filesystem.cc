#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#if !defined(__APPLE__)
#include <sys/sendfile.h>
#endif
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "platform/env.h"
#include "platform/file_system_helper.h"
#include "platform/posix/posix_filesystem.h"
#include "platform/status.h"

namespace recis {
// 128KB of copy buffer
constexpr size_t kPosixCopyFileBufferSize = 128 * 1024;

// pread() based random-access
class PosixRandomAccessFile : public RandomAccessFile {
 private:
  std::string filename_;
  int fd_;

 public:
  PosixRandomAccessFile(const std::string &fname, int fd)
      : filename_(fname), fd_(fd) {}
  ~PosixRandomAccessFile() override { close(fd_); }

  Status Read(uint64_t offset, size_t n, torch::string_view *result,
              char *scratch) const override {
    Status s;
    char *dst = scratch;
    while (n > 0 && s.ok()) {
      ssize_t r = pread(fd_, dst, n, static_cast<off_t>(offset));
      if (r > 0) {
        dst += r;
        n -= r;
        offset += r;
      } else if (r == 0) {
        s = Status(Code::OUT_OF_RANGE, "Read less bytes than requested");
      } else if (errno == EINTR || errno == EAGAIN) {
        // Retry
      } else {
        s = IOError(filename_, errno);
      }
    }
    *result = torch::string_view(scratch, dst - scratch);
    return s;
  }
};

class PosixWritableFile : public WritableFile {
 private:
  std::string filename_;
  FILE *file_;

 public:
  PosixWritableFile(const std::string &fname, FILE *f)
      : filename_(fname), file_(f) {}

  ~PosixWritableFile() override {
    if (file_ != nullptr) {
      // Ignoring any potential errors
      fclose(file_);
    }
  }

  Status Append(torch::string_view data) override {
    size_t r = fwrite(data.data(), 1, data.size(), file_);
    if (r != data.size()) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  Status Close() override {
    Status result;
    if (fclose(file_) != 0) {
      result = IOError(filename_, errno);
    }
    file_ = nullptr;
    return result;
  }

  Status Flush() override {
    if (fflush(file_) != 0) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  Status Sync() override {
    Status s;
    if (fflush(file_) != 0) {
      s = IOError(filename_, errno);
    }
    return s;
  }
};

class PosixReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  PosixReadOnlyMemoryRegion(const void *address, uint64_t length)
      : address_(address), length_(length) {}
  ~PosixReadOnlyMemoryRegion() override {
    munmap(const_cast<void *>(address_), length_);
  }
  const void *data() override { return address_; }
  uint64_t length() override { return length_; }

 private:
  const void *const address_;
  const uint64_t length_;
};

Status PosixFileSystem::NewRandomAccessFile(
    const std::string &fname, std::unique_ptr<RandomAccessFile> *result) {
  std::string translated_fname = TranslateName(fname);
  Status s;
  int fd = open(translated_fname.c_str(), O_RDONLY);
  if (fd < 0) {
    s = IOError(fname, errno);
  } else {
    result->reset(new PosixRandomAccessFile(translated_fname, fd));
  }
  return s;
}

Status PosixFileSystem::NewWritableFile(const std::string &fname,
                                        std::unique_ptr<WritableFile> *result) {
  std::string translated_fname = TranslateName(fname);
  Status s;
  FILE *f = fopen(translated_fname.c_str(), "w");
  if (f == nullptr) {
    s = IOError(fname, errno);
  } else {
    result->reset(new PosixWritableFile(translated_fname, f));
  }
  return s;
}

Status PosixFileSystem::NewAppendableFile(
    const std::string &fname, std::unique_ptr<WritableFile> *result) {
  std::string translated_fname = TranslateName(fname);
  Status s;
  FILE *f = fopen(translated_fname.c_str(), "a");
  if (f == nullptr) {
    s = IOError(fname, errno);
  } else {
    result->reset(new PosixWritableFile(translated_fname, f));
  }
  return s;
}

Status PosixFileSystem::NewReadOnlyMemoryRegionFromFile(
    const std::string &fname, std::unique_ptr<ReadOnlyMemoryRegion> *result) {
  std::string translated_fname = TranslateName(fname);
  Status s = Status::OK();
  int fd = open(translated_fname.c_str(), O_RDONLY);
  if (fd < 0) {
    s = IOError(fname, errno);
  } else {
    struct stat st;
    ::fstat(fd, &st);
    const void *address =
        mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (address == MAP_FAILED) {
      s = IOError(fname, errno);
    } else {
      result->reset(new PosixReadOnlyMemoryRegion(address, st.st_size));
    }
    close(fd);
  }
  return s;
}

Status PosixFileSystem::FileExists(const std::string &fname) {
  if (access(TranslateName(fname).c_str(), F_OK) == 0) {
    return Status::OK();
  }
  return Status(Code::NOT_FOUND, " not found");
}

Status PosixFileSystem::GetChildren(const std::string &dir,
                                    std::vector<std::string> *result) {
  std::string translated_dir = TranslateName(dir);
  result->clear();
  DIR *d = opendir(translated_dir.c_str());
  if (d == nullptr) {
    return IOError(dir, errno);
  }
  struct dirent *entry;
  while ((entry = readdir(d)) != nullptr) {
    torch::string_view basename = entry->d_name;
    if ((basename != ".") && (basename != "..")) {
      result->push_back(entry->d_name);
    }
  }
  closedir(d);
  return Status::OK();
}

Status PosixFileSystem::GetMatchingPaths(const std::string &pattern,
                                         std::vector<std::string> *results) {
  return internal::GetMatchingPaths(this, Env::Default(), pattern, results);
}

Status PosixFileSystem::DeleteFile(const std::string &fname) {
  Status result;
  if (unlink(TranslateName(fname).c_str()) != 0) {
    result = IOError(fname, errno);
  }
  return result;
}

Status PosixFileSystem::CreateDir(const std::string &name) {
  Status result;
  if (mkdir(TranslateName(name).c_str(), 0755) != 0) {
    result = IOError(name, errno);
  }
  return result;
}

Status PosixFileSystem::DeleteDir(const std::string &name) {
  Status result;
  if (rmdir(TranslateName(name).c_str()) != 0) {
    result = IOError(name, errno);
  }
  return result;
}

Status PosixFileSystem::GetFileSize(const std::string &fname, uint64_t *size) {
  Status s;
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    *size = 0;
    s = IOError(fname, errno);
  } else {
    *size = sbuf.st_size;
  }
  return s;
}

Status PosixFileSystem::Stat(const std::string &fname, FileStatistics *stats) {
  Status s;
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    s = IOError(fname, errno);
  } else {
    stats->length = sbuf.st_size;
    stats->mtime_nsec = sbuf.st_mtime * 1e9;
    stats->is_directory = S_ISDIR(sbuf.st_mode);
  }
  return s;
}

Status PosixFileSystem::RenameFile(const std::string &src,
                                   const std::string &target) {
  Status result;
  if (rename(TranslateName(src).c_str(), TranslateName(target).c_str()) != 0) {
    result = IOError(src, errno);
  }
  return result;
}

Status PosixFileSystem::CopyFile(const std::string &src,
                                 const std::string &target) {
  std::string translated_src = TranslateName(src);
  struct stat sbuf;
  if (stat(translated_src.c_str(), &sbuf) != 0) {
    return IOError(src, errno);
  }
  int src_fd = open(translated_src.c_str(), O_RDONLY);
  if (src_fd < 0) {
    return IOError(src, errno);
  }
  std::string translated_target = TranslateName(target);
  // O_WRONLY | O_CREAT:
  //   Open file for write and if file does not exist, create the file.
  // S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH:
  //   Create the file with permission of 0644
  int target_fd = open(translated_target.c_str(), O_WRONLY | O_CREAT,
                       S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  if (target_fd < 0) {
    close(src_fd);
    return IOError(target, errno);
  }
  int rc = 0;
  off_t offset = 0;
  std::unique_ptr<char[]> buffer(new char[kPosixCopyFileBufferSize]);
  while (offset < sbuf.st_size) {
    // Use uint64 for safe compare SSIZE_MAX
    uint64_t chunk = sbuf.st_size - offset;
    if (chunk > SSIZE_MAX) {
      chunk = SSIZE_MAX;
    }
#if defined(__linux__) && !defined(__ANDROID__)
    rc = sendfile(target_fd, src_fd, &offset, static_cast<size_t>(chunk));
#else
    if (chunk > kPosixCopyFileBufferSize) {
      chunk = kPosixCopyFileBufferSize;
    }
    rc = read(src_fd, buffer.get(), static_cast<size_t>(chunk));
    if (rc <= 0) {
      break;
    }
    rc = write(target_fd, buffer.get(), static_cast<size_t>(chunk));
    offset += chunk;
#endif
    if (rc <= 0) {
      break;
    }
  }

  Status result = Status::OK();
  if (rc < 0) {
    result = IOError(target, errno);
  }

  // Keep the error code
  rc = close(target_fd);
  if (rc < 0 && result == Status::OK()) {
    result = IOError(target, errno);
  }
  rc = close(src_fd);
  if (rc < 0 && result == Status::OK()) {
    result = IOError(target, errno);
  }

  return result;
}

Code ErrnoToCode(int err_number) {
  Code code;
  switch (err_number) {
    case 0:
      code = Code::OK;
      break;
    case EINVAL:        // Invalid argument
    case ENAMETOOLONG:  // Filename too long
    case E2BIG:         // Argument list too long
    case EDESTADDRREQ:  // Destination address required
    case EDOM:          // Mathematics argument out of domain of function
    case EFAULT:        // Bad address
    case EILSEQ:        // Illegal byte sequence
    case ENOPROTOOPT:   // Protocol not available
    case ENOSTR:        // Not a STREAM
    case ENOTSOCK:      // Not a socket
    case ENOTTY:        // Inappropriate I/O control operation
    case EPROTOTYPE:    // Protocol wrong type for socket
    case ESPIPE:        // Invalid seek
      code = Code::INVALID_ARGUMENT;
      break;
    case ETIMEDOUT:  // Connection timed out
    case ETIME:      // Timer expired
      code = Code::DEADLINE_EXCEEDED;
      break;
    case ENODEV:  // No such device
    case ENOENT:  // No such file or directory
    case ENXIO:   // No such device or address
    case ESRCH:   // No such process
      code = Code::NOT_FOUND;
      break;
    case EEXIST:         // File exists
    case EADDRNOTAVAIL:  // Address not available
    case EALREADY:       // Connection already in progress
      code = Code::ALREADY_EXISTS;
      break;
    case EPERM:   // Operation not permitted
    case EACCES:  // Permission denied
    case EROFS:   // Read only file system
      code = Code::PERMISSION_DENIED;
      break;
    case ENOTEMPTY:   // Directory not empty
    case EISDIR:      // Is a directory
    case ENOTDIR:     // Not a directory
    case EADDRINUSE:  // Address already in use
    case EBADF:       // Invalid file descriptor
    case EBUSY:       // Device or resource busy
    case ECHILD:      // No child processes
    case EISCONN:     // Socket is connected
#if !defined(_WIN32) && !defined(__HAIKU__)
    case ENOTBLK:  // Block device required
#endif
    case ENOTCONN:  // The socket is not connected
    case EPIPE:     // Broken pipe
#if !defined(_WIN32)
    case ESHUTDOWN:  // Cannot send after transport endpoint shutdown
#endif
    case ETXTBSY:  // Text file busy
      code = Code::FAILED_PRECONDITION;
      break;
    case ENOSPC:  // No space left on device
#if !defined(_WIN32)
    case EDQUOT:  // Disk quota exceeded
#endif
    case EMFILE:   // Too many open files
    case EMLINK:   // Too many links
    case ENFILE:   // Too many open files in system
    case ENOBUFS:  // No buffer space available
    case ENODATA:  // No message is available on the STREAM read queue
    case ENOMEM:   // Not enough space
    case ENOSR:    // No STREAM resources
#if !defined(_WIN32) && !defined(__HAIKU__)
    case EUSERS:  // Too many users
#endif
      code = Code::RESOURCE_EXHAUSTED;
      break;
    case EFBIG:      // File too large
    case EOVERFLOW:  // Value too large to be stored in data type
    case ERANGE:     // Result too large
      code = Code::OUT_OF_RANGE;
      break;
    case ENOSYS:        // Function not implemented
    case ENOTSUP:       // Operation not supported
    case EAFNOSUPPORT:  // Address family not supported
#if !defined(_WIN32)
    case EPFNOSUPPORT:  // Protocol family not supported
#endif
    case EPROTONOSUPPORT:  // Protocol not supported
#if !defined(_WIN32) && !defined(__HAIKU__)
    case ESOCKTNOSUPPORT:  // Socket type not supported
#endif
    case EXDEV:  // Improper link
      code = Code::UNIMPLEMENTED;
      break;
    case EAGAIN:        // Resource temporarily unavailable
    case ECONNREFUSED:  // Connection refused
    case ECONNABORTED:  // Connection aborted
    case ECONNRESET:    // Connection reset
    case EINTR:         // Interrupted function call
#if !defined(_WIN32)
    case EHOSTDOWN:  // Host is down
#endif
    case EHOSTUNREACH:  // Host is unreachable
    case ENETDOWN:      // Network is down
    case ENETRESET:     // Connection aborted by network
    case ENETUNREACH:   // Network unreachable
    case ENOLCK:        // No locks available
    case ENOLINK:       // Link has been severed
#if !(defined(__APPLE__) || defined(__FreeBSD__) || defined(_WIN32) || \
      defined(__HAIKU__))
    case ENONET:  // Machine is not on the network
#endif
      code = Code::UNAVAILABLE;
      break;
    case EDEADLK:  // Resource deadlock avoided
#if !defined(_WIN32)
    case ESTALE:  // Stale file handle
#endif
      code = Code::ABORTED;
      break;
    case ECANCELED:  // Operation cancelled
      code = Code::CANCELLED;
      break;
    // NOTE: If you get any of the following (especially in a
    // reproducible way) and can propose a better mapping,
    // please email the owners about updating this mapping.
    case EBADMSG:      // Bad message
    case EIDRM:        // Identifier removed
    case EINPROGRESS:  // Operation in progress
    case EIO:          // I/O error
    case ELOOP:        // Too many levels of symbolic links
    case ENOEXEC:      // Exec format error
    case ENOMSG:       // No message of the desired type
    case EPROTO:       // Protocol error
#if !defined(_WIN32) && !defined(__HAIKU__)
    case EREMOTE:  // Object is remote
#endif
      code = Code::UNKNOWN;
      break;
    default: {
      code = Code::UNKNOWN;
      break;
    }
  }
  return code;
}

Status IOError(const std::string &context, int err_number) {
  auto code = ErrnoToCode(err_number);
  return Status(code, torch::str(context, "; ", strerror(err_number)));
}
}  // namespace recis