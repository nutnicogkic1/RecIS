#pragma once
#include "c10/util/string_view.h"
#include "platform/filesystem.h"
#include "torch/torch.h"
namespace recis {
class FileOutputBuffer {
 public:
  FileOutputBuffer(WritableFile *file, size_t buffer_size)
      : file_(file), position_(0), buffer_size_(buffer_size) {
    TORCH_CHECK_GE(buffer_size, 0);
    buffer_.resize(buffer_size);
  }
  ~FileOutputBuffer();

  // Buffered append.
  Status Append(torch::string_view data);

  Status AppendSegment(torch::string_view data);
  void EndSegment(int64_t end_bytes_written);
  // Appends the buffered data, then closes the underlying file.
  Status Close();

 private:
  // Appends the buffered data to the underlying file. Does NOT flush the file.
  Status FlushBuffer();

  WritableFile *file_;  // Owned.

  // buffer_[0, position_) holds the buffered data not yet appended to the
  // underlying file.
  size_t position_;
  const size_t buffer_size_;
  std::vector<char> buffer_;

  // Checksum of all appended bytes since construction or last clear_crc32c().
};
}  // namespace recis