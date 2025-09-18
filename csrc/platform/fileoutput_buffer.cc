#include "platform/fileoutput_buffer.h"

#include "c10/util/string_view.h"
namespace recis {
FileOutputBuffer::~FileOutputBuffer() { delete file_; }

Status FileOutputBuffer::Append(torch::string_view data) {
  // In the below, it is critical to calculate the checksum on the actually
  // copied bytes, not the source bytes.  This is because "data" typically
  // points to tensor buffers, which may be concurrently written.
  if (data.size() + position_ <= buffer_size_) {
    // Can fit into the current buffer.
    memcpy(&buffer_[position_], data.data(), data.size());
  } else if (data.size() <= buffer_size_) {
    // Cannot fit, but can fit after flushing.
    RECIS_RETURN_IF_ERROR(FlushBuffer());
    memcpy(&buffer_[0], data.data(), data.size());
  } else {
    // Cannot fit even after flushing.  So we break down "data" by chunk, and
    // flush/checksum each chunk.
    RECIS_RETURN_IF_ERROR(FlushBuffer());
    for (size_t i = 0; i < data.size(); i += buffer_size_) {
      const size_t nbytes = std::min(data.size() - i, buffer_size_);
      memcpy(&buffer_[0], data.data() + i, nbytes);
      position_ = nbytes;
      RECIS_RETURN_IF_ERROR(FlushBuffer());
    }
    return Status::OK();
  }
  position_ += data.size();
  return Status::OK();
}

Status FileOutputBuffer::AppendSegment(torch::string_view data) {
  RECIS_RETURN_IF_ERROR(FlushBuffer());
  memcpy(&buffer_[0], data.data(), data.size());
  position_ = data.size();
  RECIS_RETURN_IF_ERROR(FlushBuffer());
  return Status::OK();
}

Status FileOutputBuffer::Close() {
  RECIS_RETURN_IF_ERROR(FlushBuffer());
  return file_->Close();
}

Status FileOutputBuffer::FlushBuffer() {
  if (position_ > 0) {
    RECIS_RETURN_IF_ERROR(
        file_->Append(torch::string_view(&buffer_[0], position_)));
    position_ = 0;
  }
  return Status::OK();
}
}  // namespace recis