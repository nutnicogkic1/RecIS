#pragma once
#include <stdint.h>
#include <sys/stat.h>
#include <torch/extension.h>

#include <string>

#include "c10/util/intrusive_ptr.h"
namespace recis {
namespace embedding {
class SliceInfo : public at::intrusive_ptr_target {
 public:
  SliceInfo() = default;
  SliceInfo(int64_t slice_begin, int64_t slice_end, int64_t slice_size);
  int64_t slice_begin() const { return slice_begin_; }
  int64_t slice_end() const { return slice_end_; }
  int64_t slice_size() const { return slice_size_; }
  int64_t partition_num() {
    return (slice_size_ - 1) / (slice_end_ - slice_begin_) + 1;
  }

  bool IsEqualRange(const at::intrusive_ptr<SliceInfo> lhv) const {
    return slice_size_ == lhv->slice_size_;
  }

  bool IsIntersect(const at::intrusive_ptr<SliceInfo> lhv) const {
    return IsIntersectInternal(*this, *lhv) || IsIntersectInternal(*lhv, *this);
  }

  static bool IsIntersectInternal(const SliceInfo &lhv, const SliceInfo &rhv) {
    return (lhv.slice_begin_ <= rhv.slice_begin_ &&
            rhv.slice_begin_ < lhv.slice_end_) ||
           (lhv.slice_begin_ < rhv.slice_end_ &&
            rhv.slice_end_ <= lhv.slice_end_);
  }

  std::string DebugInfo() const;
  static at::intrusive_ptr<SliceInfo> FromString(const std::string &);
  static std::string ToString(const at::intrusive_ptr<SliceInfo>);

 private:
  int64_t slice_begin_;
  int64_t slice_end_;
  int64_t slice_size_;
};
}  // namespace embedding
}  // namespace recis
