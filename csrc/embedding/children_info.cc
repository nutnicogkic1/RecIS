#include "embedding/children_info.h"

#include <string>

#include "ATen/TensorUtils.h"
#include "c10/util/Exception.h"

namespace recis {
namespace embedding {

int64_t ChildrenInfo::EncodeId(int64_t id, int64_t index) {
  return (id & IdMask()) | (index << IdBitsNum());
}

ChildrenInfo::ChildrenInfo(bool is_coalesce) : coalesce_(is_coalesce) {}

void ChildrenInfo::AddChild(const std::string &child) {
  children_.push_back(child);
  child_index_[child] = children_.size() - 1;
}

const std::vector<std::string> &ChildrenInfo::Children() { return children_; }

int64_t ChildrenInfo::ChildIndex(const std::string &child) {
  return child_index_[child];
}

bool ChildrenInfo::HasChild(const std::string &child) {
  return child_index_.count(child) > 0;
}

const std::string &ChildrenInfo::ChildAt(int64_t index) const {
  return children_[index];
}

bool ChildrenInfo::IsCoalesce() { return coalesce_; }

void ChildrenInfo::Validate() {
  TORCH_CHECK(children_.size() > 0, "empty children");
  TORCH_CHECK(children_.size() < MaxChildrenNum(), "children size[",
              children_.size(), "] exceeds max children size[",
              MaxChildrenNum(), "]")
  TORCH_CHECK(coalesce_ || (!coalesce_ && children_.size()) == 1,
              "non coalesce has more than one child");
  std::set<std::string> const children_set(children_.begin(), children_.end());
  TORCH_CHECK(children_set.size() == children_.size(), "duplicate child[",
              children_, "]");
}
}  // namespace embedding
}  // namespace recis
