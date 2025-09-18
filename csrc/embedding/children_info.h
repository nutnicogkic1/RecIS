#pragma once
#include <stdint.h>

#include <cstddef>
#include <string>
#include <vector>

#include "ATen/core/ivalue.h"
#include "c10/util/flat_hash_map.h"
namespace recis {
namespace embedding {
class ChildrenInfo : public torch::CustomClassHolder {
 public:
  constexpr static int64_t IndexBitsNum() { return 12; }
  constexpr static int64_t IdBitsNum() { return 64 - IndexBitsNum(); }
  constexpr static int64_t IndexMask() { return ~IdMask(); }
  constexpr static int64_t IdMask() { return (1LL << IdBitsNum()) - 1; }
  constexpr static int64_t MaxChildrenNum() {
    return (1LL << IndexBitsNum()) - 1;
  }
  static int64_t EncodeId(int64_t id, int64_t index);
  ChildrenInfo(bool is_coalesce);
  void AddChild(const std::string &child);

  const std::vector<std::string> &Children();
  int64_t ChildIndex(const std::string &child);
  bool HasChild(const std::string &child);
  const std::string &ChildAt(int64_t index) const;
  bool IsCoalesce();

  void Validate();

 private:
  std::vector<std::string> children_;
  ska::flat_hash_map<std::string, int64_t> child_index_;
  bool coalesce_;
};
}  // namespace embedding
}  // namespace recis
