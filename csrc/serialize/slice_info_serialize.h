#pragma once
#include <string>

#include "embedding/hashtable.h"
namespace recis {
namespace serialize {
const std::string SerializeSliceInfo(const embedding::SliceInfo &slice_info);
embedding::SliceInfo DeserializeSliceInfo(const std::string &message);
}  // namespace serialize
}  // namespace recis