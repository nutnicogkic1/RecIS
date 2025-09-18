#include "serialize/slice_info_serialize.h"

#include <string>

#include "embedding/hashtable.h"
#include "embedding/slice_info.h"
#include "serialize/string_sep.h"
#include "utils/str_util.h"
namespace recis {
namespace serialize {
const std::string SerializeSliceInfo(const embedding::SliceInfo &slice_info) {
  return torch::str(slice_info.slice_begin(), StrSep::kInterNameSep,
                    slice_info.slice_end(), StrSep::kInterNameSep,
                    slice_info.slice_size());
}

embedding::SliceInfo DeserializeSliceInfo(const std::string &message) {
  auto tokens = util::string::StrSplit(message, StrSep::kInterNameSep);
  TORCH_CHECK(tokens.size() == 3, "message format mismatch: ", message);
  return embedding::SliceInfo(std::stoll(tokens[0]), std::stoll(tokens[1]),
                              std::stoll(tokens[2]));
}
}  // namespace serialize
}  // namespace recis