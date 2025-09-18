#include "embedding/slice_info.h"

#include <string>

#include "c10/util/Exception.h"
#include "c10/util/StringUtil.h"
#include "c10/util/intrusive_ptr.h"
#include "serialize/string_sep.h"
#include "utils/str_util.h"

namespace recis {
namespace embedding {
SliceInfo::SliceInfo(int64_t slice_begin, int64_t slice_end, int64_t slice_size)
    : slice_begin_(slice_begin),
      slice_end_(slice_end),
      slice_size_(slice_size) {}

std::string SliceInfo::DebugInfo() const {
  std::string ret;
  util::string::StringAppend(ret, "[");
  util::string::StringAppend(ret, std::to_string(slice_begin_));
  util::string::StringAppend(ret, ",");
  util::string::StringAppend(ret, std::to_string(slice_end_));
  util::string::StringAppend(ret, ",");
  util::string::StringAppend(ret, std::to_string(slice_size_));
  util::string::StringAppend(ret, "]");
  return ret;
}

at::intrusive_ptr<SliceInfo> SliceInfo::FromString(const std::string& msg) {
  auto tokens = util::string::StrSplit(msg, serialize::StrSep::kIntraNameSep);
  TORCH_CHECK(tokens.size() == 3, "SliceInfo::FromString failed, msg: ", msg);
  auto ret = at::make_intrusive<SliceInfo>(
      std::stoi(tokens[0]), std::stoi(tokens[1]), std::stoi(tokens[2]));
  return ret;
}

std::string SliceInfo::ToString(const at::intrusive_ptr<SliceInfo> slice_info) {
  std::string slice_size = std::to_string(slice_info->slice_size());
  std::string slice_end = std::to_string(slice_info->slice_end());
  slice_end =
      std::string(slice_size.size() - slice_end.size(), '0') + slice_end;
  std::string slice_begin = std::to_string(slice_info->slice_begin());
  slice_begin =
      std::string(slice_size.size() - slice_begin.size(), '0') + slice_begin;
  return torch::str(slice_begin, serialize::StrSep::kIntraNameSep, slice_end,
                    serialize::StrSep::kIntraNameSep, slice_size);
}
}  // namespace embedding
}  // namespace recis
