#include "str_util.h"

#include <string.h>

#include <cstring>
#include <string>

#include "c10/util/string_view.h"

namespace recis {
namespace util {
namespace string {
std::vector<std::string> StrSplit(c10::string_view text, char delim) {
  size_t start = 0;
  size_t end = 0;

  std::vector<std::string> tokens;
  while ((start = text.find_first_not_of(delim, end)) != std::string::npos) {
    end = text.find(delim, start);
    auto token = text.substr(start, end - start);
    tokens.emplace_back(token.begin(), token.end());
  }
  return tokens;
}

std::vector<std::string> StrSplit(c10::string_view text,
                                  const std::string &delim) {
  std::vector<std::string> result;
  size_t start = 0;
  size_t end = text.find(delim, start);

  while (end != c10::string_view::npos) {
    result.emplace_back(text.substr(start, end - start));
    start = end + delim.length();
    end = text.find(delim, start);
  }
  result.emplace_back(text.substr(start));

  return result;
}

bool ConsumePrefix(c10::string_view &text, c10::string_view prefix) {
  if (text.substr(0, prefix.size()) == prefix) {
    text.remove_prefix(prefix.size());
    return true;
  }
  return false;
}

bool EndsWith(c10::string_view text, c10::string_view suffix) {
  return suffix.empty() || (text.size() >= suffix.size() &&
                            memcmp(text.data() + (text.size() - suffix.size()),
                                   suffix.data(), suffix.size()) == 0);
}

bool StartsWith(c10::string_view test, c10::string_view prefix) {
  if (test.size() < prefix.size()) {
    return false;
  }
  return test.substr(0, prefix.size()) == prefix;
}

std::string Lowercase(c10::string_view s) {
  std::string result(s.data(), s.size());
  for (char &c : result) {
    c = tolower(c);
  }
  return result;
}

void StringAppend(std::string &dst, const std::string &src) { dst.append(src); }

void StringAppend(std::string &dst, const c10::string_view &src) {
  dst.append(src.data(), src.size());
}

void StringAppend(std::string &dst, const char *src) { dst.append(src); }

std::string Replace(const std::string &str, const std::string &src,
                    const std::string &dst) {
  std::string res = str;
  if (str.find(src) != std::string::npos) {
    size_t pos = str.find(src);
    res.replace(pos, src.length(), dst);
  } else {
    LOG(WARNING) << " replace string " << str << "[" << src << "] to [" << dst
                 << "]"
                 << "failed, return " << str;
  }
  return res;
}

}  // namespace string
}  // namespace util
}  // namespace recis
