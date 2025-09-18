#pragma once
#include <vector>

#include "c10/util/Logging.h"
#include "c10/util/logging_is_not_google_glog.h"
#include "c10/util/string_view.h"
namespace recis {
namespace util {
namespace string {
std::vector<std::string> StrSplit(c10::string_view text, char delim);
std::vector<std::string> StrSplit(c10::string_view text,
                                  const std::string &delim);
bool ConsumePrefix(c10::string_view &text, c10::string_view prefix);
bool EndsWith(c10::string_view text, c10::string_view suffix);
bool StartsWith(c10::string_view test, c10::string_view prefix);
std::string Lowercase(c10::string_view s);
void StringAppend(std::string &dst, const std::string &src);
void StringAppend(std::string &dst, const c10::string_view &src);
void StringAppend(std::string &dst, const char *src);
std::string Replace(const std::string &str, const std::string &src,
                    const std::string &dst);
}  // namespace string
}  // namespace util
}  // namespace recis
