#include "column-io/lake/lake_fslib_helper.h"
#include <algorithm>
#include <memory.h>
#include <stdlib.h>
#include <string>

extern "C" {

const char *GetListDir(const char *dir) {
  std::vector<std::string> result;
  bool flag = lake::getListDir(std::string(dir), &result);
  if (!flag) {
    return "";
  }
  sort(result.begin(), result.end());
  size_t total_size = 0;
  for (size_t i = 0; i < result.size(); i++) {
    total_size += result[i].size();
    total_size++;
  }
  if( total_size == 0){
    return "";
  }
  int offset = 0;
  char *buffer = reinterpret_cast<char *>(malloc(total_size));
  buffer[total_size - 1] = '\0';
  for (size_t i = 0; i < result.size(); i++) {
    memcpy(buffer + offset, result[i].c_str(), result[i].size());
    offset += result[i].size();
    if (i != result.size() - 1) {
      buffer[offset++] = '|';
    }
  }
  return buffer;
}

void ClosePangu() { lake::closePangu(); }

} // extern "C"
