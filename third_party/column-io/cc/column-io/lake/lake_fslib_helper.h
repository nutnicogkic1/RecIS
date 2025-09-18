#ifndef PAIIO_CC_IO_ALGO_FSLIB_LISTDIR_H_
#define PAIIO_CC_IO_ALGO_FSLIB_LISTDIR_H_

#include <dlfcn.h>
#include <string>
#include <vector>

namespace lake {
bool getListDir(const std::string &dir, std::vector<std::string> *result);

bool closePangu();
} // namespace lake

#endif