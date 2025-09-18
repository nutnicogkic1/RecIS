#include "column-io/dataset/dataset.h"
#include <string>
namespace column {
namespace dataset {
DatasetBase::DatasetBase(const std::string &name) : name_(name) {}
} // namespace dataset
} // namespace column
