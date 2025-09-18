#ifndef _COLUMN_IO_CC_COLUMN_IO_DATASET_LIST_DATASET_H_
#define _COLUMN_IO_CC_COLUMN_IO_DATASET_LIST_DATASET_H_
#include <column-io/dataset/dataset.h>

#include <string>

#include "column-io/framework/tensor.h"
namespace column {
namespace dataset {
class ListStringDataset {
public:
  static std::shared_ptr<DatasetBase>
  MakeDataset(const std::vector<std::string> &inputs);
};
} // namespace dataset
} // namespace column
#endif
