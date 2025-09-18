#ifndef _COLUMN_IO_CC_COLUMN_IO_DATASET_PREFETCH_H_
#define _COLUMN_IO_CC_COLUMN_IO_DATASET_PREFETCH_H_
#include "column-io/dataset/dataset.h"
#include "column-io/framework/types.h"
#include <memory>
namespace column {
namespace dataset {
class PrefetchDataset {
public:
  static std::shared_ptr<DatasetBase>
  MakeDataset(const std::shared_ptr<DatasetBase> &input, int64 prefetch_num);
};
} // namespace dataset
} // namespace column
#endif