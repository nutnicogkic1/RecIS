#ifndef COLUMN_IO_CC_COLUMN_IO_DATASET_REPEAT_DATASET_H_
#define COLUMN_IO_CC_COLUMN_IO_DATASET_REPEAT_DATASET_H_
#include <memory>
#include <string>

#include "column-io/dataset/dataset.h"
#include "column-io/framework/types.h"
namespace column {
namespace dataset {
class RepeatDataset {
public:
  /*
  Params:
    input: the input dataset.
    take_size: the number of batch cached taked from input dataset.
    repeat: the repeat times for cached batchs.
            If repeat == -1, it will loop infinately.
  */
  static std::shared_ptr<DatasetBase>
  MakeDataset(const std::shared_ptr<DatasetBase> &input, int64 take_size,
              int64 repeat);
};
} // namespace dataset
} // namespace column
#endif