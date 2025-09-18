#ifndef _COMMON_IO_CC_COLUMN_IO_DATASET_PARALLEL_DATASET_H_
#define _COMMON_IO_CC_COLUMN_IO_DATASET_PARALLEL_DATASET_H_
#include <functional>
#include <memory>
#include <string>

#include "column-io/dataset/dataset.h"
#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"
namespace column {
namespace dataset {
class ParallelDataset {
public:
  static std::shared_ptr<DatasetBase>
  MakeDataset(const std::shared_ptr<DatasetBase> input,
              std::shared_ptr<DatasetBuilder> builder, int64 cycle_length,
              int64 block_length, bool sloppy, int64 buffer_output_elements,
              int64 prefetch_input_elements);
};
} // namespace dataset
} // namespace column
#endif