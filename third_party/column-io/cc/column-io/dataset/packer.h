#ifndef _COMMON_IO_COLUMN_DATASET_PACKER_H_
#define _COMMON_IO_COLUMN_DATASET_PACKER_H_
#include <cstddef>
#include <functional>
#include <memory>

#include "arrow/record_batch.h"
#include "column-io/dataset/dataset.h"
#include "column-io/framework/status.h"
#include "column-io/framework/thread_pool.h"
#include "column-io/framework/types.h"
namespace column {
namespace dataset {
class Packer {
public:
  static std::shared_ptr<DatasetBase>
  MakeReorderDataset(const std::shared_ptr<DatasetBase> input,
                     const std::vector<int64> &new_order);
  static std::shared_ptr<DatasetBase>
  MakeDataset(const std::shared_ptr<DatasetBase> &input, size_t batch_size,
              bool drop_remainder, const std::vector<int> &pack_tables,
              int num_tables, const std::vector<int> &ragged_ranks,
              int64 parallel, bool pinned_result, bool gpu_result);
};
} // namespace dataset
} // namespace column
#endif
