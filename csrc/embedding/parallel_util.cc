#include "embedding/parallel_util.h"

#include <ATen/Parallel.h>
#include <ATen/ParallelOpenMP.h>

#include <algorithm>
#include <cmath>

namespace recis {

int64_t CalculateIntraOpGranity(int64_t begin, int64_t end) {
  auto grain_size = (end - begin) / at::get_num_threads();
  grain_size = grain_size == 0 ? (end - begin) : grain_size;
  return grain_size;
}
}  // namespace recis
