#ifndef RECIS_FUNCTIONAL_RAGGED_TILE_H_
#define RECIS_FUNCTIONAL_RAGGED_TILE_H_

#include <torch/extension.h>

#include <vector>
namespace recis {
namespace functional {
std::vector<torch::Tensor> ragged_tile(const std::vector<int64_t>& batch,
                                       const std::vector<int64_t>& seq,
                                       torch::Tensor value,
                                       torch::Tensor offset,
                                       torch::Tensor table);

torch::Tensor ragged_tile_back(torch::Tensor batch_seq,
                               const std::vector<int64_t>& batch_info,
                               torch::Tensor value, torch::Tensor offset,
                               torch::Tensor dy);
}  // namespace functional
}  // namespace recis
#endif  // RECIS_FUNCTIONAL_RAGGED_TILE_H_
