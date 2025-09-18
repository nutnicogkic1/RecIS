#pragma once
#include <unordered_map>
#include <vector>

#include "embedding/id_map.h"
#include "torch/types.h"

namespace recis {
namespace embedding {

class CpuIdMap : public IdMap {
 public:
  CpuIdMap(torch::Device id_device);
  torch::Tensor Lookup(const torch::Tensor &ids) override;
  torch::Tensor LookupReadOnly(const torch::Tensor &ids) override;
  torch::Tensor InsertIds(const torch::Tensor &ids) override;
  torch::Tensor Ids() override;
  torch::Tensor Index() override;
  void DeleteIds(const torch::Tensor &ids, const torch::Tensor &index) override;
  std::pair<torch::Tensor, torch::Tensor> SnapShot() override;
  void Clear() override;
  void Reserve(size_t id_size) override;
  ~CpuIdMap();

 private:
  std::mutex mu_;
  ska::flat_hash_map<int64_t, int64_t> ids_map_;
};

}  // namespace embedding
}  // namespace recis
