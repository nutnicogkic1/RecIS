#pragma once
#include <c10/util/flat_hash_map.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/script.h>

#include <atomic>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/string_view.h"
#include "embedding/children_info.h"
#include "embedding/cpu_id_map.h"
#include "embedding/id_map.h"
#include "embedding/initializer.h"
#include "embedding/slice_info.h"
#include "embedding/slot_group.h"
#include "torch/types.h"

class Hashtable;
using HashTablePtr = c10::intrusive_ptr<Hashtable>;

class Hashtable : public torch::CustomClassHolder {
 public:
  using IndexType = int64_t;
  const IndexType kNullIndex{-1};

  static c10::intrusive_ptr<Hashtable> Make(
      int64_t block_size, const std::vector<int64_t> &embedding_shape,
      torch::Dtype dtype, torch::Device device, bool coalesce,
      const std::vector<std::string> &children,
      at::intrusive_ptr<recis::embedding::Generator> generator,
      int64_t slice_begin, int64_t slice_end, int64_t slice_size);
  Hashtable(int64_t block_size, const std::vector<int64_t> &embedding_shape,
            torch::Dtype dtype, torch::Device device, bool coalesce,
            const std::vector<std::string> &children,
            at::intrusive_ptr<recis::embedding::Generator> generator,
            int64_t slice_begin, int64_t slice_end, int64_t slice_size);

  std::tuple<torch::Tensor, torch::Tensor> EmbeddingLookup(
      const torch::Tensor &ids, bool readonly);
  torch::Tensor InsertLookupIndex(const torch::Tensor &ids);
  torch::Tensor InsertLookupIndexWithIndicator(const torch::Tensor &ids,
                                               const torch::Tensor &indicator);

  void Insert(const torch::Tensor &ids, const torch::Tensor &embedding);
  void UpdateSlot(const std::string &slot_name, const torch::Tensor &ids,
                  torch::Tensor embedding);
  void AppendStepFilterSlot(const std::string &filter_name, int64_t init_step);
  void AppendFilterSlot(
      const std::string &filter_name, torch::Dtype dtype, int64_t slot_dim,
      at::intrusive_ptr<recis::embedding::Generator> generator);

  void Clear();
  void ClearId();
  int64_t ids_num() { return id_map_->GetIdNum(); }
  torch::Tensor ids();
  void increment_blocknum(int64_t ids_num);
  void Reserve(size_t id_size);
  void Delete(const torch::Tensor &ids, const torch::Tensor &index,
              const std::string &preserve_slot);

  // for save/load
  std::tuple<torch::Tensor, torch::Tensor> ids_map() {
    auto id_index = id_map_->SnapShot();
    return std::make_tuple(std::move(id_index.first),
                           std::move(id_index.second));
  }

  static int64_t EncodeIdWithMask(int64_t id, int64_t mask_id);
  static std::pair<int64_t, int64_t> DecodeIdToIndexAndMask(int64_t id);

  // for opt
  bool has_grad() const { return grad_.size() > 0; };
  void AcceptGrad(const torch::Tensor &grad_index, const torch::Tensor &grad);
  torch::Tensor grad(int64_t accmulate_steps);
  void ClearGrad();

  const at::intrusive_ptr<recis::embedding::SliceInfo> SliceInfo();
  at::intrusive_ptr<recis::embedding::SlotGroup> SlotGroup();
  at::intrusive_ptr<recis::embedding::ChildrenInfo> ChildrenInfo();
  at::intrusive_ptr<recis::embedding::Slot> GetSlot(const std::string &name);

 private:
  std::tuple<torch::Tensor, torch::Tensor> EmbeddingLookupReadOnly(
      const torch::Tensor &ids);
  torch::Tensor LookupIndexReadOnly(const torch::Tensor &ids);

  // for id generate
  at::intrusive_ptr<recis::embedding::IdMap> id_map_;
  at::intrusive_ptr<recis::embedding::SliceInfo> slice_info_;
  // for coalesce
  at::intrusive_ptr<recis::embedding::ChildrenInfo> children_info_;
  // for param slot
  at::intrusive_ptr<recis::embedding::SlotGroup> slot_group_;

  // for opt
  std::vector<torch::Tensor> grad_index_;
  std::vector<torch::Tensor> grad_;
};
