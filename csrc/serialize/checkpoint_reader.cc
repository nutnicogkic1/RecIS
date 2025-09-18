#include "checkpoint_reader.h"

#include <algorithm>
#include <string>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/concat.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/scalar_tensor.h"
#include "c10/core/DeviceType.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/Exception.h"
#include "c10/util/intrusive_ptr.h"
#include "serialize/block_info.h"
#include "serialize/load_bundle.h"
#include "serialize/name.h"
#include "serialize/read_block.h"

namespace recis {
namespace serialize {

CheckpointReader::CheckpointReader(const std::string &path) : path_(path) {}

void CheckpointReader::Init() {
  load_bundle_ = LoadBundle::Make(path_);
  // load_bundle_->Build();
}

std::vector<std::string> CheckpointReader::ListTensors() {
  return load_bundle_->ListTensor();
}

at::Tensor CheckpointReader::LoadTensor(const std::string &tensor_name) {
  TORCH_CHECK(load_bundle_->HasTensor(tensor_name), tensor_name, " not found");
  auto slice_infos = load_bundle_->SliceInfos(tensor_name);
  std::sort(slice_infos.begin(), slice_infos.end());
  std::vector<at::Tensor> blocks;
  for (const auto &slice_info : slice_infos) {
    auto block_info =
        load_bundle_->GetBlockInfo(BlockNameEncode(tensor_name, slice_info));
    auto tensor = at::empty(
        block_info->Shape(),
        at::TensorOptions().device(torch::kCPU).dtype(block_info->Dtype()));
    auto tensor_read_block = TensorReadBlock::Make(
        tensor, block_info,
        load_bundle_->BlockReadFile(BlockNameEncode(tensor_name, slice_info)));
    tensor_read_block->Read();
    blocks.push_back(tensor);
  }
  if (blocks.size() == 1) {
    return blocks[0];
  } else {
    return at::concat(blocks, 0);
  }
}

std::vector<int64_t> CheckpointReader::TensorShape(
    const std::string &tensor_name) {
  TORCH_CHECK(load_bundle_->HasTensor(tensor_name));
  return load_bundle_->TensorShape(tensor_name);
}

at::Tensor CheckpointReader::TensorType(const std::string &tensor_name) {
  TORCH_CHECK(load_bundle_->HasTensor(tensor_name));
  return torch::empty({}, at::TensorOptions()
                              .dtype(load_bundle_->TensorType(tensor_name))
                              .device(torch::kCPU));
}
}  // namespace serialize
}  // namespace recis
