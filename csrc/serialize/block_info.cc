#include "serialize/block_info.h"

#include <string>
#include <tuple>
#include <utility>

#include "c10/util/Exception.h"
#include "c10/util/StringUtil.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/string_view.h"
#include "embedding/hashtable.h"
#include "nlohmann/json_fwd.hpp"
#include "serialize/dtype_serialize.h"
#include "serialize/name.h"
#include "serialize/slice_info_serialize.h"
#include "serialize/string_sep.h"
#include "torch/types.h"
#include "utils/str_util.h"
namespace recis {
namespace serialize {
at::intrusive_ptr<BlockInfo> BlockInfo::Make() {
  return at::make_intrusive<BlockInfo>();
}

int64_t BlockInfo::Size() { return OffsetEnd_ - OffsetBeg_; }

void BlockInfo::DecodeFromJson(nlohmann::json::object_t &json) {
  TORCH_CHECK(json.count(BlockKeyDtype()) != 0,
              "BlockInfo::DecodeFromJson dtype failed, json: ", json);
  Dtype_ = DeserializeDtype(json[BlockKeyDtype()].get<std::string>());
  TORCH_CHECK(json.count(BlockKeyShape()) != 0,
              "BlockInfo::DecodeFromJson shape failed, json: ", json);
  nlohmann::json::array_t shape_array =
      json[BlockKeyShape()].get<nlohmann::json::array_t>();
  for (auto &dim : shape_array) {
    Shape_.push_back(dim);
  }
  TORCH_CHECK(json.count(BlockKeyOffsets()) != 0,
              "BlockInfo::DecodeFromJson offset failed, json: ", json);
  nlohmann::json::array_t offsets_array =
      json[BlockKeyOffsets()].get<nlohmann::json::array_t>();
  TORCH_CHECK(
      offsets_array.size() == 2,
      "BlockInfo::DecodeFromJson offset shape must be 2d, json: ", json);
  OffsetBeg_ = offsets_array.at(0);
  OffsetEnd_ = offsets_array.at(1);
}
}  // namespace serialize
}  // namespace recis