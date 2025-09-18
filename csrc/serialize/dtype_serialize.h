#pragma once
#include "torch/types.h"
namespace recis {
namespace serialize {
const char *SerializeDtype(torch::Dtype dtype);
torch::Dtype DeserializeDtype(const std::string &dtype_str);
}  // namespace serialize
}  // namespace recis