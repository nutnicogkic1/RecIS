#include "serialize/dtype_serialize.h"
namespace recis {
namespace serialize {
torch::Dtype DeserializeDtype(const std::string &dtype_str) {
  if (dtype_str == "BOOL") {
    return torch::Dtype::Bool;
  } else if (dtype_str == "I8") {
    return torch::CppTypeToScalarType<int8_t>::value;
  } else if (dtype_str == "U8") {
    return torch::CppTypeToScalarType<uint8_t>::value;
  } else if (dtype_str == "I16") {
    return torch::CppTypeToScalarType<int16_t>::value;
  } else if (dtype_str == "I32") {
    return torch::CppTypeToScalarType<int32_t>::value;
  } else if (dtype_str == "I64") {
    return torch::kInt64;
  } else if (dtype_str == "F16") {
    return torch::kFloat16;
  } else if (dtype_str == "F32") {
    return torch::kFloat32;
  } else if (dtype_str == "F64") {
    return torch::kFloat64;
  } else if (dtype_str == "BF16") {
    return torch::kBFloat16;
  } else {
    TORCH_CHECK(false, "dtype: ", dtype_str, " is invalid");
  }
  return torch::Dtype::Bool;
}
const char *SerializeDtype(torch::Dtype dtype) {
  switch (dtype) {
    case torch::kBool: {
      return "bool";
    }
    case torch::kInt8: {
      return "I8";
    }
    case torch::kUInt8: {
      return "U8";
    }
    case torch::kInt16: {
      return "I16";
    }
    case torch::kInt32: {
      return "I32";
    }
    case torch::kInt64: {
      return "I64";
    }
    case torch::kFloat16: {
      return "F16";
    }
    case torch::kBFloat16: {
      return "BF16";
    }
    case torch::kFloat: {
      return "F32";
    }
    case torch::kDouble: {
      return "F64";
    }
    default: {
      TORCH_CHECK(false, "Dtype: ", dtype, " not supported!");
      return "";
    }
  }
}
}  // namespace serialize
}  // namespace recis