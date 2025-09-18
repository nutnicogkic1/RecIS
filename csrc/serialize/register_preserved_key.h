#pragma once
#include <set>
#include <string>
namespace recis {
namespace serialize {
class PreservedKeys {
 public:
  static bool RegisterKeys(const std::string &);
  std::set<std::string> GetKeys();
};

}  // namespace serialize
}  // namespace recis

#define RECIS_SERIALIZE_CONCATENATE_DETAIL(x, y) x##y
#define RECIS_SERIALIZE_CONCATENATE(x, y) \
  RECIS_SERIALIZE_CONCATENATE_DETAIL(x, y)

#define RECIS_SERIALIZE_UNIQUE_NAME(base) \
  RECIS_SERIALIZE_CONCATENATE(base, __COUNTER__)

#define RECIS_SERIALIZE_REGISTER_PRESERVED_KEY(KEY_NAME)          \
  static bool RECIS_SERIALIZE_UNIQUE_NAME(xxx) [[maybe_unused]] = \
      ::recis::serialize::PreservedKeys::RegisterKeys(KEY_NAME);