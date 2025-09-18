#include "serialize/register_preserved_key.h"

#include <set>
#include <string>
namespace recis {
namespace serialize {
namespace {
std::set<std::string> preserved_keys;
}

bool PreservedKeys::RegisterKeys(const std::string &key) {
  return preserved_keys.insert(key).second;
}

std::set<std::string> PreservedKeys::GetKeys() { return preserved_keys; }
}  // namespace serialize
}  // namespace recis