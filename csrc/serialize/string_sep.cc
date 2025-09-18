#include "serialize/string_sep.h"

#include "serialize/register_preserved_key.h"
namespace recis {
namespace serialize {
const char StrSep::kInterNameSep{'^'};
const char StrSep::kIntraNameSep{'@'};

RECIS_SERIALIZE_REGISTER_PRESERVED_KEY("@");
RECIS_SERIALIZE_REGISTER_PRESERVED_KEY("^");
}  // namespace serialize
}  // namespace recis