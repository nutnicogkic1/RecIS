#include "platform/env.h"

#include <fnmatch.h>

#include "platform/posix/posix_filesystem.h"
namespace recis {
namespace {

class PosixEnv : public Env {
 public:
  PosixEnv() {}

  ~PosixEnv() override { LOG(FATAL) << "Env::Default() must not be destroyed"; }

  bool MatchPath(const std::string &path, const std::string &pattern) override {
    return fnmatch(pattern.c_str(), path.c_str(), FNM_PATHNAME) == 0;
  }

 private:
};

}  // namespace

REGISTER_FILE_SYSTEM("", PosixFileSystem);
REGISTER_FILE_SYSTEM("file", LocalPosixFileSystem);
Env *Env::Default() {
  static Env *default_env = new PosixEnv;
  return default_env;
}
}  // namespace recis