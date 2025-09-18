/*
 * Copyright 1999-2018 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#ifdef TF_ENABLE_ODPS_COLUMN

#include "column-io/odps/proxy/utils.h"

#include <dlfcn.h>

#include <fstream>
#include <mutex>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "absl/log/log.h"
#include "column-io/framework/status.h"
#include "rapidjson/document.h"

namespace column {
namespace odps {
namespace proxy {

namespace {

const char *kEnvForOdpsIoConfigFile = "ODPS_IO_CONFIG_FILE";
const char *kDefaultOdpsIoConfigFile = "./odps_io_config";
const char *kInputPrefix = "input_";
const char *kOutputPrefix = "output_";
long ODPS_IO_CONFIG_TIME = -1;

} // namespace

std::string GetOdpsIoConfigFile() {
  const char *env_var = getenv(kEnvForOdpsIoConfigFile);
  if (env_var == NULL) {
    return kDefaultOdpsIoConfigFile;
  }
  return std::string(env_var);
}

long GetOdpsIoConfigModifyTime() {
  FILE *fp = fopen(GetOdpsIoConfigFile().c_str(), "r");
  if (NULL != fp) {
    int fd = fileno(fp);
    struct stat buf;
    fstat(fd, &buf);
    return buf.st_mtime;
  } else {
    return 0;
  }
}

IoConfig &GetOdpsIoConfig() {
  static auto get_conf = []() -> IoConfig {
    IoConfig io_config;
    std::ifstream in(GetOdpsIoConfigFile());
    if (!in.good()) {
      LOG(ERROR) << "Open odps config failed.";
      return io_config;
    }

    std::string contents((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
    rapidjson::Document doc;
    doc.Parse(contents.c_str());
    if (!doc.IsObject()) {
      LOG(ERROR) << "Read odps config failed.";
      return io_config;
    }

    rapidjson::Value &cap = doc["capability"];
    if (!cap.IsString()) {
      LOG(ERROR) << "Read odps config failed.";
    }
    io_config.capability = cap.GetString();

    rapidjson::Value &confs = doc["conf"];
    if (!confs.IsObject()) {
      LOG(ERROR) << "Read odps config failed.";
      return io_config;
    }

    for (auto it = confs.MemberBegin(); it != confs.MemberEnd(); ++it) {
      const char *key = it->name.GetString();
      const char *value = it->value.GetString();
      io_config.confs.insert({std::string(key), std::string(value)});
    }
    return io_config;
  };
  static IoConfig conf = get_conf();
  static std::mutex mu_;
  long conf_modify_time = GetOdpsIoConfigModifyTime();
  if (conf_modify_time == ODPS_IO_CONFIG_TIME) {
    std::lock_guard<std::mutex> l(mu_);
    return conf;
  } else {
    std::lock_guard<std::mutex> l(mu_);
    conf = get_conf();
    ODPS_IO_CONFIG_TIME = conf_modify_time;
    return conf;
  }
}

bool GetConfigByName(const std::string &name, std::string *config) {
  std::string defined_name = kInputPrefix + name;
  auto &confs = GetOdpsIoConfig().confs;
  auto it = confs.find(defined_name);
  if (it != confs.end()) {
    *config = B64Decode(it->second);
    return true;
  }

  return false;
}

Status LoadLibrary(const char *library_filename, void **handle) {
  *handle = dlopen(library_filename, RTLD_NOW | RTLD_LOCAL);
  if (!*handle) {
    LOG(ERROR) << "dlopen failed: " << std::string(library_filename)
               << ", Msg: " << dlerror();
    return Status::NotFound();
  }
  return Status();
}

Status GetSymbolFromLibrary(void *handle, const char *symbol_name,
                            void **symbol) {
  *symbol = dlsym(handle, symbol_name);
  if (!*symbol) {
    LOG(ERROR) << "dlsym failed: " << std::string(symbol_name)
               << ", Msg: " << dlerror();
    return Status::NotFound();
  }
  return Status();
}

std::string RemovePrefix(const std::string &name, const std::string &prefix) {
  if (!prefix.empty()) {
    if (name.find(prefix) == 0) {
      return name.substr(prefix.size());
    } else {
      return name;
    }
  }

  // param prefix is empty, remove either "input_" or "output_" prefix
  static std::string input_prefix = std::string(kInputPrefix);
  static std::string output_prefix = std::string(kOutputPrefix);

  if (name.find(input_prefix) == 0) {
    return name.substr(input_prefix.size());
  } else if (name.find(output_prefix) == 0) {
    return name.substr(output_prefix.size());
  }

  return name;
}

std::string GetObjectIdentifier(const std::string &name) {
  // remove odps://
  if (name.length() < 7) {
    return "";
  }

  std::vector<std::string> items = absl::StrSplit(name.substr(7), "/");
  if (items.size() < 2) {
    return "";
  }
  return items[1];
}

bool CheckOdpsIoConfigFile() {
  const char *kNvWaMountConfigFile = "/apsara/nuwa/nuwa.cfg";
  const char *kNvWaConfigFile =
      "/apsara/conf/conffiles/nuwa/client/nuwa_config.json";
  auto kOdpsIoConfig = GetOdpsIoConfigFile();

  const int kTryTime = 180;
  int i = 0;
  while (i < kTryTime) {
    if (access(kNvWaMountConfigFile, F_OK) == -1 ||
        access(kNvWaConfigFile, F_OK) == -1 ||
        access(kOdpsIoConfig.c_str(), F_OK) == -1) {
      LOG(ERROR) << "Waiting Odps config file sync ready, times tried: "
                 << std::to_string(i);
      ++i;
      sleep(10);
    } else {
      break;
    }
  }

  if (i == kTryTime) {
    LOG(ERROR) << "No odps config file synced, max times has been tried!";
    return false;
  }
  return true;
}

std::string B64Decode(const std::string &in) {
  std::string out;
  if (!absl::Base64Unescape(in, &out)) {
    LOG(ERROR) << "could not base64 decode table conf";
  }
  return out;
}

} // namespace proxy
} // namespace odps
} // namespace column

#endif // TF_ENABLE_ODPS_COLUMN
