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

#ifndef PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_LOGGING_H_
#define PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_LOGGING_H_

#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <limits>
#include <sstream>

#define __FILENAME__                                                           \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

namespace apsara {
namespace odps {
namespace algo {
namespace tf {
namespace plugins {
namespace odps {

enum LogLevel { DEBUG, INFO, WARNING, ERROR, FATAL };

class LogMessage : public std::basic_ostringstream<char> {
public:
  LogMessage(const char *fname, int line, LogLevel severity);
  ~LogMessage();

protected:
  void GenerateLogMessage();

private:
  const char *fname_;
  int line_;
  int severity_;
};

class LogMessageFatal : public LogMessage {
public:
  LogMessageFatal(const char *file, int line);
  ~LogMessageFatal();
};

#define _LOG_INFO                                                              \
  ::apsara::odps::algo::tf::plugins::odps::LogMessage(                         \
      __FILENAME__, __LINE__, ::apsara::odps::algo::tf::plugins::odps::INFO)

#define _LOG_DEBUG                                                             \
  ::apsara::odps::algo::tf::plugins::odps::LogMessage(                         \
      __FILENAME__, __LINE__, ::apsara::odps::algo::tf::plugins::odps::DEBUG)

#define _LOG_WARNING                                                           \
  ::apsara::odps::algo::tf::plugins::odps::LogMessage(                         \
      __FILENAME__, __LINE__,                                                  \
      ::apsara::odps::algo::tf::plugins::odps::WARNING)

#define _LOG_ERROR                                                             \
  ::apsara::odps::algo::tf::plugins::odps::LogMessage(                         \
      __FILENAME__, __LINE__, ::apsara::odps::algo::tf::plugins::odps::ERROR)

#define _LOG_FATAL                                                             \
  ::apsara::odps::algo::tf::plugins::odps::LogMessageFatal(__FILENAME__,       \
                                                           __LINE__)

#define LOG(severity) _LOG_##severity

} // namespace odps
} // namespace plugins
} // namespace tf
} // namespace algo
} // namespace odps
} // namespace apsara

#endif // PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_LOGGING_H_
