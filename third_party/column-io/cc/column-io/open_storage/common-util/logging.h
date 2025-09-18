#ifndef COMMON_IO_SWIFT_COLUMN_CLIENT_LOGGING_H_
#define COMMON_IO_SWIFT_COLUMN_CLIENT_LOGGING_H_

#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <limits>
#include <sstream>

#define __FILENAME__ \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

namespace xdl
{
namespace paiio
{
namespace third_party
{
namespace common_util
{

enum LogLevel { DEBUG, INFO, WARNING, ERROR, FATAL };

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, LogLevel severity);
  ~LogMessage();

 protected:
  void GenerateLogMessage();

 private:
  const char* fname_;
  int line_;
  int severity_;
};

class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line);
  ~LogMessageFatal();
};

#define _LOG_INFO                                              \
  xdl::paiio::third_party::common_util::LogMessage(__FILENAME__, __LINE__, \
                                       xdl::paiio::third_party::common_util::INFO)
#define _LOG_DEBUG                                             \
  xdl::paiio::third_party::common_util::LogMessage(__FILENAME__, __LINE__, \
                                       xdl::paiio::third_party::common_util::DEBUG)
#define _LOG_WARNING                                           \
  xdl::paiio::third_party::common_util::LogMessage(__FILENAME__, __LINE__, \
                                       xdl::paiio::third_party::common_util::WARNING)
#define _LOG_ERROR                                             \
  xdl::paiio::third_party::common_util::LogMessage(__FILENAME__, __LINE__, \
                                       xdl::paiio::third_party::common_util::ERROR)
#define _LOG_FATAL \
  xdl::paiio::third_party::common_util::LogMessageFatal(__FILENAME__, __LINE__)

#define LOG(severity) _LOG_##severity

} // common_util
} // third_party
} // paiio
} // xdl
#endif  // COMMON_IO_SWIFT_COLUMN_CLIENT_LOGGING_H_
