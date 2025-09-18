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

#ifndef PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_EXCEPTION_TO_RET_H_
#define PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_EXCEPTION_TO_RET_H_

#include "algo/include/common/exception.h"
#include "column-io/odps/plugins/logging.h"

namespace apsara {
namespace odps {
namespace algo {
namespace tf {
namespace plugins {
namespace odps {

#define EXCEPTION_TO_RET(s)                                                    \
  catch (const apsara::odps::algo::NoPermissionException &ex) {                \
    LOG(INFO) << "catch apsara::odps::algo::NoPermissionException: "           \
              << ex.GetMessage();                                              \
    s = false;                                                                 \
  }                                                                            \
  catch (const apsara::odps::algo::NotFoundException &ex) {                    \
    LOG(INFO) << "catch apsara::odps::algo::NotFoundException: "               \
              << ex.GetMessage();                                              \
    s = false;                                                                 \
  }                                                                            \
  catch (const apsara::odps::algo::AlreadyExistException &ex) {                \
    LOG(INFO) << "catch apsara::odps::algo::AlreadyExistException: "           \
              << ex.GetMessage();                                              \
    s = false;                                                                 \
  }                                                                            \
  catch (const apsara::odps::algo::InvalidArgumentException &ex) {             \
    LOG(INFO) << "catch apsara::odps::algo::InvalidArgumentException: "        \
              << ex.GetMessage();                                              \
    s = false;                                                                 \
  }                                                                            \
  catch (const apsara::odps::algo::TimeoutException &ex) {                     \
    LOG(INFO) << "catch apsara::odps::algo::TimeoutException: "                \
              << ex.GetMessage();                                              \
    s = false;                                                                 \
  }                                                                            \
  catch (const apsara::odps::algo::OutOfResourceException &ex) {               \
    LOG(INFO) << "catch apsara::odps::algo::OutOfResourceException: "          \
              << ex.GetMessage();                                              \
    s = false;                                                                 \
  }                                                                            \
  catch (const apsara::odps::algo::SystemException &ex) {                      \
    LOG(INFO) << "catch apsara::odps::algo::SystemException: "                 \
              << ex.GetMessage();                                              \
    s = false;                                                                 \
  }                                                                            \
  catch (const apsara::odps::algo::UserException &ex) {                        \
    LOG(INFO) << "catch apsara::odps::algo::UserException: "                   \
              << ex.GetMessage();                                              \
    s = false;                                                                 \
  }                                                                            \
  catch (const apsara::odps::algo::BaseException &ex) {                        \
    LOG(INFO) << "catch apsara::odps::algo::BaseException: "                   \
              << ex.GetMessage();                                              \
    s = false;                                                                 \
  }                                                                            \
  catch (const std::exception &ex) {                                           \
    LOG(INFO) << "catch std::exception: " << ex.what();                        \
    s = false;                                                                 \
  }                                                                            \
  catch (...) {                                                                \
    LOG(INFO) << "catch unknown error";                                        \
    s = false;                                                                 \
  }

#define EXIT_IF_EXCEPTION(STMTS)                                               \
  try {                                                                        \
    STMTS;                                                                     \
  } catch (const apsara::odps::algo::NotFoundException &ex) {                  \
    LOG(INFO) << "catch apsara::odps::algo::NotFoundException: "               \
              << ex.GetMessage();                                              \
    _exit(12);                                                                 \
  } catch (const std::exception &ex) {                                         \
    LOG(FATAL) << "odps exception: " << ex.what();                             \
  } catch (...) {                                                              \
    LOG(FATAL) << "odps unknown error";                                        \
  }

} // namespace odps
} // namespace plugins
} // namespace tf
} // namespace algo
} // namespace odps
} // namespace apsara

#endif // PAIIO_CC_IO_ALGO_COLUMN_PLUGINS_ODPS_EXCEPTION_TO_RET_H_
