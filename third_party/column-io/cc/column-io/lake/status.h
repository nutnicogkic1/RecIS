// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// A Status encapsulates the result of an operation.  It may indicate success,
// or it may indicate an error with an associated error message.
//
// Multiple threads can invoke const methods on a Status without
// external synchronization, but if any of the threads may call a
// non-const method, all threads accessing the same Status must use
// external synchronization.

#pragma once
#include <algorithm>
#include <string>
namespace lake {
class LakeException {
public:
  LakeException(const std::string &reason) : what_(reason) {}
  const char *what() const noexcept { return what_.c_str(); }

private:
  std::string what_;
};

// TODO <ziying.dl> refine Status
class Status : public LakeException {
public:
  // Ensure its value is the same as StatusCode in
  // niagara::query::proto::StatusCode 0 ~ 999: service_contract's table::proto
  // Status code 1000 ~ 1999: hologram::coordinator's Status code
  // ...
  // TODO, add more codes to represent all Status PB used in various RPCs
  enum Code {
    // START commonly used code
    kOk = 0,
    kUnknown = 400,
    kError = 401,

    // START code for service_contract (i.e. niagara.table.proto.Status)
    kNotFound = 1,
    kCorruption = 2,
    kNotSupported = 3,
    kInvalidArgument = 4,
    kIOError = 5,
    kMergeInProgress = 6,
    kIncomplete = 7,
    kShutdownInProgress = 8,
    kTimedOut = 9,
    kAborted = 10,
    kBusy = 11,
    kExpired = 12,
    kTryAgain = 13,
    kInternalError = 14,
    kInvalidRequest = 15, // placeholder for rpc
    kDeletePending = 16,  // path delelte pending
    kReadonlyMode = 17,   // path delelte pending
    kLoopExit = 20,
    kRetry = 21,         // retry for sample lake
    kArrowInternal = 22, // for arrow internal error
    kNotImplemented = 23,
    kOutOfRange = 24, // for LakeFileReader OutOfRange

    kTabletNotExist = 101,     // aka TG_SHARD_NOT_OPENED
    kTabletAlreadyExist = 102, // aka TG_SHARD_ALREADY_OPENED
    kTabletWorking = 103,      // aka TG_SHARD_OPERATION_IN_PROGRESS
    kTabletReadOnly = 104,     // aka TG_SHARD_READ_ONLY
    kTabletFollower = 105,     // aka TG_SHARD_IN_FOLLOWER

    kSchemaInvalid = 110,
    kSchemaNotFound = 111,
    kSchemaValidationError = 112,
    kSchemaServerVersionHigher =
        113, // The client schema version is less than server verison.
    kSchemaServerVersionLower =
        114, // The client schema version is bigger than server version.

    kInvalidShardId = 201,

    kBinLogFutureTime = 251,

    kSuicide = 300,

    kReplicaNotReady = 500,
    kReplicaNotFound = 501,
    kReplicaDuplicate = 502,
    kReplicaNoData = 503,
    kReplicaClosed = 504,
    kReplicaNotLeader = 505,
    kReplicaInvalid = 506,
    // END code for service_contract (i.e. niagara.table.proto.Status)

    // START code for  hologram.coordinator.proto.Status
    kCoordinator_ShardNotReady = 1002,
    kCoordinator_PQEServiceNotReady = 1003,
    kCoordinator_SQEServiceNotReady = 1004,
    kCoordinator_QueryNotStarted = 1005,
    // END code for  hologram.coordinator.proto.Status
  };
  Code code() const { return code_; }
  Status(Status::Code code = Status::kOk, const std::string &reason = "")
      : LakeException(reason), code_(code) {}
  bool Ok() { return this->code_ == kOk; }
  // factory functions
  static Status OK() { return Status(Status::kOk, ""); }
  static Status Corruption(const std::string &reason) {
    return Status(Status::Code::kCorruption, reason);
  }
  static Status InvalidArgument(const std::string &reason) {
    return Status(Status::kInvalidArgument, reason);
  }
  static Status OutOfRange() { return Status(Status::kOutOfRange, ""); }
  static Status IOError(const std::string &reason) {
    return Status(Status::kIOError, reason);
  }
  static Status Retry(const std::string &reason) {
    return Status(Status::kRetry, reason);
  }
  static Status NotImplemented(const std::string &reason) {
    return Status(Status::kNotImplemented, reason);
  }

private:
  Code code_;
  using NoRetryType = int;
};

} // namespace lake
