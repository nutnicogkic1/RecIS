#include "c10/util/Exception.h"
#include "c10/util/string_view.h"
#include "torch/extension.h"
#pragma once
namespace recis {
enum Code {
  OK = 0,
  CANCELLED = 1,
  UNKNOWN = 2,
  INVALID_ARGUMENT = 3,
  DEADLINE_EXCEEDED = 4,
  NOT_FOUND = 5,
  ALREADY_EXISTS = 6,
  PERMISSION_DENIED = 7,
  UNAUTHENTICATED = 16,
  RESOURCE_EXHAUSTED = 8,
  FAILED_PRECONDITION = 9,
  ABORTED = 10,
  OUT_OF_RANGE = 11,
  UNIMPLEMENTED = 12,
  INTERNAL = 13,
  UNAVAILABLE = 14,
  DATA_LOSS = 15,
};
class Status {
 public:
  /// Create a success status.
  Status() {}

  /// \brief Create a status with the specified error code and msg as a
  /// human-readable string containing more detailed information.
  Status(Code code, torch::string_view msg);

  /// Copy the specified status.
  Status(const Status &s);
  void operator=(const Status &s);

  static Status OK() { return Status(); }

  /// Returns true iff the status indicates success.
  bool ok() const { return (state_ == NULL); }

  Code code() const { return ok() ? Code::OK : state_->code; }

  const std::string &error_message() const {
    return ok() ? empty_string() : state_->msg;
  }

  bool operator==(const Status &x) const;
  bool operator!=(const Status &x) const;

  /// \brief If `ok()`, stores `new_status` into `*this`.  If `!ok()`,
  /// preserves the current status, but may augment with additional
  /// information about `new_status`.
  ///
  /// Convenient way of keeping track of the first error encountered.
  /// Instead of:
  ///   `if (overall_status.ok()) overall_status = new_status`
  /// Use:
  ///   `overall_status.Update(new_status);`
  void Update(const Status &new_status);

  /// \brief Return a string representation of this status suitable for
  /// printing. Returns the string `"OK"` for success.
  std::string ToString() const;

  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  void IgnoreError() const;

 private:
  static const std::string &empty_string();
  struct State {
    Code code;
    std::string msg;
  };
  // OK status has a `NULL` state_.  Otherwise, `state_` points to
  // a `State` structure containing the error code and message(s)
  std::unique_ptr<State> state_;

  void SlowCopyFrom(const State *src);
};

inline Status::Status(const Status &s)
    : state_((s.state_ == NULL) ? NULL : new State(*s.state_)) {}

inline void Status::operator=(const Status &s) {
  // The following condition catches both aliasing (when this == &s),
  // and the common case where both s and *this are ok.
  if (state_ != s.state_) {
    SlowCopyFrom(s.state_.get());
  }
}

inline bool Status::operator==(const Status &x) const {
  return (this->state_ == x.state_) || (ToString() == x.ToString());
}

inline bool Status::operator!=(const Status &x) const { return !(*this == x); }

/// @ingroup core
std::ostream &operator<<(std::ostream &os, const Status &x);

}  // namespace recis

#define RECIS_PREDICT_FALSE(x) (x)
#define RECIS_RETURN_IF_ERROR(...)                          \
  do {                                                      \
    const ::recis::Status _status = (__VA_ARGS__);          \
    if (RECIS_PREDICT_FALSE(!_status.ok())) return _status; \
  } while (0)

#define RECIS_THROW_IF_ERROR(status)                                  \
  do {                                                                \
    if (RECIS_PREDICT_FALSE(!status.ok())) {                          \
      TORCH_CHECK(false, __FILE__, __LINE__, status.error_message()); \
    }                                                                 \
  } while (0);

#define RECIS_STATUS_COND(...)                     \
  do {                                             \
    const ::recis::Status _status = (__VA_ARGS__); \
    RECIS_THROW_IF_ERROR(_status)                  \
  } while (0);