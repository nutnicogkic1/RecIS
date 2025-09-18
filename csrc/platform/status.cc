#include "platform/status.h"

#include "c10/util/string_view.h"
namespace recis {
Status::Status(Code code, torch::string_view msg) {
  assert(code != Code::OK);
  state_ = std::unique_ptr<State>(new State);
  state_->code = code;
  state_->msg = std::string(msg);
}

void Status::Update(const Status &new_status) {
  if (ok()) {
    *this = new_status;
  }
}

void Status::SlowCopyFrom(const State *src) {
  if (src == nullptr) {
    state_ = nullptr;
  } else {
    state_ = std::unique_ptr<State>(new State(*src));
  }
}

const std::string &Status::empty_string() {
  static std::string *empty = new std::string;
  return *empty;
}

std::string Status::ToString() const {
  if (state_ == nullptr) {
    return "OK";
  } else {
    char tmp[30];
    const char *type;
    switch (code()) {
      case Code::CANCELLED:
        type = "Cancelled";
        break;
      case Code::UNKNOWN:
        type = "Unknown";
        break;
      case Code::INVALID_ARGUMENT:
        type = "Invalid argument";
        break;
      case Code::DEADLINE_EXCEEDED:
        type = "Deadline exceeded";
        break;
      case Code::NOT_FOUND:
        type = "Not found";
        break;
      case Code::ALREADY_EXISTS:
        type = "Already exists";
        break;
      case Code::PERMISSION_DENIED:
        type = "Permission denied";
        break;
      case Code::UNAUTHENTICATED:
        type = "Unauthenticated";
        break;
      case Code::RESOURCE_EXHAUSTED:
        type = "Resource exhausted";
        break;
      case Code::FAILED_PRECONDITION:
        type = "Failed precondition";
        break;
      case Code::ABORTED:
        type = "Aborted";
        break;
      case Code::OUT_OF_RANGE:
        type = "Out of range";
        break;
      case Code::UNIMPLEMENTED:
        type = "Unimplemented";
        break;
      case Code::INTERNAL:
        type = "Internal";
        break;
      case Code::UNAVAILABLE:
        type = "Unavailable";
        break;
      case Code::DATA_LOSS:
        type = "Data loss";
        break;
      default:
        snprintf(tmp, sizeof(tmp), "Unknown code(%d)",
                 static_cast<int>(code()));
        type = tmp;
        break;
    }
    std::string result(type);
    result += ": ";
    result += state_->msg;
    return result;
  }
}

void Status::IgnoreError() const {
  // no-op
}

std::ostream &operator<<(std::ostream &os, const Status &x) {
  os << x.ToString();
  return os;
}
}  // namespace recis