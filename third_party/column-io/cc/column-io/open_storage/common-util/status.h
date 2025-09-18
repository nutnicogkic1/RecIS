#ifndef COMMON_IO_STATUS_H_
#define COMMON_IO_STATUS_H_

#include <string>

namespace apsara {
namespace odps {
namespace algo {
namespace commonio {

class Status
{
public:
    enum Code
    {
        kOk = 0,
        kCancelled = 1,
        kUnknown = 2,
        kInvalidArgument = 3,
        kDeadlineExceeded = 4,
        kNotFound = 5,
        kAlreadyExists = 6,
        kPermissionDenied = 7,
        kResourceExhausted = 8,
        kFailedPrecondition = 9,
        kAborted = 10,
        kOutOfRange = 11,
        kUnimplemented = 12,
        kInternal = 13,
        kUnavailable = 14,
        kDataLoss = 15,
        kUnauthenticated = 16,
        kWait = 17,
		kRefreshFailed = 18,
		kNoNeedRefresh = 19
    };

    Status(Status::Code code = Status::kOk, const std::string& msg = "")
      : mCode(code), mMsg(msg)
    {
    }

    void Assign(Status::Code code, const std::string& msg = "")
    {
        mCode = code;
        mMsg = msg;
    }

    static Status OK() { return Status(); }

    bool Ok() const
    {
        return mCode == kOk;
    }

    bool Wait() const
    {
        return mCode == kWait;
    }

	bool NoNeedRefresh() const
    {
        return mCode == kNoNeedRefresh;
    }

	bool RefreshFailed() const
    {
        return mCode == kRefreshFailed;
    }

    Code GetCode() const
    {
        return mCode;
    }

    std::string GetMsg() const
    {
        return mMsg;
    }

private:
    Code mCode;
    std::string mMsg;
};

} // namespace commonio
} // namespace algo
} // namespace odps
} // namespace apsara

#endif // COMMON_IO_STATUS_H_
