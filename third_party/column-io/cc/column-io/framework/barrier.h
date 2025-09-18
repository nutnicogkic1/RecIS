#include <atomic>
#include <condition_variable>
#include <mutex>
#ifndef _COLUMN_IO_CC_COLUMN_IO_FRAMEWORK_BARRIER_H_
#define _COLUMN_IO_CC_COLUMN_IO_FRAMEWORK_BARRIER_H_
namespace column {
namespace framework {
class Barrier {
public:
  Barrier(int size) : count_down_(size) {}
  void CountDown() {
    std::lock_guard<std::mutex> lock(mu_);
    count_down_.fetch_sub(1, std::memory_order_relaxed);
  }

  void Wait() {
    std::unique_lock<std::mutex> lock(mu_);
    cond_var_.wait(lock, [this]() -> bool {
      return count_down_.load(std::memory_order_relaxed) == 0;
    });
  }

private:
  std::mutex mu_;
  std::condition_variable cond_var_;
  std::atomic_int count_down_;
};
} // namespace framework
} // namespace column
#endif