#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#ifndef _COLUMN_IO_CC_COLUMN_IO_FRAMEWORK_THREAD_POOL_H_
#define _COLUMN_IO_CC_COLUMN_IO_FRAMEWORK_THREAD_POOL_H_
namespace column {
namespace framework {
template <typename T> class JobQueue {
public:
  T Pop() {
    std::unique_lock<std::mutex> l(mu_);
    cv_.wait(l, [this]() { return shutdown_.load() || !jobs_.empty(); });
    if (shutdown_.load()) {
      return default_element_;
    }
    auto element = jobs_.front();
    jobs_.pop_front();
    return element;
  }

  void Push(const T &element) {
    std::lock_guard<std::mutex> l(mu_);
    jobs_.push_back(element);
    cv_.notify_all();
  }

  void Shutdown() {
    std::lock_guard<std::mutex> l(shutdown_mu_);
    if (shutdown_.load()) {
      return;
    }
    shutdown_.store(true, std::memory_order_release);
    cv_.notify_all();
  }

  size_t empty() { return jobs_.empty(); }

  T PopCheckEmpty(bool &empty) {
    std::unique_lock<std::mutex> l(mu_);
    if (jobs_.empty()) {
      empty = true;
      return default_element_;
    }
    cv_.wait(l, [this]() { return shutdown_.load(); });
    if (shutdown_.load()) {
      return default_element_;
    }
    empty = jobs_.empty();
    if (!empty) {
      auto element = jobs_.front();
      jobs_.pop_front();
      return element;
    } else {
      return default_element_;
    }
  }

private:
  std::atomic<bool> shutdown_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::mutex shutdown_mu_;
  T default_element_;
  std::deque<T> jobs_;
};
class StdThread {
public:
  using LoopFn = std::function<void()>;
  StdThread(LoopFn fn);
  StdThread(StdThread &&rhv);
  ~StdThread();

private:
  std::thread thread_;
};
class StdThreadPool {
public:
  using TaskFunc = std::function<void()>;
  StdThreadPool(const std::string &name, int num_threads);
  ~StdThreadPool();

  void Schedule(std::function<void()> fn);

  void Shutdown();

  void Join();

private:
  void Loop(int i);

private:
  std::atomic<bool> shutdown_;
  std::atomic<size_t> counter_;
  std::mutex shutdown_mu_;
  std::vector<std::unique_ptr<JobQueue<TaskFunc>>> job_queues_;
  std::vector<std::unique_ptr<std::thread>> threads_;
};
} // namespace framework
} // namespace column
#endif
