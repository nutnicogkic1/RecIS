#include "column-io/framework/thread_pool.h"
#include <thread>
namespace column {
namespace framework {

StdThread::StdThread(LoopFn fn) : thread_((fn)) {}
StdThread::StdThread(StdThread &&rhv) : thread_(std::move(rhv.thread_)) {}

StdThread::~StdThread() {
  if (thread_.joinable()) {
    thread_.join();
  }
}

StdThreadPool::StdThreadPool(const std::string &name, int num_threads)
    : shutdown_(false), counter_(0) {
  for (int i = 0; i < num_threads; ++i) {
    job_queues_.emplace_back(new JobQueue<std::function<void()>>());
  }
  for (int i = 0; i < num_threads; ++i) {
    threads_.emplace_back(new std::thread([this, i]() { Loop(i); }));
  }
}

StdThreadPool::~StdThreadPool() { Shutdown(); }

void StdThreadPool::Schedule(std::function<void()> fn) {
  job_queues_[++counter_ % job_queues_.size()]->Push(fn);
}

void StdThreadPool::Shutdown() {
  std::lock_guard<std::mutex> l(shutdown_mu_);
  if (shutdown_.load()) {
    return;
  }

  shutdown_.store(true, std::memory_order_release);

  for (auto &job_queue : job_queues_) {
    job_queue->Shutdown();
  }

  for (auto &thread : threads_) {
    thread->join();
  }
}

void StdThreadPool::Join() {
  for (auto &thread : threads_) {
    thread->join();
  }
}

void StdThreadPool::Loop(int i) {
  auto &job_queue = job_queues_[i % job_queues_.size()];
  while (!shutdown_.load(std::memory_order_acquire)) {
    std::function<void()> fn = job_queue->Pop();
    if (shutdown_.load()) {
      break;
    }
    fn();
  }
}
} // namespace framework
} // namespace column
