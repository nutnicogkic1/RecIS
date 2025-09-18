#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>
#include <deque>
#include <utility>
#include <atomic>
#include <chrono>
#include <thread>
// #include <rapidjson/document.h>

#include "openstorage_metric_reporter.h"

int main(int argc, char** argv) {
    openstorage::MetricReporter reporter;
    reporter.UpdateReadBatch(10, 999, true);
    std::this_thread::sleep_for(std::chrono::seconds(3));
    reporter.UpdateReadBatch(3, 15, false);
    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::cout << "[INFO] report done" << std::endl;
    // reporter.Stop();
    std::this_thread::sleep_for(std::chrono::seconds(15));
    std::cout << "[INFO] report exit..." << std::endl;
    return 0;
}
