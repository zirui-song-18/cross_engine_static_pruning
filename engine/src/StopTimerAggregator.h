#ifndef __STOP_TIMER_AGGREGATOR_H__
#define __STOP_TIMER_AGGREGATOR_H__
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

using collector_t = std::unordered_map<
    std::string,
    std::unordered_map<std::string, std::chrono::duration<double, std::milli>>>;

extern thread_local collector_t stopTimerAggregator;

class StopTimerAggregator {
  private:
    // Private constructor to prevent direct instantiation
    StopTimerAggregator() {}

    // Delete copy constructor and assignment operator
    StopTimerAggregator(const StopTimerAggregator &) = delete;
    StopTimerAggregator &operator=(const StopTimerAggregator &) = delete;
    collector_t collection;
    mutable std::mutex mutex_; // Add mutex as class member

  public:
    static std::shared_ptr<StopTimerAggregator> instance() {
        static std::shared_ptr<StopTimerAggregator> instance{
            new StopTimerAggregator()};
        return instance;
    }

    // collect from current thread
    void collect();

    // report the time proportion used by each tag in a same group
    void report();
};

#endif // __STOP_TIMER_AGGREGATOR_H__
