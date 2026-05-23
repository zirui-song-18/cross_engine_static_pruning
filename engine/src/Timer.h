#ifndef __ANN_TIMER_H__
#define __ANN_TIMER_H__
#include <chrono>
#include <iostream>
#include <string>

#include "StopTimerAggregator.h"

static const bool TIMER_DEBUG =
    (nullptr != std::getenv("TIMER_DEBUG") &&
     std::string(std::getenv("TIMER_DEBUG")) == "1");

template <class T> class DurationT {
  public:
    DurationT(int diffInTime) : diffInTime(diffInTime) {
        startTime = std::chrono::high_resolution_clock::now();
    }

    // if diffInMicro is not set, then it's always false
    bool IsTimeUp() {
        if (diffInTime <= 0) {
            return false;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<T>(end - startTime).count();
        return diff >= diffInTime;
    }

  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    int diffInTime;
};

using Duration = DurationT<std::chrono::microseconds>;

class Timer {
  public:
    Timer(const std::string &tag, int64_t count = 0) : tag(tag), count(count) {
        startTime = std::chrono::high_resolution_clock::now();
    };
    ~Timer() {
        if (!TIMER_DEBUG) {
            return;
        }
        auto milli = Get();
        if (count) {
            std::cout << tag << " time: " << milli.count()
                      << " ms, average time:" << milli.count() / count << " ms"
                      << std::endl;
        } else {
            std::cout << tag << " time: " << milli.count() << " ms"
                      << std::endl;
        }
    };

    std::chrono::duration<double, std::milli> Get() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - startTime);
    }

    void start();
    void stop();
    double getElapsedTime();

  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    std::string tag;
    int64_t count;
};

template <class T> std::string get_time_unit() {
    if (std::is_same_v<T, std::milli>)
        return "ms";
    else if (std::is_same_v<T, std::micro>)
        return "µs";
    else if (std::is_same_v<T, std::nano>)
        return "ns";
    else if (std::is_same_v<T, std::ratio<1>>)
        return "s";
    else if (std::is_same_v<T, std::ratio<60>>)
        return "min";
    else if (std::is_same_v<T, std::ratio<3600>>)
        return "h";
    return "unknown unit";
}

template <class T> class LogReporter {
  public:
    void report(const std::string &group, const std::string &tag,
                std::chrono::duration<double, T> diff) {
        std::cout << tag
                  << " time: " << std::chrono::duration<double, T>(diff).count()
                  << " " << get_time_unit<T>() << std::endl;
    }
};

template <class T> class AggregatedReporter {
  public:
    void report(const std::string &group, const std::string &tag,
                std::chrono::duration<double, T> diff) {
        stopTimerAggregator[group][tag] += diff;
    }
};

template <class T, template <class> class Reporter = LogReporter>
class StopTimerT {
    Reporter<T> reporter;

  public:
    StopTimerT() : group("default group"), reporter() {}
    StopTimerT(const std::string &group) : group(group), reporter() {}
    void Stop(const std::string &tag) {
        if (!TIMER_DEBUG) {
            return;
        }
        auto now = std::chrono::high_resolution_clock::now();
        stops.push_back({now, tag});
    };

    void Done() {
        if (!TIMER_DEBUG || stops.empty()) {
            return;
        }
        auto end = std::chrono::high_resolution_clock::now();
        stops.push_back({end, "end"});
        auto [last_time, last_tag] = stops[0];
        for (size_t i = 1; i < stops.size(); i++) {
            auto [now_time, now_tag] = stops[i];
            reporter.report(group, last_tag, now_time - last_time);
            last_time = now_time;
            last_tag = now_tag;
        }
        stops.clear();
    }

    ~StopTimerT() { Done(); };

  private:
    std::string group;
    std::vector<
        std::pair<std::chrono::time_point<std::chrono::high_resolution_clock>,
                  std::string>>
        stops;
};

#endif // __ANN_TIMER_H__
