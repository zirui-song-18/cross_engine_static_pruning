#include "StopTimerAggregator.h"

#include <iomanip> // for std::fixed and std::setprecision
#include <iostream>

// ANSI color codes
const std::string RESET = "\033[0m";
const std::string GREEN = "\033[32m";
const std::string BLUE = "\033[34m";
const std::string YELLOW = "\033[33m";
const std::string CYAN = "\033[36m";

thread_local collector_t stopTimerAggregator;

// collect from current thread
void StopTimerAggregator::collect() {
    std::lock_guard<std::mutex> lock(mutex_); // RAII-style locking
    for (auto &[group, group_collection] : stopTimerAggregator) {
        for (auto &[tag, duration] : group_collection) {
            collection[group][tag] += duration;
        }
    }
}

// report the time proportion used by each tag in a same group
void StopTimerAggregator::report() {
    for (auto &[group, group_collection] : collection) {
        // Print group name with color
        std::cout << GREEN << "└── " << group << RESET << std::endl;

        double total = 0;
        for (auto &[tag, duration] : group_collection) {
            total += duration.count();
        }

        // Iterator to track the last element
        auto it = group_collection.begin();
        auto end = group_collection.end();

        for (; it != end; ++it) {
            const auto &[tag, duration] = *it;
            double percentage = (duration.count() / total) * 100;

            // Use different prefix for last item
            bool isLast = (std::next(it) == end);
            std::string prefix = isLast ? "    └── " : "    ├── ";

            // Print with formatting and colors
            std::cout << BLUE << prefix << RESET << YELLOW << std::left
                      << std::setw(30) << tag << RESET << CYAN
                      << " time: " << std::fixed << std::setprecision(2)
                      << duration.count() << " ms, " << std::setprecision(1)
                      << percentage << "%" << RESET << std::endl;
        }
    }
}