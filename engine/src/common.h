#ifndef __ANN_COMMON_H__
#define __ANN_COMMON_H__

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using scores_t = std::vector<float>;
static constexpr size_t DEFAULT_WINDOW_SIZE = 100000;

struct PostingInfo {
    const std::vector<value_size_t> *docs;
    value_t query_value;
};

static const bool IS_DEBUG = (nullptr != std::getenv("ANN_DEBUG") &&
                              std::string(std::getenv("ANN_DEBUG")) == "1");
static const bool ANALYSIS_DEBUG =
    (nullptr != std::getenv("ANALYSIS_DEBUG") &&
     std::string(std::getenv("ANALYSIS_DEBUG")) == "1");

template <typename... Args> inline void print(const Args &...args) {
    if (!ANALYSIS_DEBUG)
        return;
    (std::cout << ... << args) << std::endl;
}

#endif