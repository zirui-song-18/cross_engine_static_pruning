#ifndef __TYPES_H__
#define __TYPES_H__
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

// #define USE_FLOAT

using metadata_t = uint64_t;
using term_t = uint16_t;
using size32_t = uint32_t;
using s_size_t = size32_t;
#ifdef USE_FLOAT
using value_t = float;
using u_value_t = float;
#else
using value_t = int8_t;
using u_value_t = int8_t;
#endif

using indptr_t = uint64_t;
using value_term_t = std::pair<value_t, term_t>;
using term_value_t = std::pair<term_t, value_t>;
using value_size_t = std::pair<value_t, s_size_t>;
using size_value_t = std::pair<s_size_t, value_t>;
using rindex_t = std::unordered_map<term_t, std::vector<value_size_t>>;

#endif