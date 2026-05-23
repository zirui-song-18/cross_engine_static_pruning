#ifndef __DISTANCE_H__
#define __DISTANCE_H__

#if defined(HAVE_AVX512) || defined(HAVE_AVX2) || defined(HAVE_AVX)
#include <immintrin.h>
#endif

#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <numeric>

#include "DataTypes.h"
#include "TypeConvert.h"
#include "types.h"
// the smaller is returns, the closer two vectors are
using DistanceFunc =
    std::function<float(const SparseVector &, const SparseVector &)>;
// the greater is returns, the more similar two vectors are
using SimilarityFunc =
    std::function<float(const SparseVector &, const SparseVector &)>;

template <class A, class B>
float DistanceFuncT(const A &v1, const B &v2,
                    Distance distance_algo = Distance::DP) {
    switch (distance_algo) {
    case Distance::DP:
        return dot_product_distance(v1, v2);
    default:
        return 0.0f;
    }
}

#ifdef USE_FLOAT
using RET_T = float;
#else
using RET_T = int32_t;
#endif
inline RET_T dot_product(const SparseVector &v1, const SparseVector &v2) {
    RET_T result = 0;
    size_t p1 = 0, p2 = 0;
    const size_t p1_size = v1.size();
    const size_t p2_size = v2.size();
    while (p1 < p1_size && p2 < p2_size) {
        if (v1[p1].index == v2[p2].index) {
            result += RET_T(v1[p1].value) * v2[p2].value;
            ++p1;
            ++p2;
        } else if (v1[p1].index < v2[p2].index) {
            ++p1;
        } else {
            ++p2;
        }
    }
    return result;
}

template <class T, class R>
inline float dot_product_dense(const std::vector<T> &v1,
                               const std::vector<T> &v2) {
    R result = 0;
    for (size_t i = 0; i < v1.size(); ++i) {
        result += R(v1[i]) * v2[i];
    }
    return result;
}

inline float euclidean_norm(const SparseVector &v) {
    float result = 0.0f;
    for (const auto &element : v) {
        result += float(element.value) * element.value;
    }
    return std::sqrt(result);
}

inline float cosine_similarity(const SparseVector &v1, const SparseVector &v2) {
    float result = dot_product(v1, v2);
    result /= euclidean_norm(v1);
    result /= euclidean_norm(v2);
    return result;
}

inline RET_T dot_product_scalar(const SparseVector &v1, const DenseVector &v2) {
    RET_T result = 0;
    const size_t p2_size = v2.size();
    for (const auto &element : v1) {
        if (element.index >= p2_size)
            break;
        result += RET_T(element.value) * v2[element.index];
    }
    return result;
}

inline RET_T dot_product(const SparseVector &v1, const DenseVector &v2) {
    RET_T result = 0;
    const size_t v2_size = v2.size();
    const size_t v1_size = v1.size();

    // Early exit for empty vectors
    if (v1_size == 0 || v2_size == 0)
        return 0;
#pragma GCC unroll 4
    for (int i = 0; i < v1_size; ++i) {
        if (v1[i].index >= v2_size) [[unlikely]]
            break;
        result += RET_T(v1[i].value) * v2[v1[i].index];
    }
    return result;
}

inline RET_T dot_product_unroll(const SparseVector &v1, const DenseVector &v2) {
    RET_T result = 0;
    const size_t v2_size = v2.size();
    const size_t v1_size = v1.size();

    // Early exit for empty vectors
    if (v1_size == 0 || v2_size == 0)
        return 0;

    // Loop unrolling for better performance
    const size_t unroll_factor = 4;
    const size_t limit = v1_size - (v1_size % unroll_factor);
    size_t i = 0;

    // Main loop with unrolling
    for (; i < limit; i += unroll_factor) {
        if (v1[i].index >= v2_size)
            break;
        result += RET_T(v1[i].value) * v2[v1[i].index];

        if (v1[i + 1].index >= v2_size) {
            ++i;
            break;
        }
        result += RET_T(v1[i + 1].value) * v2[v1[i + 1].index];

        if (v1[i + 2].index >= v2_size) {
            i += 2;
            break;
        }
        result += RET_T(v1[i + 2].value) * v2[v1[i + 2].index];

        if (v1[i + 3].index >= v2_size) {
            i += 3;
            break;
        }
        result += RET_T(v1[i + 3].value) * v2[v1[i + 3].index];
    }

    // Handle remaining elements
    for (; i < v1_size; ++i) {
        if (v1[i].index >= v2_size)
            break;
        result += RET_T(v1[i].value) * v2[v1[i].index];
    }

    return result;
}

inline RET_T dot_product_float(const SparseVector &v1,
                               const std::vector<float> &v2) {
    RET_T result = 0;
    const size_t v2_size = v2.size();
    const size_t v1_size = v1.size();

    // Early exit for empty vectors
    if (v1_size == 0 || v2_size == 0)
        return 0;

    // Loop unrolling for better performance
    const size_t unroll_factor = 4;
    const size_t limit = v1_size - (v1_size % unroll_factor);
    size_t i = 0;

    // Main loop with unrolling
    for (; i < limit; i += unroll_factor) {
        if (v1[i].index >= v2_size)
            break;
        result += RET_T(v1[i].value) * v2[v1[i].index];

        if (v1[i + 1].index >= v2_size) {
            ++i;
            break;
        }
        result += RET_T(v1[i + 1].value) * v2[v1[i + 1].index];

        if (v1[i + 2].index >= v2_size) {
            i += 2;
            break;
        }
        result += RET_T(v1[i + 2].value) * v2[v1[i + 2].index];

        if (v1[i + 3].index >= v2_size) {
            i += 3;
            break;
        }
        result += RET_T(v1[i + 3].value) * v2[v1[i + 3].index];
    }

    // Handle remaining elements
    for (; i < v1_size; ++i) {
        if (v1[i].index >= v2_size)
            break;
        result += RET_T(v1[i].value) * v2[v1[i].index];
    }

    return result;
}

inline RET_T dot_product_scalar(const DenseVector &v1, const DenseVector &v2) {
    RET_T result = 0;
    size_t i = 0;
    const size_t v1_size = v1.size();
    const size_t v2_size = v2.size();
    for (size_t i = 0; i < v1_size && i < v2_size; ++i) {
        result += RET_T(v1[i]) * v2[i];
    }
    return result;
}

inline float dot_product_distance(const SparseVector &v1,
                                  const SparseVector &v2) {
    return 1.0f / (float(dot_product(v1, v2)) + 0.000000001);
}

inline float dot_product_distance(const SparseVector &v1,
                                  const DenseVector &v2) {
    return 1.0f / (float(dot_product(v1, v2)) + 0.000000001);
}

inline float euclidean(const SparseVector &v1, const SparseVector &v2) {
    float result = 0.0;
    size_t p1 = 0, p2 = 0;
    while (p1 < v1.size() && p2 < v2.size()) {
        if (v1[p1].index == v2[p2].index) {
            result +=
                (v1[p1].value - v2[p2].value) * (v1[p1].value - v2[p2].value);
            p1++;
            p2++;
        } else if (v1[p1].index < v2[p2].index) {
            result += v1[p1].value * v1[p1].value;
            p1++;
        } else {
            result += v2[p2].value * v2[p2].value;
            p2++;
        }
    }
    while (p1 < v1.size()) {
        result += v1[p1].value * v1[p1].value;
        p1++;
    }
    while (p2 < v2.size()) {
        result += v2[p2].value * v2[p2].value;
        p2++;
    }
    return std::sqrt(result);
}

inline float l1_distance(const SparseVector &v1, const SparseVector &v2) {
    float result = 0.0;
    size_t p1 = 0, p2 = 0;
    while (p1 < v1.size() && p2 < v2.size()) {
        if (v1[p1].index == v2[p2].index) {
            result += std::fabs(v1[p1].value - v2[p2].value);
            p1++;
            p2++;
        } else if (v1[p1].index < v2[p2].index) {
            result += std::fabs(v1[p1].value);
            p1++;
        } else {
            result += std::fabs(v2[p2].value);
            p2++;
        }
    }
    while (p1 < v1.size()) {
        result += std::fabs(v1[p1].value);
        p1++;
    }
    while (p2 < v2.size()) {
        result += std::fabs(v2[p2].value);
        p2++;
    }
    return result;
}

#endif