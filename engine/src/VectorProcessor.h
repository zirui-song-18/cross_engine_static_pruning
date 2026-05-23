#ifndef __VECTOR_PROCESSOR_H__
#define __VECTOR_PROCESSOR_H__
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <unordered_map>

#include "DataTypes.h"
#include "ranker.h"

class VectorProcessor {
  public:
    inline static SparseVector Cut(const SparseVector &query, int cut) {
        if (query.size() <= cut || cut <= 0)
            return query;
        SparseVector ret = query;
        std::sort(ret.begin(), ret.end(), [](const auto &a, const auto &b) {
            return a.value > b.value;
        });
        ret.resize(cut);
        std::sort(ret.begin(), ret.end(), [](const auto &a, const auto &b) {
            return a.index < b.index;
        });
        return ret;
    }

    inline static SparseVector GetAlphaMassSubVector(const SparseVector &vector,
                                                     float alpha) {
        if (vector.empty() || alpha <= 0 || alpha >= 1.0f) {
            return vector; // no pruning
        }
        SparseVector ret = vector;
        float sum = 0.0f;
        for (auto &ele : ret) {
            sum += std::fabs(ele.value);
        }
        std::sort(ret.begin(), ret.end(), [](const auto &a, const auto &b) {
            return std::fabs(a.value) > std::fabs(b.value);
        });
        float count = 0.0f;
        for (size_t i = 0; i < ret.size(); ++i) {
            count += std::fabs(ret[i].value);
            if (count > alpha * sum) {
                ret.resize(i);
                ret.shrink_to_fit();
                break;
            }
        }
        std::sort(ret.begin(), ret.end(), [](const auto &a, const auto &b) {
            return a.index < b.index;
        });
        return ret;
    }

    // Keep weights > max_weight * max_ratio; if max_ratio < 0 or > 1, return
    // original
    inline static SparseVector GetMaxRatioSubVector(const SparseVector &vector,
                                                    float max_ratio) {
        if (vector.empty() || max_ratio < 0 || max_ratio >= 1.0f)
            return vector;
        value_t max_weight = 0;
        for (const auto &ele : vector) {
            if (std::fabs(ele.value) > max_weight)
                max_weight = std::fabs(ele.value);
        }
        value_t threshold = max_weight * max_ratio;
        SparseVector ret;
        for (const auto &ele : vector) {
            if (std::fabs(ele.value) > threshold)
                ret.push_back(ele);
        }
        return ret;
    }

    // Keep top fixed_top weights; if fixed_top >= size, return original
    inline static SparseVector GetFixedTopSubVector(const SparseVector &vector,
                                                    int fixed_top) {
        if (vector.empty() || fixed_top < 0 ||
            static_cast<size_t>(fixed_top) >= vector.size())
            return vector;
        SparseVector ret = vector;
        std::sort(ret.begin(), ret.end(), [](const auto &a, const auto &b) {
            return std::fabs(a.value) > std::fabs(b.value);
        });
        ret.resize(fixed_top);
        std::sort(ret.begin(), ret.end(), [](const auto &a, const auto &b) {
            return a.index < b.index;
        });
        return ret;
    }

    inline static DenseVector
    Sparse2DenseVector(const SparseVector &sparseVector) {
        if (sparseVector.empty())
            return DenseVector();
        DenseVector denseVector(sparseVector.back().index + 1, 0);
        for (auto &element : sparseVector) {
            denseVector[element.index] = element.value;
        }
        return denseVector;
    }

    inline static SparseVector
    Dense2SparseVector(const std::vector<value_t> &denseVector) {
        SparseVector sparseVector;

        for (term_t i = 0; i < denseVector.size(); ++i) {
            // Only include elements whose absolute value is greater than
            // epsilon
#ifdef USE_FLOAT
            if (std::abs(denseVector[i]) > 1e-6f) {
#else
            if (std::abs(denseVector[i]) > 0) {
#endif
                sparseVector.emplace_back(i,
                                          static_cast<value_t>(denseVector[i]));
            }
        }

        return sparseVector;
    }

    template <class T>
    inline static std::vector<T>
    Sparse2DenseVector(const std::vector<SparseVectorElementT<T>> &sparseVector,
                       size_t size) {
        if (sparseVector.empty())
            return vector<T>(size, 0);
        std::vector<T> denseVector(size, 0);
        for (auto &element : sparseVector) {
            if (element.index >= size) {
                std::cerr << "Sparse2DenseVector element.index: "
                          << element.index << " is larger than size:" << size
                          << std::endl;
            }
            denseVector[element.index] = element.value;
        }
        return denseVector;
    }

    inline static value_t GetValue(const SparseVector &sparseVector,
                                   const term_t term_id) {
        for (auto &element : sparseVector) {
            if (element.index == term_id) {
                return element.value;
            }
        }
        return 0;
    }

    inline static float Sum(const SparseVector &sparseVector) {
        float total = 0;
        for (auto &element : sparseVector) {
            total += element.value;
        }
        return total;
    }

    inline static const SparseVector
    SortQueryBasedOnImpactWithBudget(const SparseVector &query) {
        SparseVector sorted = query;
        std::sort(
            sorted.begin(), sorted.end(),
            [](const auto &a, const auto &b) { return a.value > b.value; });
        return sorted;
    }

    // Record pruned tokens and their posting list lengths
    static void
    RecordPrunedTokens(const SparseVector &original, const SparseVector &pruned,
                       const std::function<size_t(term_t)> &get_posting_length);
    static void SavePrunedTokensToFile(const std::string &filename);

  private:
    static std::unordered_map<term_t, size_t> pruned_token_counts_;
    static std::unordered_map<term_t, size_t> posting_list_lengths_;
};

#endif
