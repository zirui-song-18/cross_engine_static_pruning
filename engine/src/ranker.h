#ifndef __RANKER_H__
#define __RANKER_H__

#include <algorithm>
#include <functional>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>

#include "types.h"

template <typename T, typename Comparator = std::greater<float>>
class TopKHolder {
    using P = std::pair<float, T>;
    int k;
    struct CompareP {
        Comparator comp;
        bool operator()(const P &p1, const P &p2) const {
            return comp(p1.first, p2.first);
        }
    };

    std::priority_queue<P, std::vector<P>, CompareP> pq;

  public:
    TopKHolder(int k) : k(k) {
        std::vector<P> vec;
        vec.reserve(k);
        pq = std::priority_queue<P, std::vector<P>, CompareP>(CompareP(),
                                                              std::move(vec));
    }

    // use a priority_queue to hold the top K items with highest scores
    void Add(const float score, const T &item) {
        if (pq.size() < k) {
            pq.push(std::make_pair(score, item));
        } else if (pq.top().first < score) {
            pq.pop();
            pq.push(std::make_pair(score, item));
        }
    }

    void Add_simple(const float score, const T &item) {
        pq.push(std::make_pair(score, item));
    }

    void Pop_simple() { pq.pop(); }

    /**
     *  get data from pq, this is a disruptive operation
     */
    std::vector<T> TopK() {
        std::vector<T> ret;
        ret.reserve(pq.size());
        while (!pq.empty()) {
            ret.push_back(pq.top().second);
            pq.pop();
        }
        return ret;
    }

    bool empty() { return pq.empty(); }

    size_t size() { return pq.size(); }

    float PeekScore() { return pq.top().first; }
};

template <typename T, typename ID_T = size_t,
          typename Comparator = std::greater<float>>
class DedupeTopKHolder {
    using P = std::pair<float, std::pair<ID_T, T>>;

  private:
    int k;
    struct CompareP {
        Comparator comp;
        bool operator()(const P &p1, const P &p2) const {
            return comp(p1.first, p2.first);
        }
    };

    std::priority_queue<P, std::vector<P>, CompareP> pq;
    std::unordered_set<ID_T> dedupe;

  public:
    DedupeTopKHolder(int k) : k(k) {
        dedupe.reserve(k);
        std::vector<P> vec;
        vec.reserve(k);
        pq = std::priority_queue<P, std::vector<P>, CompareP>(CompareP(),
                                                              std::move(vec));
    }

    // use a priority_queue to hold the top K items with highest scores
    void Add(const float score, ID_T id, const T &item) {
        if (pq.size() >= k && score <= pq.top().first) {
            return;
        }
        if (dedupe.find(id) != dedupe.end()) {
            return;
        }
        if (pq.size() < k) {
            pq.emplace(std::make_pair(score, std::make_pair(id, item)));
            dedupe.insert(id);
        } else if (pq.top().first < score) {
            auto top = pq.top();
            dedupe.erase(top.second.first);
            pq.pop();
            pq.emplace(std::make_pair(score, std::make_pair(id, item)));
            dedupe.insert(id);
        }
    }

    void Add(const float score, ID_T id) {
        if (pq.size() >= k && score <= pq.top().first) {
            return;
        }
        if (dedupe.find(id) != dedupe.end()) {
            return;
        }
        if (pq.size() < k) {
            pq.push({score, {id, id}});
            dedupe.insert(id);
        } else if (pq.top().first < score) {
            auto top = pq.top();
            dedupe.erase(top.second.first);
            pq.pop();
            pq.push({score, {id, id}});
            dedupe.insert(id);
        }
    }

    bool IsFull() { return pq.size() == k; }

    /**
     *  get data from pq, this is a disruptive operation
     */
    std::vector<T> TopK() {
        std::vector<T> ret;
        ret.reserve(pq.size());
        while (!pq.empty()) {
            ret.push_back(pq.top().second.second);
            pq.pop();
        }
        return ret;
    }

    bool empty() { return pq.empty(); }

    size_t size() { return pq.size(); }

    float PeekScore() { return pq.top().first; }
};

/**
 * OrderedTopKHolder - A variant of TopKHolder that returns results in
 * descending order by score (highest to lowest).
 *
 * This class maintains the same interface as TopKHolder but ensures that
 * TopK() returns results sorted from highest score to lowest score.
 *
 * Implementation: Uses a min-heap (like TopKHolder) but extracts elements
 * into a temporary vector and reverses it for O(k log k) sorting, which is
 * efficient for small k values typical in search scenarios.
 */
template <typename T, typename Comparator = std::greater<float>>
class OrderedTopKHolder {
    using P = std::pair<float, T>;
    int k;
    struct CompareP {
        Comparator comp;
        bool operator()(const P &p1, const P &p2) const {
            return comp(p1.first, p2.first);
        }
    };

    std::priority_queue<P, std::vector<P>, CompareP> pq;

  public:
    OrderedTopKHolder(int k) : k(k) {
        std::vector<P> vec;
        vec.reserve(k);
        pq = std::priority_queue<P, std::vector<P>, CompareP>(CompareP(),
                                                              std::move(vec));
    }

    // Add item with score, maintaining top K highest scores
    void Add(const float score, const T &item) {
        if (pq.size() < k) {
            pq.push(std::make_pair(score, item));
        } else if (pq.top().first < score) {
            pq.pop();
            pq.push(std::make_pair(score, item));
        }
    }

    void Add_simple(const float score, const T &item) {
        pq.push(std::make_pair(score, item));
    }

    void Pop_simple() { pq.pop(); }

    /**
     * Get top K items sorted by score in descending order (highest first).
     * This is a disruptive operation that empties the internal queue.
     *
     * Time complexity: O(k log k) for extraction + O(k) for reversal = O(k log
     * k) Space complexity: O(k) for the result vector
     *
     * Note: Popping from a min-heap gives elements in ascending order,
     * so we reverse to get descending order.
     */
    std::vector<T> TopK() {
        std::vector<P> temp;
        temp.reserve(pq.size());

        // Extract all elements with their scores
        // Popping from min-heap gives ascending order: smallest to largest
        while (!pq.empty()) {
            temp.push_back(pq.top());
            pq.pop();
        }

        // Reverse to get descending order (highest score first)
        std::reverse(temp.begin(), temp.end());

        // Extract just the items (without scores)
        std::vector<T> ret;
        ret.reserve(temp.size());
        for (const auto &p : temp) {
            ret.push_back(p.second);
        }

        return ret;
    }

    bool empty() { return pq.empty(); }

    size_t size() { return pq.size(); }

    float PeekScore() { return pq.top().first; }
};

/**
 * OrderedDedupeTopKHolder - A variant of DedupeTopKHolder that returns
 * results in descending order by score (highest to lowest).
 *
 * This class combines deduplication with ordered results, ensuring that:
 * 1. Each ID appears at most once in the results
 * 2. Results are sorted by score in descending order
 */
template <typename T, typename ID_T = size_t,
          typename Comparator = std::greater<float>>
class OrderedDedupeTopKHolder {
    using P = std::pair<float, std::pair<ID_T, T>>;

  private:
    int k;
    struct CompareP {
        Comparator comp;
        bool operator()(const P &p1, const P &p2) const {
            return comp(p1.first, p2.first);
        }
    };

    std::priority_queue<P, std::vector<P>, CompareP> pq;
    std::unordered_set<ID_T> dedupe;

  public:
    OrderedDedupeTopKHolder(int k) : k(k) {
        dedupe.reserve(k);
        std::vector<P> vec;
        vec.reserve(k);
        pq = std::priority_queue<P, std::vector<P>, CompareP>(CompareP(),
                                                              std::move(vec));
    }

    // Add item with score and ID, maintaining top K highest scores with
    // deduplication
    void Add(const float score, ID_T id, const T &item) {
        if (pq.size() >= k && score <= pq.top().first) {
            return;
        }
        if (dedupe.find(id) != dedupe.end()) {
            return;
        }
        if (pq.size() < k) {
            pq.emplace(std::make_pair(score, std::make_pair(id, item)));
            dedupe.insert(id);
        } else if (pq.top().first < score) {
            auto top = pq.top();
            dedupe.erase(top.second.first);
            pq.pop();
            pq.emplace(std::make_pair(score, std::make_pair(id, item)));
            dedupe.insert(id);
        }
    }

    void Add(const float score, ID_T id) {
        if (pq.size() >= k && score <= pq.top().first) {
            return;
        }
        if (dedupe.find(id) != dedupe.end()) {
            return;
        }
        if (pq.size() < k) {
            pq.push({score, {id, id}});
            dedupe.insert(id);
        } else if (pq.top().first < score) {
            auto top = pq.top();
            dedupe.erase(top.second.first);
            pq.pop();
            pq.push({score, {id, id}});
            dedupe.insert(id);
        }
    }

    bool IsFull() { return pq.size() == k; }

    /**
     * Get top K items sorted by score in descending order (highest first).
     * This is a disruptive operation that empties the internal queue.
     *
     * Note: Popping from a min-heap gives elements in ascending order,
     * so we reverse to get descending order.
     */
    std::vector<T> TopK() {
        std::vector<P> temp;
        temp.reserve(pq.size());

        // Extract all elements with their scores
        // Popping from min-heap gives ascending order
        while (!pq.empty()) {
            temp.push_back(pq.top());
            pq.pop();
        }

        // Reverse to get descending order (highest score first)
        std::reverse(temp.begin(), temp.end());

        // Extract just the items (without scores)
        std::vector<T> ret;
        ret.reserve(temp.size());
        for (const auto &p : temp) {
            ret.push_back(p.second.second);
        }

        return ret;
    }

    bool empty() { return pq.empty(); }

    size_t size() { return pq.size(); }

    float PeekScore() { return pq.top().first; }
};

#endif