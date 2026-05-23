#ifndef __LINSCAN_INDEX_H__
#define __LINSCAN_INDEX_H__
#include <memory>
#include <vector>

#include "DataTypes.h"
#include "InMemoryStorage.h"
#include "Index.h"
#include "InvertedIndexWindowed.h"
#include "common.h"
#include "ranker.h"

class LinscanIndex : public Index {
  public:
    LinscanIndex();
    explicit LinscanIndex(size_t window_size);
    virtual ~LinscanIndex() {}
    virtual void OnLoadComplete();
    virtual vector<vector<s_size_t>>
    BatchSearch(const SparseVectorData &data, const QueryArguments &arguments);
    virtual std::pair<size_t, size_t> GetMemoryUsage() const override;

  private:
    vector<vector<s_size_t>> SearchWorker(int start, int end,
                                          const SparseVectorData &data,
                                          const QueryArguments &arguments);
    vector<s_size_t> Query(const SparseVector &query,
                           const QueryArguments &arguments);

    // Partial DP with optional time budget (ip_budget in microseconds)
    void PartialDP(const std::vector<std::pair<term_t, value_t>> &query_terms,
                   OrderedTopKHolder<s_size_t> &heap, int ip_budget_us,
                   size_t &inverted_index_length);

  private:
    std::unique_ptr<InvertedIndexWindowed> m_inverted_index;
    size_t m_window_size = DEFAULT_WINDOW_SIZE;
    size_t m_num_windows = 0;
};

#endif
