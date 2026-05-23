#ifndef __ALPHAMASSQUERY_H__
#define __ALPHAMASSQUERY_H__
#include <memory>
#include <vector>

#include "DataTypes.h"
#include "InMemoryStorage.h"
#include "Index.h"
#include "InvertedIndexWindowed.h"
#include "common.h"

class AlphaMassQuery : public Index {
  public:
    AlphaMassQuery();
    explicit AlphaMassQuery(size_t window_size);
    virtual ~AlphaMassQuery() {}
    virtual void OnLoadComplete() override;
    virtual vector<vector<s_size_t>>
    BatchSearch(const SparseVectorData &data, const QueryArguments &arguments);
    virtual std::pair<size_t, size_t> GetMemoryUsage() const override;

    void SetWindowSize(size_t window_size) { m_window_size = window_size; }
    size_t GetWindowSize() const { return m_window_size; }

  private:
    vector<vector<s_size_t>> SearchWorker(int start, int end,
                                          const SparseVectorData &data,
                                          const QueryArguments &arguments);
    vector<s_size_t> Query(const SparseVector &query,
                           const QueryArguments &arguments);

  private:
    std::unique_ptr<InvertedIndexWindowed> m_inverted_index;
    size_t m_window_size = DEFAULT_WINDOW_SIZE;
};

#endif
