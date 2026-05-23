#ifndef __ALPHAMASSDOC_ALPHAMASSPOSTING_H__
#define __ALPHAMASSDOC_ALPHAMASSPOSTING_H__
#include <memory>
#include <vector>

#include "DataTypes.h"
#include "InMemoryStorage.h"
#include "Index.h"
#include "InvertedIndexWindowed.h"
#include "common.h"

class AlphaMassDoc_AlphaMassPosting : public Index {
  public:
    AlphaMassDoc_AlphaMassPosting();
    explicit AlphaMassDoc_AlphaMassPosting(size_t window_size);
    virtual ~AlphaMassDoc_AlphaMassPosting() {}
    virtual void OnLoadComplete() override;
    virtual vector<vector<s_size_t>>
    BatchSearch(const SparseVectorData &data, const QueryArguments &arguments);
    virtual std::pair<size_t, size_t> GetMemoryUsage() const override;
    void WritePerQueryLatencies(const std::string &path);

  protected:
    virtual void AddVectorsAlphaMass(vector<SparseVector> vectors, int size,
                                     int index, bool last) override;

  private:
    vector<vector<s_size_t>> SearchWorker(int start, int end,
                                          const SparseVectorData &data,
                                          const QueryArguments &arguments);
    vector<s_size_t> Query(const SparseVector &query,
                           const QueryArguments &arguments,
                           int query_idx = -1);

  private:
    std::unique_ptr<InvertedIndexWindowed> m_inverted_index;
    size_t m_window_size = DEFAULT_WINDOW_SIZE;
    size_t m_num_windows = 0;
};

#endif
