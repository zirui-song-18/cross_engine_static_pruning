#ifndef __IN_MEMORY_STORAGES_H__
#define __IN_MEMORY_STORAGES_H__

#include <span>
#include <vector>

#include "DataTypes.h"
// #include "Storage.h"

using std::span;
using std::vector;

class InMemoryStorage // : public Storage
{
  public:
    InMemoryStorage(){};
    virtual ~InMemoryStorage(){};

    virtual void AddVectors(vector<SparseVector> vectors, int size, int index);
    virtual const span<const SparseVector> GetRange(int from, int end) const;
    virtual const span<const SparseVector> GetAll() const;
    virtual void UpdateVectors(const vector<SparseVector> &new_vectors);
    size_t GetMemoryUsage() const;

  private:
    vector<SparseVector> m_vectors;
};

#endif
