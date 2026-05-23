#ifndef CSRDATA_H
#define CSRDATA_H
#include <functional>
#include <string>

#include "DataTypes.h"

typedef std::function<void(std::vector<SparseVector>, int size, int start,
                           bool last)>
    sparse_vector_processor;
typedef std::function<void(CsrMetaData &)> set_metadata;

class CsrReader {
  public:
    static CsrMetaData Read(std::string file_path, int batch_size,
                            sparse_vector_processor callback,
                            set_metadata set_meta);
};

#endif