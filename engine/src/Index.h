#ifndef INDEX_H
#define INDEX_H

#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "CsrReader.h"
#include "DataTypes.h"
#include "InMemoryStorage.h"
#include "TypeConvert.h"
#include "types.h"

/**
 * A base class for all indices. It handles reading CSR data and query, stores
 * them in memory.
 */
class Index {
  public:
    Index() : m_storage(nullptr) {}
    virtual ~Index() {}
    /**
     * A default implementation to load sparse vector data from CSR file
     */
    void Load(std::string filename, int batch_size) {
        if (m_storage == nullptr) {
            m_storage = std::make_unique<InMemoryStorage>();
        }
        CsrMetaData meta = CsrReader::Read(
            filename, batch_size,
            [this](std::vector<SparseVector> vectors, int size, int index,
                   bool last) {
                this->AddVectors(std::move(vectors), size, index, last);
            },
            [this](CsrMetaData meta) { this->SetMetadata(meta); });
    }

    void LoadWithIndexFile(std::string filename, int batch_size,
                           std::string index_file_name) {
        if (m_storage == nullptr) {
            m_storage = std::make_unique<InMemoryStorage>();
        }
        SetIndexFile(index_file_name);
        CsrReader::Read(
            filename, batch_size,
            [this](std::vector<SparseVector> vectors, int size, int index,
                   bool last) {
                this->AddVectors(std::move(vectors), size, index, last);
            },
            [this](CsrMetaData meta) { this->SetMetadata(meta); });
    }

    /**
     * Load sparse vector data from CSR file with alpha-mass pruning.
     * Original vectors are stored, but inverted index is built from pruned
     * vectors. This allows fast PartialDP (using pruned index) and accurate
     * reranking (using original vectors).
     */
    void LoadAlphaMass(std::string filename, int batch_size,
                       float alpha_prune_ratio, float list_alpha_prune_ratio,
                       float idf_prune_percent = 0.0f,
                       float doc_max_ratio = -1.0f, int doc_fixed_top = -1,
                       float list_max_ratio = -1.0f, int list_fixed_top = -1) {
        if (m_storage == nullptr) {
            m_storage = std::make_unique<InMemoryStorage>();
        }
        m_alpha_prune_ratio = alpha_prune_ratio;
        m_list_alpha_prune_ratio = list_alpha_prune_ratio;
        m_idf_prune_percent = idf_prune_percent;
        m_doc_max_ratio = doc_max_ratio;
        m_doc_fixed_top = doc_fixed_top;
        m_list_max_ratio = list_max_ratio;
        m_list_fixed_top = list_fixed_top;
        CsrMetaData meta = CsrReader::Read(
            filename, batch_size,
            [this](std::vector<SparseVector> vectors, int size, int index,
                   bool last) {
                this->AddVectorsAlphaMass(std::move(vectors), size, index,
                                          last);
            },
            [this](CsrMetaData meta) { this->SetMetadata(meta); });
    }

    virtual vector<vector<s_size_t>>
    Search(const std::tuple<int, int> &shape,
           const std::vector<uint32_t> &indptr,
           const std::vector<uint32_t> &indices, const std::vector<float> &data,
           QueryArguments arguments) {
        int rows = std::get<0>(shape);
        int cols = std::get<1>(shape);
        if (indptr.size() < 1)
            return {};
        CsrMetaData metadata(cols, rows, data.size());
        auto vectors = vector<SparseVector>(rows);
        for (size_t i = 0; i < indptr.size() - 1; i++) {
            int start = indptr[i];
            int end = indptr[i + 1];
            vectors[i].resize(end - start);
            for (int j = start; j < end; j++) {
#ifdef USE_FLOAT
                vectors[i][j - start] =
                    SparseVectorElement(indices[j], data[j]);
#else
                vectors[i][j - start] = SparseVectorElement(
                    indices[j], mapPositiveFloatToUint(data[j]));
#endif
            }
        }
        SparseVectorData vector_data(metadata, std::move(vectors));
        return BatchSearch(vector_data, arguments);
    }
    virtual void FromFile(const std::string &file_name) {};
    virtual void ToFile(const std::string &file_name) {};
    virtual std::pair<size_t, size_t> GetMemoryUsage() const {
        return {0, m_storage ? m_storage->GetMemoryUsage() : 0};
    }

  protected:
    virtual void OnLoadComplete() {}
    virtual void AddVectors(vector<SparseVector> vectors, int size, int index,
                            bool last) {
        if (!m_storage) {
            m_storage = std::make_unique<InMemoryStorage>();
        }
        m_storage->AddVectors(std::move(vectors), size, index);
        if (last) {
            this->OnLoadComplete();
        }
    }

    virtual void AddVectorsAlphaMass(vector<SparseVector> vectors, int size,
                                     int index, bool last) {
        if (!m_storage) {
            m_storage = std::make_unique<InMemoryStorage>();
        }
        // Default implementation: just store original vectors
        // Subclasses can override to implement custom behavior
        m_storage->AddVectors(std::move(vectors), size, index);
        if (last) {
            this->OnLoadComplete();
        }
    }

    virtual void testtest() { std::cout << "testtest" << std::endl; }

    virtual vector<vector<s_size_t>>
    BatchSearch(const SparseVectorData &data,
                const QueryArguments &arguments) = 0;
    virtual void SetIndexFile(const std::string &file_name) {
        index_file_name = file_name;
    }
    virtual void SetMetadata(CsrMetaData meta) { metadata = meta; }
    CsrMetaData metadata;
    std::string index_file_name;
    std::unique_ptr<InMemoryStorage> m_storage = nullptr;

    float m_alpha_prune_ratio = 1.0f;
    float m_list_alpha_prune_ratio = 1.0f;
    float m_idf_prune_percent = 0.0f;
    float m_doc_max_ratio = -1.0f;
    int m_doc_fixed_top = -1;
    float m_list_max_ratio = -1.0f;
    int m_list_fixed_top = -1;
};

#endif
