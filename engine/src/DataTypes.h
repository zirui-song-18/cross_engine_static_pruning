#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <algorithm>
#include <cassert>
#include <fstream>
#include <memory>
#include <numeric>
#include <span>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "TypeConvert.h"
#include "types.h"

template <class V> struct SparseVectorElementT {
    V value;
    term_t index;
    SparseVectorElementT() : value(0), index(0) {}

    SparseVectorElementT(term_t index, V value) : value(value), index(index) {}

    bool operator==(const SparseVectorElementT &other) const {
        return (this->index == other.index && this->value == other.value);
    }

    template <class Archive> void save(Archive &ar) const {
        term_t new_index = index;
        value_t new_value = value;
        ar(new_index, new_value);
    }

    template <class Archive> void load(Archive &ar) {
        term_t new_index;
        value_t new_value;
        ar(new_index, new_value);
        index = new_index;
        value = new_value;
    }
} __attribute__((packed, aligned(1)));

using SparseVectorElement = SparseVectorElementT<value_t>;

using DenseVector = std::vector<value_t>;

template <class T> using SparseVectorT = std::vector<SparseVectorElementT<T>>;
using SparseVector = SparseVectorT<value_t>;

struct CsrMetaData {
    metadata_t n_col;
    metadata_t n_row;
    metadata_t n_value;
    CsrMetaData() : n_col(0), n_row(0), n_value(0) {}
    CsrMetaData(metadata_t n_col, metadata_t n_row, metadata_t n_value)
        : n_col(n_col), n_row(n_row), n_value(n_value) {}
};

struct SparseVectorData {
    CsrMetaData metadata;
    std::vector<SparseVector> sparse_vectors;
    SparseVectorData(CsrMetaData metadata,
                     std::vector<SparseVector> sparse_vectors)
        : metadata(metadata), sparse_vectors(std::move(sparse_vectors)) {}
};

enum class Distance : int { DP = 0, L1 = 1, L2 = 2, COSINE = 3 };

// most of parameters are better not to be exposed
struct Parameter {
    Parameter(const Parameter &params) {
        l = params.l;
        a = params.a;
        b = params.b;
        k = params.k;

        threads = params.threads;
    }
    Parameter &operator=(const Parameter &params) {
        l = params.l;
        a = params.a;
        b = params.b;
        k = params.k;
        threads = params.threads;
        return *this;
    }
    Parameter(int l, float a, int b, int k, int threads)
        : l(l), a(a), b(b), k(k), threads(threads) {}

    const std::string to_string() const {
        std::ostringstream oss;
        oss << "Parameter:\n"
            << "----------------------------------------\n"
            << "l:      " << this->l << "\n"
            << "a:      " << this->a << "\n"
            << "b:      " << this->b << "\n"
            << "k:      " << this->k << "\n"
            << "threads:" << this->threads << "\n"
            << "----------------------------------------\n";
        return oss.str();
    }

    int l;   // lambda, for static pruning
    float a; // a-mass subvector, to prune summary vector
    int b;   // beta, blocks for dynamic pruning
    int k;   // kmeans iterations
    int threads;
};

struct QueryArguments {
    int k;             // search for top k results
    int kprime;        // k'
    int cut;           // query cut
    float heap_factor; // heap factor
    int ip_budget;     // time budget to run a query in micro seconds
    int doc_limit;  // limit of documents to probe (used to prune the number of
                    // clusters)
    int num_worker; // number of worker threads
    float alpha_prune_ratio; // alpha mass subvector ratio for pruning
    float max_ratio; // keep weights > max_weight * max_ratio; <0 or >1 means
                     // no-op
    int fixed_top;   // keep top fixed_top weights; > size means no-op
    QueryArguments(int k, int kprime, int cut, float hf, int ip_budget,
                   int doc_limit, int num_worker, float alpha_prune_ratio,
                   float max_ratio, int fixed_top)
        : k(k), kprime(kprime), cut(cut), heap_factor(hf), ip_budget(ip_budget),
          doc_limit(doc_limit), num_worker(num_worker),
          alpha_prune_ratio(alpha_prune_ratio), max_ratio(max_ratio),
          fixed_top(fixed_top) {}
    QueryArguments(int k, int kprime, int cut, float hf, int ip_budget,
                   int doc_limit, int num_worker, float alpha_prune_ratio)
        : QueryArguments(k, kprime, cut, hf, ip_budget, doc_limit, num_worker,
                         alpha_prune_ratio, -1.0f, -1) {}
    QueryArguments(int k, int kprime, int cut, float hf, int ip_budget,
                   int doc_limit, int num_worker)
        : QueryArguments(k, kprime, cut, hf, ip_budget, doc_limit, num_worker,
                         0.0f, -1.0f, -1) {}
    QueryArguments(int k, int kprime, int cut, float hf, int ip_budget,
                   int doc_limit)
        : QueryArguments(k, kprime, cut, hf, ip_budget, doc_limit, 8, 0.0f,
                         -1.0f, -1) {}
    QueryArguments(int k, int kprime, int cut, float hf, int ip_budget)
        : QueryArguments(k, kprime, cut, hf, ip_budget, 0, 8, 0.0f, -1.0f, -1) {
    }
    QueryArguments(int k, int kprime, int cut, float hf)
        : QueryArguments(k, kprime, cut, hf, 0, 0, 8, 0.0f, -1.0f, -1) {}

    const std::string to_string() const {
        std::ostringstream oss;
        oss << "query parameters:\n"
            << "----------------------------------------\n"
            << "k:       " << this->k << "\n"
            << "kprime:  " << this->kprime << "\n"
            << "cut:     " << this->cut << "\n"
            << "hf:      " << this->heap_factor << "\n"
            << "budget:  " << this->ip_budget << "\n"
            << "doc_limit: " << this->doc_limit << "\n"
            << "num_worker: " << this->num_worker << "\n"
            << "alpha_prune_ratio: " << this->alpha_prune_ratio << "\n"
            << "max_ratio: " << this->max_ratio << "\n"
            << "fixed_top: " << this->fixed_top << "\n"
            << "----------------------------------------\n";
        return oss.str();
    }
};

template <typename T, typename JNI_T> class AutoCloseJniObj {
  public:
    AutoCloseJniObj(T obj, JNI_T *env) : obj(obj), env(env) {}

    T operator->() const noexcept { return obj; }
    operator T() const { return obj; }

  private:
    const T obj;
    const JNI_T *env;
};

#endif
