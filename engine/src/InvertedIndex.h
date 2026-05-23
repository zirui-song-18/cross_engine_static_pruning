#ifndef __INVERTED_INDEX_H__
#define __INVERTED_INDEX_H__

#include <concepts>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataTypes.h"
#include "Timer.h"
#include "types.h"

using std::pair;
using std::span;
using std::unordered_map;
using std::vector;

/**
 * This is just a data structure to hold IV index. The class which inherits from
 * Index is IvIndex. T is value stored in iv, Func is helper function to
 * construct T from doc
 */
template <class T, class Func>
    requires std::same_as<
        std::invoke_result_t<Func, SparseVectorElement, s_size_t>, T>
class InvertedIndexT {
  public:
    using rindex_t = unordered_map<term_t, vector<T>>;
    InvertedIndexT(){};
    ~InvertedIndexT(){};

    static std::unique_ptr<InvertedIndexT<T, Func>>
    Create(span<const SparseVector> vectors) {
        Timer t("InvertedIndex::Create");
        std::cout << "InvertedIndex::Create, " << vectors.size() << std::endl;
        auto inverted_index = std::make_unique<InvertedIndexT<T, Func>>();
        for (size_t i = 0; i < vectors.size(); ++i) {
            inverted_index->AddDoc(vectors[i], i);
        }
        return std::move(inverted_index);
    }

    // clear docs to release memory
    void Clear(term_t term_id) {
        auto it = rindex.find(term_id);
        if (it == rindex.end()) {
            return;
        }
        it->second.clear();
        it->second.shrink_to_fit();
    }

    const vector<T> &GetDocs(term_t term_id) const {
        static const vector<T> empty_vector;
        auto it = rindex.find(term_id);
        if (it == rindex.end()) {
            return empty_vector;
        }
        return it->second;
    }

    const vector<term_t> GetAllTerms() const {
        vector<term_t> terms;
        terms.reserve(rindex.size());
        for (auto &p : rindex) {
            terms.push_back(p.first);
        }
        return terms;
    }

    // after Prune, reverted index are sorted by value
    void Prune(int lambda) {
        Timer t("InvertedIndex::Prune");
        if (lambda <= 0) {
            return;
        }
        for (auto &[term, docs] : rindex) {
            if (docs.size() <= lambda)
                continue;
            std::sort(docs.begin(), docs.end(),
                      [](const value_size_t &a, const value_size_t &b) {
                          return a.first > b.first;
                      });
            docs.resize(lambda);
            docs.shrink_to_fit();
        }
    }

    void AlphaPrune(float ratio) {

        if (ratio >= 1.0f || ratio <= 0.0f) {
            return;
        }

        for (auto &[term, docs] : rindex) {
            if (docs.empty())
                continue;

            float sum = 0.0f;
            for (const auto &doc : docs) {
                sum += std::fabs(doc.first);
            }

            std::sort(docs.begin(), docs.end(),
                      [](const value_size_t &a, const value_size_t &b) {
                          return std::fabs(a.first) > std::fabs(b.first);
                      });

            float count = 0.0f;
            size_t cutoff = docs.size();
            for (size_t i = 0; i < docs.size(); ++i) {
                count += std::fabs(docs[i].first);
                if (count > ratio * sum) {
                    cutoff = i;
                    break;
                }
            }

            if (cutoff < docs.size()) {
                docs.resize(cutoff);
                docs.shrink_to_fit();
            }

            std::sort(docs.begin(), docs.end(),
                      [](const value_size_t &a, const value_size_t &b) {
                          return a.second < b.second;
                      });
        }
    }

    // Keep docs with value > max_value * max_ratio in each posting list
    void MaxRatioPrune(float max_ratio) {
        if (max_ratio <= 0.0f || max_ratio >= 1.0f)
            return;

        for (auto &[term, docs] : rindex) {
            if (docs.empty())
                continue;

            auto max_val = std::fabs(docs[0].first);
            for (const auto &doc : docs) {
                if (std::fabs(doc.first) > max_val)
                    max_val = std::fabs(doc.first);
            }

            auto threshold = max_val * max_ratio;
            auto new_end = std::remove_if(
                docs.begin(), docs.end(), [threshold](const auto &d) {
                    return std::fabs(d.first) <= threshold;
                });
            docs.erase(new_end, docs.end());
            docs.shrink_to_fit();
        }
    }

    // Keep top fixed_top docs by value in each posting list
    void FixedTopPrune(int fixed_top) {
        if (fixed_top <= 0)
            return;

        for (auto &[term, docs] : rindex) {
            if (docs.size() <= static_cast<size_t>(fixed_top))
                continue;

            std::sort(docs.begin(), docs.end(),
                      [](const auto &a, const auto &b) {
                          return std::fabs(a.first) > std::fabs(b.first);
                      });
            docs.resize(fixed_top);
            docs.shrink_to_fit();
            std::sort(docs.begin(), docs.end(),
                      [](const auto &a, const auto &b) {
                          return a.second < b.second;
                      });
        }
    }

    const int GetTotalDocs() const { return total_docs; }
    void Add(const SparseVectorElement &ele, s_size_t vector_index) {
        rindex[ele.index].push_back(Func()(ele, vector_index));
    }

    size_t GetMemoryUsage() const {
        size_t total = sizeof(*this);
        total += rindex.bucket_count() * sizeof(void *);
        for (const auto &[term, docs] : rindex) {
            total += sizeof(term_t) + sizeof(vector<T>);
            total += docs.capacity() * sizeof(T);
        }
        return total;
    }

    // Remove top percent of terms with longest posting lists (high IDF = short
    // list, low IDF = long list)
    void PruneTopIdfTerms(float percent) {
        if (percent <= 0.0f || percent >= 1.0f || rindex.empty())
            return;

        std::vector<std::pair<term_t, size_t>> term_lengths;
        term_lengths.reserve(rindex.size());
        for (const auto &[term, docs] : rindex) {
            term_lengths.emplace_back(term, docs.size());
        }

        std::sort(
            term_lengths.begin(), term_lengths.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });

        size_t num_to_remove =
            static_cast<size_t>(term_lengths.size() * percent);
        for (size_t i = 0; i < num_to_remove; ++i) {
            rindex.erase(term_lengths[i].first);
        }
        std::cout << "PruneTopIdfTerms: removed " << num_to_remove << " terms"
                  << std::endl;
    }

  private:
    void AddDoc(const SparseVector &doc, s_size_t vector_index) {
        for (auto &ele : doc) {
            rindex[ele.index].push_back(Func()(ele, vector_index));
        }
    }

    rindex_t rindex;
    int total_docs;
};

struct make_value_size_t {
    value_size_t operator()(const SparseVectorElement &ele,
                            s_size_t size) const {
        return {ele.value, size};
    }
};

struct make_s_size_t {
    s_size_t operator()(const SparseVectorElement &ele, s_size_t size) const {
        return size;
    }
};

using InvertedIndex = InvertedIndexT<value_size_t, make_value_size_t>;

#endif
