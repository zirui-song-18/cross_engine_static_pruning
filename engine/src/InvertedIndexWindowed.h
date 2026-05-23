#ifndef __INVERTED_INDEX_WINDOWED_H__
#define __INVERTED_INDEX_WINDOWED_H__

#include <algorithm>
#include <cmath>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataTypes.h"
#include "InvertedIndex.h"
#include "Timer.h"
#include "common.h"
#include "types.h"

/**
 * InvertedIndexWindowed - Window-aware inverted index for optimized cache
 * locality.
 *
 * Instead of storing posting lists as term -> vector<(value, doc_id)>,
 * this stores them as term -> vector<vector<(value, local_idx)>> where
 * the outer vector is indexed by window_id.
 *
 * This eliminates binary search during window-based query processing.
 */
class InvertedIndexWindowed {
  public:
    // Each window's posting list: vector of (value, local_index_within_window)
    using WindowPostingList = std::vector<value_size_t>;
    // All windows for a term: indexed by window_id
    using TermPostingLists = std::vector<WindowPostingList>;
    // Main index: term -> windows
    using WindexT = std::unordered_map<term_t, TermPostingLists>;

    InvertedIndexWindowed()
        : m_window_size(DEFAULT_WINDOW_SIZE), m_num_windows(0) {}
    explicit InvertedIndexWindowed(size_t window_size)
        : m_window_size(window_size), m_num_windows(0) {}
    ~InvertedIndexWindowed() {}

    // Create from sparse vectors directly (no pruning)
    static std::unique_ptr<InvertedIndexWindowed>
    Create(std::span<const SparseVector> vectors,
           size_t window_size = DEFAULT_WINDOW_SIZE) {
        Timer t("InvertedIndexWindowed::Create");
        std::cout << "InvertedIndexWindowed::Create, docs=" << vectors.size()
                  << ", window_size=" << window_size << std::endl;

        auto index = std::make_unique<InvertedIndexWindowed>(window_size);
        index->m_num_windows = (vectors.size() + window_size - 1) / window_size;
        index->m_total_docs = vectors.size();

        // First pass: collect all terms to pre-allocate window vectors
        std::unordered_set<term_t> all_terms;
        for (const auto &vec : vectors) {
            for (const auto &ele : vec) {
                all_terms.insert(ele.index);
            }
        }

        // Pre-allocate window vectors for each term
        for (term_t term : all_terms) {
            index->m_windex[term].resize(index->m_num_windows);
        }

        // Second pass: populate posting lists
        for (size_t doc_id = 0; doc_id < vectors.size(); ++doc_id) {
            const size_t window_id = doc_id / window_size;
            const size_t local_idx = doc_id % window_size;

            for (const auto &ele : vectors[doc_id]) {
                index->m_windex[ele.index][window_id].emplace_back(
                    ele.value, static_cast<s_size_t>(local_idx));
            }
        }

        return index;
    }

    // Create from a pruned InvertedIndex - converts and takes ownership
    static std::unique_ptr<InvertedIndexWindowed>
    CreateFromPruned(std::unique_ptr<InvertedIndex> pruned_index,
                     size_t total_docs,
                     size_t window_size = DEFAULT_WINDOW_SIZE) {
        Timer t("InvertedIndexWindowed::CreateFromPruned");

        auto index = std::make_unique<InvertedIndexWindowed>(window_size);
        index->m_num_windows = (total_docs + window_size - 1) / window_size;
        index->m_total_docs = total_docs;

        std::cout << "InvertedIndexWindowed::CreateFromPruned, total_docs="
                  << total_docs << ", window_size=" << window_size
                  << ", num_windows=" << index->m_num_windows << std::endl;

        // Get all terms from pruned index
        auto terms = pruned_index->GetAllTerms();

        // Convert each term's posting list to windowed format
        for (term_t term : terms) {
            const auto &docs = pruned_index->GetDocs(term);
            if (docs.empty())
                continue;

            // Pre-allocate windows for this term
            TermPostingLists windows(index->m_num_windows);

            // Distribute entries to windows
            for (const auto &[value, doc_id] : docs) {
                size_t w = doc_id / window_size;
                size_t local_idx = doc_id % window_size;
                windows[w].emplace_back(value,
                                        static_cast<s_size_t>(local_idx));
            }

            index->m_windex[term] = std::move(windows);
        }

        // Release the old index memory
        pruned_index.reset();

        return index;
    }

    // Get posting list for a specific term and window - O(1) access
    const WindowPostingList &GetWindowDocs(term_t term_id,
                                           size_t window_id) const {
        static const WindowPostingList empty;
        auto it = m_windex.find(term_id);
        if (it == m_windex.end() || window_id >= it->second.size()) {
            return empty;
        }
        return it->second[window_id];
    }

    // Check if term exists
    bool HasTerm(term_t term_id) const {
        return m_windex.find(term_id) != m_windex.end();
    }

    size_t GetNumWindows() const { return m_num_windows; }
    size_t GetWindowSize() const { return m_window_size; }
    size_t GetTotalDocs() const { return m_total_docs; }

    size_t GetMemoryUsage() const {
        size_t total = sizeof(*this);
        total += m_windex.bucket_count() * sizeof(void *);
        for (const auto &[term, windows] : m_windex) {
            total += sizeof(term_t) + sizeof(TermPostingLists);
            total += windows.size() * sizeof(WindowPostingList);
            for (const auto &wpl : windows) {
                total += wpl.size() * sizeof(value_size_t);
            }
        }
        return total;
    }

    const std::vector<term_t> GetAllTerms() const {
        std::vector<term_t> terms;
        terms.reserve(m_windex.size());
        for (const auto &[term, _] : m_windex) {
            terms.push_back(term);
        }
        return terms;
    }

    // Analyze posting list locality within each window
    void AnalyzeLocality() const {
        size_t total_entries = 0;
        size_t total_consecutive = 0;
        size_t total_gaps = 0;
        size_t total_windows_with_data = 0;
        size_t total_comparisons = 0;

        for (const auto &[term, windows] : m_windex) {
            for (size_t w = 0; w < windows.size(); ++w) {
                const auto &wpl = windows[w];
                if (wpl.size() < 2) {
                    total_entries += wpl.size();
                    continue;
                }
                total_entries += wpl.size();
                total_windows_with_data++;

                // Collect and sort local indices for this window
                std::vector<s_size_t> indices;
                indices.reserve(wpl.size());
                for (const auto &[val, local_idx] : wpl) {
                    indices.push_back(local_idx);
                }
                std::sort(indices.begin(), indices.end());

                for (size_t i = 1; i < indices.size(); ++i) {
                    s_size_t gap = indices[i] - indices[i - 1];
                    if (gap == 1)
                        total_consecutive++;
                    total_gaps += gap;
                    total_comparisons++;
                }
            }
        }

        if (total_comparisons > 0) {
            double avg_gap = (double)total_gaps / total_comparisons;
            double consecutive_ratio =
                (double)total_consecutive / total_comparisons;
            std::cout << "=== InvertedIndexWindowed Locality Analysis ==="
                      << std::endl;
            std::cout << "Total entries: " << total_entries << std::endl;
            std::cout << "Windows with data: " << total_windows_with_data
                      << std::endl;
            std::cout << "Avg gap (within window): " << avg_gap << std::endl;
            std::cout << "Consecutive ratio: " << consecutive_ratio
                      << std::endl;
            std::cout << "==============================================="
                      << std::endl;
        }
    }

  private:
    WindexT m_windex;
    size_t m_window_size;
    size_t m_num_windows;
    size_t m_total_docs = 0;
};

#endif
