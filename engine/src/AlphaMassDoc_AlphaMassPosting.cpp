#include "AlphaMassDoc_AlphaMassPosting.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <future>
#include <string>
#include <thread>
#include <unordered_set>

#include "IndexAnalyzer.h"
#include "Parallelization.h"
#include "Serializer.h"
#include "Timer.h"
#include "VectorProcessor.h"
#include "common.h"
#include "distance.h"
#include "ranker.h"

AlphaMassDoc_AlphaMassPosting::AlphaMassDoc_AlphaMassPosting()
    : m_inverted_index(nullptr), m_window_size(DEFAULT_WINDOW_SIZE) {}

AlphaMassDoc_AlphaMassPosting::AlphaMassDoc_AlphaMassPosting(size_t window_size)
    : m_inverted_index(nullptr), m_window_size(window_size) {}

std::pair<size_t, size_t>
AlphaMassDoc_AlphaMassPosting::GetMemoryUsage() const {
    size_t inverted = m_inverted_index ? m_inverted_index->GetMemoryUsage() : 0;
    size_t forward = m_storage ? m_storage->GetMemoryUsage() : 0;
    return {inverted, forward};
}

void AlphaMassDoc_AlphaMassPosting::OnLoadComplete() {
    Timer t("Time to build index after load");
    assert(m_inverted_index == nullptr);

    auto original_vectors = m_storage->GetAll();
    const size_t total_docs = original_vectors.size();

    bool need_doc_prune =
        (m_alpha_prune_ratio > 0.0f && m_alpha_prune_ratio < 1.0f) ||
        (m_doc_max_ratio >= 0.0f && m_doc_max_ratio < 1.0f) ||
        (m_doc_fixed_top > 0);
    bool need_list_prune =
        (m_list_alpha_prune_ratio > 0.0f && m_list_alpha_prune_ratio < 1.0f) ||
        (m_list_max_ratio > 0.0f && m_list_max_ratio < 1.0f) ||
        (m_list_fixed_top > 0) ||
        (m_idf_prune_percent > 0.0f && m_idf_prune_percent < 1.0f);

    if (need_list_prune) {
        // Build regular InvertedIndex first, apply prune, then convert to
        // windowed
        std::unique_ptr<InvertedIndex> temp_index;

        if (need_doc_prune) {
            vector<SparseVector> pruned_vectors;
            pruned_vectors.reserve(total_docs);
            for (const auto &vec : original_vectors) {
                SparseVector pruned = vec;
                if (m_alpha_prune_ratio > 0.0f && m_alpha_prune_ratio < 1.0f)
                    pruned = VectorProcessor::GetAlphaMassSubVector(
                        pruned, m_alpha_prune_ratio);
                if (m_doc_max_ratio >= 0.0f && m_doc_max_ratio < 1.0f)
                    pruned = VectorProcessor::GetMaxRatioSubVector(
                        pruned, m_doc_max_ratio);
                if (m_doc_fixed_top > 0)
                    pruned = VectorProcessor::GetFixedTopSubVector(
                        pruned, m_doc_fixed_top);
                pruned_vectors.push_back(std::move(pruned));
            }
            temp_index = InvertedIndex::Create(pruned_vectors);
        } else {
            temp_index = InvertedIndex::Create(original_vectors);
        }

        // Apply posting list pruning
        if (m_list_alpha_prune_ratio > 0.0f && m_list_alpha_prune_ratio < 1.0f)
            temp_index->AlphaPrune(m_list_alpha_prune_ratio);
        if (m_list_max_ratio > 0.0f && m_list_max_ratio < 1.0f)
            temp_index->MaxRatioPrune(m_list_max_ratio);
        if (m_list_fixed_top > 0)
            temp_index->FixedTopPrune(m_list_fixed_top);
        if (m_idf_prune_percent > 0.0f && m_idf_prune_percent < 1.0f)
            temp_index->PruneTopIdfTerms(m_idf_prune_percent);

        // Convert to windowed format (releases temp_index)
        m_inverted_index = InvertedIndexWindowed::CreateFromPruned(
            std::move(temp_index), total_docs, m_window_size);
    } else {
        // No list pruning - build windowed index directly
        if (need_doc_prune) {
            vector<SparseVector> pruned_vectors;
            pruned_vectors.reserve(total_docs);
            for (const auto &vec : original_vectors) {
                SparseVector pruned = vec;
                if (m_alpha_prune_ratio > 0.0f && m_alpha_prune_ratio < 1.0f)
                    pruned = VectorProcessor::GetAlphaMassSubVector(
                        pruned, m_alpha_prune_ratio);
                if (m_doc_max_ratio >= 0.0f && m_doc_max_ratio < 1.0f)
                    pruned = VectorProcessor::GetMaxRatioSubVector(
                        pruned, m_doc_max_ratio);
                if (m_doc_fixed_top > 0)
                    pruned = VectorProcessor::GetFixedTopSubVector(
                        pruned, m_doc_fixed_top);
                pruned_vectors.push_back(std::move(pruned));
            }
            m_inverted_index =
                InvertedIndexWindowed::Create(pruned_vectors, m_window_size);
        } else {
            m_inverted_index =
                InvertedIndexWindowed::Create(original_vectors, m_window_size);
        }
    }

    m_num_windows = m_inverted_index->GetNumWindows();
    std::cout << "InvertedIndexWindowed: window_size=" << m_window_size
              << ", num_windows=" << m_num_windows
              << ", total_docs=" << m_inverted_index->GetTotalDocs()
              << std::endl;
    std::cout << "InvertedIndexWindowed memory: "
              << m_inverted_index->GetMemoryUsage() / 1024.0 / 1024.0 << " MB"
              << std::endl;
}

void AlphaMassDoc_AlphaMassPosting::AddVectorsAlphaMass(
    vector<SparseVector> vectors, int size, int index, bool last) {
    if (!m_storage) {
        m_storage = std::make_unique<InMemoryStorage>();
    }
    m_storage->AddVectors(std::move(vectors), size, index);

    if (last) {
        this->OnLoadComplete();
    }
}

void AlphaMassDoc_AlphaMassPosting::WritePerQueryLatencies(
    const std::string &path) {
    IndexAnalyzer::GetInstance()->WritePerQueryCSV(path);
}

vector<s_size_t>
AlphaMassDoc_AlphaMassPosting::Query(const SparseVector &query,
                                     const QueryArguments &arguments,
                                     int query_idx) {
    StopTimerT<std::milli, AggregatedReporter> t("Query");
    auto query_start = std::chrono::high_resolution_clock::now();

    t.Stop("init");
    auto vectors = m_storage->GetAll();
    const int k_prime = arguments.kprime > 0 ? arguments.kprime : arguments.k;

    static thread_local scores_t scores;
    if (scores.size() != m_window_size) {
        scores.resize(m_window_size);
    }
    static thread_local std::vector<size_t> active_indices;
    active_indices.clear();

    OrderedTopKHolder<s_size_t> heap(k_prime);

    std::vector<std::pair<term_t, value_t>> query_terms;
    query_terms.reserve(query.size());
    for (const auto &ele : query) {
        if (m_inverted_index->HasTerm(ele.index)) {
            query_terms.emplace_back(ele.index, ele.value);
        }
    }

    // Apply query-level pruning
    if (arguments.alpha_prune_ratio > 0.0f && arguments.alpha_prune_ratio < 1.0f) {
        // Alpha-Mass pruning: keep smallest prefix with cumulative sum >= alpha * total_sum
        value_t total_sum = 0.0f;
        for (const auto &[term_id, query_value] : query_terms) {
            total_sum += query_value;
        }

        // Sort by value descending
        std::sort(query_terms.begin(), query_terms.end(),
                  [](const auto &a, const auto &b) { return a.second > b.second; });

        value_t target_sum = arguments.alpha_prune_ratio * total_sum;
        value_t cumulative_sum = 0.0f;
        size_t keep_count = 0;
        for (size_t i = 0; i < query_terms.size(); ++i) {
            cumulative_sum += query_terms[i].second;
            keep_count = i + 1;
            if (cumulative_sum >= target_sum) {
                break;
            }
        }
        query_terms.resize(keep_count);
    } else if (arguments.max_ratio > 0.0f) {
        // Max-Ratio pruning: keep terms with value >= max_ratio * max_value
        value_t max_value = 0.0f;
        for (const auto &[term_id, query_value] : query_terms) {
            max_value = std::max(max_value, query_value);
        }

        value_t threshold = arguments.max_ratio * max_value;
        query_terms.erase(
            std::remove_if(query_terms.begin(), query_terms.end(),
                          [threshold](const auto &term) { return term.second < threshold; }),
            query_terms.end());
    } else if (arguments.fixed_top > 0 && static_cast<size_t>(arguments.fixed_top) < query_terms.size()) {
        // Fixed-Top pruning: keep only top-N terms by value
        std::partial_sort(query_terms.begin(),
                         query_terms.begin() + arguments.fixed_top,
                         query_terms.end(),
                         [](const auto &a, const auto &b) { return a.second > b.second; });
        query_terms.resize(arguments.fixed_top);
    }

    t.Stop("partialDP+scatter-add");
    size_t inverted_index_length = 0;
    for (size_t w = 0; w < m_num_windows; ++w) {
        const size_t window_start = w * m_window_size;

        for (const auto &[term_id, query_value] : query_terms) {
            const auto &docs = m_inverted_index->GetWindowDocs(term_id, w);
            for (const auto &[value, local_idx] : docs) {
                if (scores[local_idx] == 0) {
                    active_indices.push_back(local_idx);
                }
                scores[local_idx] += value * query_value;
                ++inverted_index_length;
            }
        }

        for (const size_t local_idx : active_indices) {
            if (scores[local_idx] > 0) {
                heap.Add(scores[local_idx],
                         static_cast<s_size_t>(window_start + local_idx));
            }
            scores[local_idx] = 0;
        }
        active_indices.clear();
    }

    t.Stop("re-rank");
#ifdef NO_RERANK
    // Single-stage: return top-k directly from sparse scoring (no dense re-ranking)
    auto ret = heap.TopK();
    ret.resize(arguments.k);
#else
    // Two-stage: re-rank top-k' candidates with full dense dot product
    DenseVector dense_query = VectorProcessor::Sparse2DenseVector(query);
    auto top_prime = heap.TopK();

    OrderedTopKHolder<s_size_t> final_heap(arguments.k);
    for (auto &idx : top_prime) {
        auto score = dot_product(vectors[idx], dense_query);
        final_heap.Add(score, idx);
    }
    auto ret = final_heap.TopK();
    ret.resize(arguments.k);
#endif

    t.Stop("end");
    auto query_end = std::chrono::high_resolution_clock::now();
    auto query_duration =
        std::chrono::duration<double, std::milli>(query_end - query_start);
    IndexAnalyzer::GetInstance()->RecordQuery(0, inverted_index_length);
    IndexAnalyzer::GetInstance()->RecordQueryLatency(query_duration.count());
    if (query_idx >= 0) {
        IndexAnalyzer::GetInstance()->RecordQueryLatencyAt(
            static_cast<size_t>(query_idx), query_duration.count());
    }
    t.Done();

    return ret;
}

vector<vector<s_size_t>>
AlphaMassDoc_AlphaMassPosting::SearchWorker(int start, int end,
                                            const SparseVectorData &data,
                                            const QueryArguments &arguments) {
    Timer t("AlphaMassDoc_AlphaMassPosting::SearchWorker: " +
                std::to_string(start) + " to " + std::to_string(end),
            end - start);
    std::vector<std::vector<s_size_t>> search_results;
    search_results.reserve(end - start);
    for (int i = start; i < end; ++i) {
        auto ret = Query(data.sparse_vectors[i], arguments, i);
        search_results.emplace_back(ret);
    }
    StopTimerAggregator::instance()->collect();
    return search_results;
}

vector<vector<s_size_t>>
AlphaMassDoc_AlphaMassPosting::BatchSearch(const SparseVectorData &query_data,
                                           const QueryArguments &arguments) {
    std::cout << "AlphaMassDoc_AlphaMassPosting: " << arguments.to_string();
    std::cout << "InvertedIndexWindowed: window_size="
              << m_inverted_index->GetWindowSize()
              << ", num_windows=" << m_inverted_index->GetNumWindows()
              << std::endl;
    Timer t("AlphaMassDoc_AlphaMassPosting::BatchSearch");
    IndexAnalyzer::GetInstance()->Reset();
    IndexAnalyzer::GetInstance()->SetQueryCount(query_data.metadata.n_row);
    std::vector<std::vector<s_size_t>> search_results;
    search_results.reserve(query_data.metadata.n_row);

    const size_t N_WORKERS = arguments.num_worker;
    auto futures = Parallelization::Run<std::vector<std::vector<s_size_t>>>(
        N_WORKERS, (size_t)query_data.metadata.n_row,
        &AlphaMassDoc_AlphaMassPosting::SearchWorker, this, query_data,
        arguments);
    for (auto &future : futures) {
        auto ret = future.get();
        search_results.insert(search_results.end(), ret.begin(), ret.end());
    }
    StopTimerAggregator::instance()->report();
    IndexAnalyzer::GetInstance()->AnalyzeQuery(t.Get());

    return search_results;
}
