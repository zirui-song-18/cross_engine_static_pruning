#include "AlphaMassQuery.h"

#include <algorithm>
#include <cassert>

#include "IndexAnalyzer.h"
#include "Parallelization.h"
#include "Timer.h"
#include "VectorProcessor.h"
#include "distance.h"
#include "ranker.h"

AlphaMassQuery::AlphaMassQuery()
    : m_inverted_index(nullptr), m_window_size(DEFAULT_WINDOW_SIZE) {}

AlphaMassQuery::AlphaMassQuery(size_t window_size)
    : m_inverted_index(nullptr), m_window_size(window_size) {}

std::pair<size_t, size_t> AlphaMassQuery::GetMemoryUsage() const {
    size_t inverted = m_inverted_index ? m_inverted_index->GetMemoryUsage() : 0;
    size_t forward = m_storage ? m_storage->GetMemoryUsage() : 0;
    return {inverted, forward};
}

void AlphaMassQuery::OnLoadComplete() {
    Timer t("Time to build windowed index after load");
    assert(m_inverted_index == nullptr);

    m_inverted_index =
        InvertedIndexWindowed::Create(m_storage->GetAll(), m_window_size);

    std::cout << "AlphaMassQuery: window_size=" << m_window_size
              << ", num_windows=" << m_inverted_index->GetNumWindows()
              << ", total_docs=" << m_storage->GetAll().size() << std::endl;
}

vector<s_size_t> AlphaMassQuery::Query(const SparseVector &query,
                                       const QueryArguments &arguments) {
    StopTimerT<std::milli, AggregatedReporter> t("Query");

    auto query_start = std::chrono::high_resolution_clock::now();

    t.Stop("init");
    auto vectors = m_storage->GetAll();
    const size_t total_docs = vectors.size();
    const int k_prime = arguments.kprime > 0 ? arguments.kprime : arguments.k;
    const size_t num_windows = m_inverted_index->GetNumWindows();
    const size_t window_size = m_inverted_index->GetWindowSize();

    static thread_local scores_t scores;
    if (scores.size() != window_size) {
        scores.resize(window_size);
    }

    static thread_local std::vector<size_t> active_indices;
    active_indices.clear();

    size_t inverted_index_length = 0;

    t.Stop("prune");
    SparseVector pruned_query = query;
    if (arguments.alpha_prune_ratio > 0.0f &&
        arguments.alpha_prune_ratio < 1.0f) {
        pruned_query = VectorProcessor::GetAlphaMassSubVector(
            query, arguments.alpha_prune_ratio);
    }
    if (arguments.max_ratio > 0.0f && arguments.max_ratio < 1.0f) {
        pruned_query = VectorProcessor::GetMaxRatioSubVector(
            pruned_query, arguments.max_ratio);
    }
    if (arguments.fixed_top > 0) {
        pruned_query = VectorProcessor::GetFixedTopSubVector(
            pruned_query, arguments.fixed_top);
    }

    t.Stop("partialDP+scatter-add");

    OrderedTopKHolder<s_size_t> heap(k_prime);

    // Collect valid terms from query
    std::vector<std::pair<term_t, value_t>> query_terms;
    query_terms.reserve(pruned_query.size());
    for (const auto &ele : pruned_query) {
        if (m_inverted_index->HasTerm(ele.index)) {
            query_terms.emplace_back(ele.index, ele.value);
        }
    }

    // Process each window - O(1) access per term, no binary search!
    for (size_t w = 0; w < num_windows; ++w) {
        const size_t window_start = w * window_size;

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

        // Collect results from this window
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
    DenseVector dense_query = VectorProcessor::Sparse2DenseVector(query);
    auto top_prime = heap.TopK();

    OrderedTopKHolder<s_size_t> final_heap(arguments.k);
    for (auto &idx : top_prime) {
        auto score = dot_product(vectors[idx], dense_query);
        final_heap.Add(score, idx);
    }
    auto ret = final_heap.TopK();
    ret.resize(arguments.k);

    t.Stop("end");
    IndexAnalyzer::GetInstance()->RecordQuery(0, inverted_index_length);

    auto query_end = std::chrono::high_resolution_clock::now();
    auto query_duration =
        std::chrono::duration<double, std::milli>(query_end - query_start);
    IndexAnalyzer::GetInstance()->RecordQueryLatency(query_duration.count());
    t.Done();
    return ret;
}

vector<vector<s_size_t>>
AlphaMassQuery::SearchWorker(int start, int end, const SparseVectorData &data,
                             const QueryArguments &arguments) {
    Timer t("AlphaMassQuery::SearchWorker: " + std::to_string(start) + " to " +
                std::to_string(end),
            end - start);
    std::vector<std::vector<s_size_t>> search_results;
    search_results.reserve(end - start);
    for (int i = start; i < end; ++i) {
        auto ret = Query(data.sparse_vectors[i], arguments);
        search_results.emplace_back(ret);
    }
    StopTimerAggregator::instance()->collect();
    return search_results;
}

vector<vector<s_size_t>>
AlphaMassQuery::BatchSearch(const SparseVectorData &query_data,
                            const QueryArguments &arguments) {
    std::cout << "AlphaMassQuery: " << arguments.to_string();
    std::cout << "Windowed index: window_size="
              << m_inverted_index->GetWindowSize()
              << ", num_windows=" << m_inverted_index->GetNumWindows()
              << std::endl;
    Timer t("AlphaMassQuery::BatchSearch");
    IndexAnalyzer::GetInstance()->Reset();
    std::vector<std::vector<s_size_t>> search_results;
    search_results.reserve(query_data.metadata.n_row);

    const size_t N_WORKERS = arguments.num_worker;
    auto futures = Parallelization::Run<std::vector<std::vector<s_size_t>>>(
        N_WORKERS, (size_t)query_data.metadata.n_row,
        &AlphaMassQuery::SearchWorker, this, query_data, arguments);
    for (auto &future : futures) {
        auto ret = future.get();
        search_results.insert(search_results.end(), ret.begin(), ret.end());
    }
    StopTimerAggregator::instance()->report();
    IndexAnalyzer::GetInstance()->AnalyzeQuery(t.Get());

    return search_results;
}
