#include "LinscanIndex.h"

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

LinscanIndex::LinscanIndex()
    : m_inverted_index(nullptr), m_window_size(DEFAULT_WINDOW_SIZE) {}

LinscanIndex::LinscanIndex(size_t window_size)
    : m_inverted_index(nullptr), m_window_size(window_size) {}

std::pair<size_t, size_t> LinscanIndex::GetMemoryUsage() const {
    size_t inverted = m_inverted_index ? m_inverted_index->GetMemoryUsage() : 0;
    size_t forward = m_storage ? m_storage->GetMemoryUsage() : 0;
    return {inverted, forward};
}

void LinscanIndex::OnLoadComplete() {
    Timer t("Time to build index after load");
    assert(m_inverted_index == nullptr);
    m_inverted_index =
        InvertedIndexWindowed::Create(m_storage->GetAll(), m_window_size);

    m_num_windows = m_inverted_index->GetNumWindows();
    std::cout << "InvertedIndexWindowed: window_size=" << m_window_size
              << ", num_windows=" << m_num_windows
              << ", total_docs=" << m_inverted_index->GetTotalDocs()
              << std::endl;
    std::cout << "InvertedIndexWindowed memory: "
              << m_inverted_index->GetMemoryUsage() / 1024.0 / 1024.0 << " MB"
              << std::endl;
}

void LinscanIndex::PartialDP(
    const std::vector<std::pair<term_t, value_t>> &query_terms,
    OrderedTopKHolder<s_size_t> &heap, int ip_budget_us,
    size_t &inverted_index_length) {

    static thread_local scores_t scores;
    if (scores.size() != m_window_size) {
        scores.resize(m_window_size);
    }
    static thread_local std::vector<size_t> active_indices;
    active_indices.clear();

    Duration duration(ip_budget_us);

    for (size_t w = 0; w < m_num_windows; ++w) {
        if (duration.IsTimeUp()) {
            break;
        }

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
}

vector<s_size_t> LinscanIndex::Query(const SparseVector &query,
                                     const QueryArguments &arguments) {
    StopTimerT<std::milli, AggregatedReporter> t("Query");
    auto query_start = std::chrono::high_resolution_clock::now();
    t.Stop("init");

    auto vectors = m_storage->GetAll();
    const int k_prime = arguments.kprime > 0 ? arguments.kprime : arguments.k;

    OrderedTopKHolder<s_size_t> heap(k_prime);

    std::vector<std::pair<term_t, value_t>> query_terms;
    query_terms.reserve(query.size());
    for (const auto &ele : query) {
        if (m_inverted_index->HasTerm(ele.index)) {
            query_terms.emplace_back(ele.index, ele.value);
        }
    }

    t.Stop("partialDP+scatter-add");
    size_t inverted_index_length = 0;
    PartialDP(query_terms, heap, arguments.ip_budget, inverted_index_length);

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
LinscanIndex::SearchWorker(int start, int end, const SparseVectorData &data,
                           const QueryArguments &arguments) {
    Timer t("LinscanIndex::SearchWorker: " + std::to_string(start) + " to " +
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
LinscanIndex::BatchSearch(const SparseVectorData &query_data,
                          const QueryArguments &arguments) {
    std::cout << "LinscanIndex: " << arguments.to_string();
    std::cout << "InvertedIndexWindowed: window_size="
              << m_inverted_index->GetWindowSize()
              << ", num_windows=" << m_inverted_index->GetNumWindows()
              << std::endl;
    Timer t("LinscanIndex::BatchSearch");
    IndexAnalyzer::GetInstance()->Reset();
    std::vector<std::vector<s_size_t>> search_results;
    search_results.reserve(query_data.metadata.n_row);

    const size_t N_WORKERS = arguments.num_worker;
    auto futures = Parallelization::Run<std::vector<std::vector<s_size_t>>>(
        N_WORKERS, (size_t)query_data.metadata.n_row,
        &LinscanIndex::SearchWorker, this, query_data, arguments);
    for (auto &future : futures) {
        auto ret = future.get();
        search_results.insert(search_results.end(), ret.begin(), ret.end());
    }
    StopTimerAggregator::instance()->report();
    IndexAnalyzer::GetInstance()->AnalyzeQuery(t.Get());

    return search_results;
}
