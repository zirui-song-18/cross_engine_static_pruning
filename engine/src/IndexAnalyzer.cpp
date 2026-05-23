#include "IndexAnalyzer.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>

#include "Cluster.h"
#include "common.h"

void IndexAnalyzer::AnalyzeInvertedIndex(InvertedIndex *invertedIndex) {
    if (!ANALYSIS_DEBUG)
        return;
    auto terms = invertedIndex->GetAllTerms();
    std::cout << "Inverted index has terms:" << terms.size() << std::endl;
    size_t sum = 0;
    size_t max_size = 0;
    size_t min_size = 1000000;
    for (auto term : terms) {
        size_t doc_size = invertedIndex->GetDocs(term).size();
        sum += doc_size;
        max_size = std::max(max_size, doc_size);
        min_size = std::min(min_size, doc_size);
    }
    std::cout << "Average docs per term:" << sum / terms.size()
              << ", max doc size:" << max_size << ", min doc size:" << min_size
              << std::endl;
}

void IndexAnalyzer::RecordQuery(uint64_t _touched_indices,
                                uint64_t _inverted_index_length) {
    if (!ANALYSIS_DEBUG)
        return;
    ++counter;
    cluster_examined += _touched_indices;
    dp += _inverted_index_length;
    touched_indices_sum += _touched_indices;
    inverted_index_length_sum += _inverted_index_length;
}

void IndexAnalyzer::RecordQueryLatency(double latency_ms) {
    if (!ANALYSIS_DEBUG)
        return;
    std::lock_guard<std::mutex> lock(latency_mutex);
    query_latencies.push_back(latency_ms);
}

void IndexAnalyzer::SetQueryCount(size_t n) {
    per_query_latencies.assign(n, 0.0);
}

void IndexAnalyzer::RecordQueryLatencyAt(size_t query_idx, double latency_ms) {
    // Thread-safe: each query_idx is written by exactly one thread
    if (query_idx < per_query_latencies.size()) {
        per_query_latencies[query_idx] = latency_ms;
    }
}

void IndexAnalyzer::WritePerQueryCSV(const std::string &path) {
    std::ofstream f(path);
    f << "query_idx,latency_ms\n";
    for (size_t i = 0; i < per_query_latencies.size(); i++) {
        f << i << "," << per_query_latencies[i] << "\n";
    }
    std::cout << "Wrote " << per_query_latencies.size()
              << " per-query latencies to " << path << std::endl;
}

void IndexAnalyzer::Reset() {
    if (!ANALYSIS_DEBUG)
        return;
    counter = 0;
    cluster_examined = 0;
    dp = 0;
    touched_indices_sum = 0;
    inverted_index_length_sum = 0;

    // Clear per-query latency data
    std::lock_guard<std::mutex> lock(latency_mutex);
    query_latencies.clear();
}

void IndexAnalyzer::AnalyzeQuery(
    std::chrono::duration<double, std::milli> batch_time) {
    if (counter == 0 && query_latencies.empty())
        return;

    // Calculate statistics from per-query latencies if available
    if (!query_latencies.empty()) {
        double sum_latency = std::accumulate(query_latencies.begin(),
                                             query_latencies.end(), 0.0);
        double avg_latency = sum_latency / query_latencies.size();

        // Calculate min and max
        double min_latency =
            *std::min_element(query_latencies.begin(), query_latencies.end());
        double max_latency =
            *std::max_element(query_latencies.begin(), query_latencies.end());

        // Calculate median
        std::vector<double> sorted_latencies = query_latencies;
        std::sort(sorted_latencies.begin(), sorted_latencies.end());
        double median_latency = sorted_latencies[sorted_latencies.size() / 2];

        // Calculate p95 and p99
        size_t p95_idx = static_cast<size_t>(sorted_latencies.size() * 0.95);
        size_t p99_idx = static_cast<size_t>(sorted_latencies.size() * 0.99);
        double p95_latency =
            sorted_latencies[std::min(p95_idx, sorted_latencies.size() - 1)];
        double p99_latency =
            sorted_latencies[std::min(p99_idx, sorted_latencies.size() - 1)];

        std::cout << "Query: " << query_latencies.size();

        if (counter > 0) {
            std::cout << ", average touched_indices: "
                      << touched_indices_sum / counter
                      << ", average inverted_index_length: "
                      << inverted_index_length_sum / counter;
        }

        std::cout << " QPS:"
                  << (query_latencies.size() /
                      std::chrono::duration_cast<std::chrono::duration<double>>(
                          batch_time)
                          .count())
                  << ", average latency per query:" << avg_latency << " ms"
                  << ", min:" << min_latency << " ms" << ", max:" << max_latency
                  << " ms" << ", median:" << median_latency << " ms"
                  << ", p95:" << p95_latency << " ms" << ", p99:" << p99_latency
                  << " ms" << std::endl;
    } else if (counter > 0) {
        // Fallback to old method if no per-query latencies recorded
        std::cout << "Query: " << counter << ", average touched_indices: "
                  << touched_indices_sum / counter
                  << ", average inverted_index_length: "
                  << inverted_index_length_sum / counter << " QPS:"
                  << (counter /
                      std::chrono::duration_cast<std::chrono::duration<double>>(
                          batch_time)
                          .count())
                  << ", average latency per query:"
                  << batch_time.count() / counter << " ms" << std::endl;
    }
}