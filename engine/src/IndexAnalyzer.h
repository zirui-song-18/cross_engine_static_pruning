#ifndef __INDEX_ANALYZER_H__
#define __INDEX_ANALYZER_H__

#include <atomic>
#include <chrono>
#include <unordered_map>
#include <vector>

#include "Cluster.h"
#include "DataTypes.h"
#include "InvertedIndex.h"

class IndexAnalyzer {
  private:
    IndexAnalyzer() {}
    ~IndexAnalyzer() {}

  public:
    static IndexAnalyzer *GetInstance() {
        static IndexAnalyzer *instance = new IndexAnalyzer();
        return instance;
    }
    void AnalyzeInvertedIndex(InvertedIndex *invertedIndex);
    void AnalyzeClusters(const cluster_map_t &clusters);
    void RecordQuery(uint64_t touched_indices_count,
                     uint64_t inverted_index_length);
    void RecordQueryLatency(double latency_ms);
    void SetQueryCount(size_t n);
    void RecordQueryLatencyAt(size_t query_idx, double latency_ms);
    void WritePerQueryCSV(const std::string &path);
    void Reset();
    void AnalyzeQuery(std::chrono::duration<double, std::milli> milli);

  private:
    std::atomic<uint64_t> cluster_examined;
    std::atomic<uint64_t> dp;
    std::atomic<uint64_t> counter;
    std::atomic<uint64_t> touched_indices_sum;
    std::atomic<uint64_t> inverted_index_length_sum;

    // For per-query latency tracking
    std::mutex latency_mutex;
    std::vector<double>
        query_latencies; // Store individual query latencies in ms

    // Indexed per-query latencies (thread-safe without mutex: each idx written once)
    std::vector<double> per_query_latencies;
};

#endif