#include "VectorProcessor.h"
#include <unordered_set>

std::unordered_map<term_t, size_t> VectorProcessor::pruned_token_counts_;
std::unordered_map<term_t, size_t> VectorProcessor::posting_list_lengths_;

void VectorProcessor::RecordPrunedTokens(
    const SparseVector &original, const SparseVector &pruned,
    const std::function<size_t(term_t)> &get_posting_length) {
    std::unordered_set<term_t> pruned_indices;
    for (const auto &ele : pruned) {
        pruned_indices.insert(ele.index);
    }

    for (const auto &ele : original) {
        if (pruned_indices.find(ele.index) == pruned_indices.end()) {
            pruned_token_counts_[ele.index]++;
            if (posting_list_lengths_.find(ele.index) ==
                posting_list_lengths_.end()) {
                posting_list_lengths_[ele.index] =
                    get_posting_length(ele.index);
            }
        }
    }
}

void VectorProcessor::SavePrunedTokensToFile(const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    file << "token_id,pruned_count,posting_list_length\n";
    for (const auto &[token_id, count] : pruned_token_counts_) {
        size_t posting_length = posting_list_lengths_[token_id];
        file << token_id << "," << count << "," << posting_length << "\n";
    }

    file.close();
    std::cout << "Pruned tokens saved to " << filename << std::endl;
}
