#include "InMemoryStorage.h"

#include <iostream>

void InMemoryStorage::AddVectors(vector<SparseVector> vectors, int size,
                                 int index) {
    if (size < 0 || size > vectors.size()) {
        std::cerr << "Invalid size parameter" << std::endl;
        throw std::invalid_argument("Invalid size parameter");
    }

    m_vectors.reserve(m_vectors.size() + size);

    try {
        m_vectors.insert(m_vectors.end(),
                         std::make_move_iterator(vectors.begin()),
                         std::make_move_iterator(vectors.begin() + size));
    } catch (const std::exception &e) {
        std::cerr << "Exception during vector insertion: " << e.what()
                  << std::endl;
        throw;
    }
}

const span<const SparseVector> InMemoryStorage::GetRange(int start,
                                                         int end) const {
    // Ensure the range is valid
    int count = end - start;
    if (start + count > m_vectors.size()) {
        count = m_vectors.size() - start;
    }
    return std::span<const SparseVector>(m_vectors.data() + start, count);
}

const span<const SparseVector> InMemoryStorage::GetAll() const {
    return std::span<const SparseVector>(m_vectors.data(), m_vectors.size());
}

void InMemoryStorage::UpdateVectors(const vector<SparseVector> &new_vectors) {
    try {
        // Clear existing vectors and reserve space for new ones
        m_vectors.clear();
        m_vectors.reserve(new_vectors.size());

        // Copy new vectors into storage
        m_vectors = new_vectors;
    } catch (const std::exception &e) {
        std::cerr << "Exception during vector update: " << e.what()
                  << std::endl;
        throw;
    }
}

size_t InMemoryStorage::GetMemoryUsage() const {
    size_t total = sizeof(*this);
    total += m_vectors.size() * sizeof(SparseVector);
    for (const auto &vec : m_vectors) {
        total += vec.size() * sizeof(SparseVectorElement);
    }
    return total;
}