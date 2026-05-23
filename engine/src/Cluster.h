#ifndef __CLUSTERT_H__
#define __CLUSTERT_H__

#include <cereal/types/unordered_set.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataTypes.h"
#include "VectorProcessor.h"
#include "types.h"

template <template <typename...> class CollectionType, typename ValueType>
class ClusterT {
  public:
    SparseVector centroid;
    SparseVector summary;
    using indices_t = CollectionType<ValueType>;
    indices_t indices;
    s_size_t cluster_id;
    ClusterT() {}
    ClusterT(SparseVector centroid, SparseVector summary, indices_t indices)
        : centroid(centroid), summary(summary), indices(indices),
          cluster_id(0) {}
    ClusterT(const ClusterT &other)
        : centroid(other.centroid), summary(other.summary),
          indices(other.indices), cluster_id(other.cluster_id) {}

    ClusterT &operator=(const ClusterT &other) {
        this->centroid = other.centroid;
        this->summary = other.summary;
        this->indices = other.indices;
        this->cluster_id = other.cluster_id;
        return *this;
    }

    ClusterT(const ClusterT &&other)
        : centroid(other.centroid), summary(other.summary),
          indices(other.indices), cluster_id(other.cluster_id) {}

    ClusterT(indices_t &indices) : indices(std::move(indices)){};

    void Clear() { indices.clear(); }

    void ClearCentroid() {
        centroid.clear();
        centroid.shrink_to_fit();
    }

    void Add(ValueType index) {
        if constexpr (std::is_same_v<CollectionType<ValueType>,
                                     std::vector<ValueType>>) {
            indices.push_back(index);
        } else if constexpr (std::is_same_v<CollectionType<ValueType>,
                                            std::unordered_set<ValueType>>) {
            indices.insert(index);
        }
    }

    // Helper function to get the integer value
    static auto GetIndex(const ValueType &value) {
        return GetIndexType<ValueType>::get(value);
    }

    void UpdateCenter(std::span<const SparseVector> vectors) {
        std::map<term_t, std::vector<value_t>> center;
        for (auto &index : indices) {
            auto idx = ClusterT<CollectionType, ValueType>::GetIndex(index);
            assert(idx < vectors.size());
            for (size_t j = 0; j < vectors[idx].size(); ++j) {
                auto &element = vectors[idx][j];
                center[element.index].push_back(element.value);
            }
        }
        centroid = SparseVector(center.size());
        int i = 0;
        for (auto &p : center) {
            if (p.second.empty()) {
                centroid[i] = {p.first, value_t(0)};
            } else {
                float sum = 0.0;
                for (const auto &val : p.second) {
                    sum += val;
                }
                centroid[i] = {p.first,
                               static_cast<value_t>(sum / p.second.size())};
            }
            ++i;
        }
    }

    void GenerateSummary(std::span<const SparseVector> vectors) {
        if (indices.empty()) {
            summary = SparseVector();
            return;
        }
        std::map<term_t, value_t> summary_map;
        for (auto index : indices) {
            auto idx = ClusterT<CollectionType, ValueType>::GetIndex(index);
            assert(idx < vectors.size());
            const auto &vec = vectors[idx];
            for (size_t j = 0; j < vec.size(); ++j) {
                auto &element = vec[j];
                auto [it, inserted] =
                    summary_map.try_emplace(element.index, element.value);
                if (!inserted) {
                    it->second = std::max(it->second, element.value);
                }
            }
        }
        summary = SparseVector(summary_map.size());
        int i = 0;
        for (const auto &[term, value] : summary_map) {
            summary[i].index = term;
            summary[i].value = value;
            ++i;
        }
    }

    void SummaryToAMassSubVector(float alpha) {
        summary = VectorProcessor::GetAlphaMassSubVector(summary, alpha);
    }

    // we only serialize indices and build summary on the fly
    template <class Archive>
    void save(Archive &ar, std::uint32_t const version) const {
        ar(indices);
    }

    template <class Archive>
    void load(Archive &ar, std::uint32_t const version) {
        ar(indices);
    }

    void SetId(s_size_t id) { cluster_id = id; }

  private:
    // Type trait to get the index type from either an integer or a pair
    template <typename T> struct GetIndexType {
        using type = T;
        static const T &get(const T &value) { return value; }
    };

    template <typename First, typename Second>
    struct GetIndexType<std::pair<First, Second>> {
        using type = First; // Assuming the integer is always the first element
        static const First &get(const std::pair<First, Second> &value) {
            return value.first;
        }
    };
};

using ValuedCluster = ClusterT<std::vector, size_value_t>;
using valued_cluster_vector = std::vector<ValuedCluster>;
using valued_cluster_map_t = std::unordered_map<term_t, valued_cluster_vector>;

using Cluster = ClusterT<std::vector, s_size_t>;
using cluster_vector = std::vector<Cluster>;
using cluster_map_t = std::unordered_map<term_t, cluster_vector>;

inline static ValuedCluster
ToValuedCluster(const term_t term, const Cluster &cluster,
                std::span<const SparseVector> vectors) {
    std::vector<std::pair<s_size_t, value_t>> indices;
    for (auto &idx : cluster.indices) {
        auto &vec = vectors[idx];
        for (auto &element : vec) {
            if (element.index == term) {
                indices.push_back({idx, element.value});
            }
        }
    }
    return ValuedCluster(cluster.centroid, cluster.summary, indices);
}

inline static valued_cluster_vector
ToValuedClusters(const term_t term, const cluster_vector &kmean_cluster_vector,
                 std::span<const SparseVector> vectors) {
    valued_cluster_vector clusters;
    clusters.reserve(kmean_cluster_vector.size());
    for (auto &kmean_cluster : kmean_cluster_vector) {
        clusters.push_back(ToValuedCluster(term, kmean_cluster, vectors));
    }
    return clusters;
}

using cluster_center_t =
    std::pair<ClusterT<std::vector, s_size_t>, SparseVector>;
using v_cluster_center_t = std::vector<cluster_center_t>;
#endif
