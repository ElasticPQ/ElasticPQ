#pragma once

#include "epq/structure_builder.h"

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <list>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace epq::structure_builder_internal {

using RowMatrixXf =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Groups = std::vector<std::vector<int>>;
using Bits = std::vector<int>;
using DimsKey = std::vector<int>;
using DGKey = std::pair<DimsKey, int>;
using PartitionKey = std::vector<DimsKey>;

#ifndef EPQ_STRUCTURE_DEBUG_LEVEL
#define EPQ_STRUCTURE_DEBUG_LEVEL 0
#endif

#ifndef EPQ_ENABLE_STRUCTURE_TRACE
#define EPQ_ENABLE_STRUCTURE_TRACE 0
#endif

#if EPQ_STRUCTURE_DEBUG_LEVEL > 0
#define EPQ_STRUCTURE_DEBUG_LOG(level, expr)                                     \
    do {                                                                         \
        if (EPQ_STRUCTURE_DEBUG_LEVEL >= (level)) {                              \
            std::cout << "[epq][structure] " << expr << '\n';                    \
        }                                                                        \
    } while (0)
#else
#define EPQ_STRUCTURE_DEBUG_LOG(level, expr) \
    do {                                     \
    } while (0)
#endif

struct BitAllocResult {
    double J = 0.0;
    Bits bits;
};

template <typename Key>
inline void hash_combine(std::size_t& seed, const Key& value) {
    seed ^= std::hash<Key>{}(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) +
            (seed >> 2);
}

struct DimsKeyHash {
    std::size_t operator()(const DimsKey& dims) const noexcept {
        std::size_t seed = dims.size();
        for (int dim : dims) {
            hash_combine(seed, dim);
        }
        return seed;
    }
};

struct DGKeyHash {
    std::size_t operator()(const DGKey& key) const noexcept {
        std::size_t seed = DimsKeyHash{}(key.first);
        hash_combine(seed, key.second);
        return seed;
    }
};

struct PartitionKeyHash {
    std::size_t operator()(const PartitionKey& key) const noexcept {
        std::size_t seed = key.size();
        for (const auto& dims : key) {
            hash_combine(seed, DimsKeyHash{}(dims));
        }
        return seed;
    }
};

struct IntPairHash {
    std::size_t operator()(const std::pair<int, int>& key) const noexcept {
        std::size_t seed = 0;
        hash_combine(seed, key.first);
        hash_combine(seed, key.second);
        return seed;
    }
};

template <typename Key, typename Value, typename Hash = std::hash<Key>>
struct LruMapCache {
    static size_t weight_of(const Value&) {
        return 1;
    }

    struct Entry {
        Value value;
        size_t weight = 0;
        typename std::list<Key>::iterator order_it;
    };

    explicit LruMapCache(size_t max_size_in = 0) : max_size(max_size_in) {}

    const Value* get(const Key& key) {
        auto it = entries.find(key);
        if (it == entries.end()) {
            return nullptr;
        }
        order.splice(order.begin(), order, it->second.order_it);
        return &it->second.value;
    }

    template <typename V>
    void set(const Key& key, V&& value) {
        if (max_size == 0 && max_weight == 0) {
            return;
        }
        const size_t new_weight = weight_of(value);
        auto it = entries.find(key);
        if (it != entries.end()) {
            current_weight -= it->second.weight;
            it->second.value = std::forward<V>(value);
            it->second.weight = new_weight;
            current_weight += new_weight;
            order.splice(order.begin(), order, it->second.order_it);
            trim();
            return;
        }
        order.push_front(key);
        entries.emplace(
                key,
                Entry{
                        .value = std::forward<V>(value),
                        .weight = new_weight,
                        .order_it = order.begin(),
                });
        current_weight += new_weight;
        trim();
    }

    size_t size() const {
        return entries.size();
    }

    size_t weight() const {
        return current_weight;
    }

    bool enabled() const {
        return max_size > 0 || max_weight > 0;
    }

    size_t max_size = 0;
    size_t max_weight = 0;
    size_t current_weight = 0;
    std::list<Key> order;
    std::unordered_map<Key, Entry, Hash> entries;

   private:
    void trim() {
        while (((max_size > 0 && entries.size() > max_size) ||
                (max_weight > 0 && current_weight > max_weight)) &&
               !order.empty()) {
            const Key evict_key = order.back();
            auto it = entries.find(evict_key);
            if (it != entries.end()) {
                current_weight -= it->second.weight;
            }
            order.pop_back();
            entries.erase(evict_key);
        }
    }
};

struct ProxyCacheStats {
    uint64_t d_hits = 0;
    uint64_t d_misses = 0;
    uint64_t d_fast_hits = 0;
    uint64_t d_fast_misses = 0;
    uint64_t xtr_hits = 0;
    uint64_t xtr_misses = 0;
    uint64_t xev_hits = 0;
    uint64_t xev_misses = 0;
    uint64_t pca_hits = 0;
    uint64_t pca_misses = 0;
    uint64_t pca_fast_hits = 0;
    uint64_t pca_fast_misses = 0;
};

struct ProxyWorkStats {
    uint64_t d_calls = 0;
    uint64_t d_empty_calls = 0;
    uint64_t d_fast_calls = 0;
    uint64_t d_fast_empty_calls = 0;
    uint64_t kmeans_calls = 0;
    uint64_t kmeans_k_total = 0;
    uint64_t kmeans_dims_total = 0;
    uint64_t kmeans_train_rows_total = 0;
    uint64_t kmeans_eval_rows_total = 0;
    uint64_t kmeans_fast_calls = 0;
    uint64_t kmeans_fast_k_total = 0;
    uint64_t kmeans_fast_dims_total = 0;
    uint64_t kmeans_fast_train_rows_total = 0;
    uint64_t kmeans_fast_eval_rows_total = 0;
    uint64_t pca_approx_calls = 0;
    uint64_t pca_fits = 0;
    uint64_t pca_full_dims_total = 0;
    uint64_t pca_proj_dims_total = 0;
    uint64_t pca_tail_dims_total = 0;
    uint64_t pca_fast_approx_calls = 0;
    uint64_t pca_fast_fits = 0;
    uint64_t pca_fast_full_dims_total = 0;
    uint64_t pca_fast_proj_dims_total = 0;
    uint64_t pca_fast_tail_dims_total = 0;
    uint64_t solve_bits_calls = 0;
    uint64_t solve_bits_groups_total = 0;
    uint64_t solve_bits_cost_evals = 0;
    uint64_t solve_bits_dp_states = 0;
    uint64_t solve_bits_dp_transitions = 0;
};

struct ChainTailProfile {
    bool used = false;
    uint64_t iterations = 0;
    uint64_t iters_with_candidates = 0;
    uint64_t seeds_raw_total = 0;
    uint64_t seeds_kept_total = 0;
    uint64_t candidates_total = 0;
    uint64_t exact_local_reranked_total = 0;
    uint64_t exact_local_kept_total = 0;
    uint64_t local_gate_pruned_total = 0;
    uint64_t donor_small_stops_total = 0;
    uint64_t no_step_stops_total = 0;
    uint64_t prefix_cut_stops_total = 0;
    uint64_t total_steps = 0;
    uint64_t max_steps = 0;
    uint64_t exact_attempted = 0;
    uint64_t exact_children = 0;
    uint64_t exact_dup_pruned = 0;
    uint64_t exact_seen_pruned = 0;
    uint64_t improved_iters = 0;
};

int min_feasible_groups(const BuildContext& ctx);
void validate_build_context(const BuildContext& ctx);
std::vector<int> distribute_bits_evenly(int total_bits, int groups);
std::vector<std::vector<int>> balanced_groups(int d, int groups);
RowMatrixXf sample_rows(
        faiss::idx_t n,
        const float* x,
        int d,
        int max_rows,
        int seed);
RowMatrixXf gather_columns(const RowMatrixXf& x, const std::vector<int>& dims);
uint64_t stable_hash_dims(const std::vector<int>& dims);

template <typename T>
void shuffle_vector(std::vector<T>& values, std::mt19937& rng) {
    std::shuffle(values.begin(), values.end(), rng);
}

template <typename T>
std::vector<T> sample_vector(
        const std::vector<T>& values,
        int count,
        std::mt19937& rng) {
    if (count <= 0 || values.empty()) {
        return {};
    }
    if (static_cast<int>(values.size()) <= count) {
        return values;
    }
    std::vector<T> out = values;
    std::shuffle(out.begin(), out.end(), rng);
    out.resize(static_cast<size_t>(count));
    return out;
}

template <typename T>
std::vector<int> top_indices(
        const std::vector<T>& scores,
        int k,
        bool descending = true) {
    std::vector<int> ids(scores.size());
    std::iota(ids.begin(), ids.end(), 0);
    auto cmp = [&](int lhs, int rhs) {
        return descending ? (scores[lhs] > scores[rhs]) : (scores[lhs] < scores[rhs]);
    };
    if (static_cast<int>(ids.size()) > k) {
        std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), cmp);
        ids.resize(static_cast<size_t>(k));
    } else {
        std::sort(ids.begin(), ids.end(), cmp);
    }
    return ids;
}

double median_value(std::vector<double> values);
RowMatrixXf train_kmeans_seeded(
        const RowMatrixXf& x,
        int k,
        int niter,
        int nredo,
        int seed,
        int min_points_per_centroid);
double kmeans_recon_mse_holdout(
        const RowMatrixXf& x_train,
        const RowMatrixXf& x_eval,
        int k,
        int niter,
        int nredo,
        int seed,
        int min_points_per_centroid);

struct TrainEvalSplit {
    RowMatrixXf train;
    RowMatrixXf eval;
};

struct ProxyPcaSlice {
    RowMatrixXf train_proj;
    RowMatrixXf eval_proj;
    double tail_eval_mse = 0.0;
    int full_dims = 0;
    int proj_dims = 0;
};

template <>
inline size_t LruMapCache<DimsKey, RowMatrixXf, DimsKeyHash>::weight_of(
        const RowMatrixXf& value) {
    return static_cast<size_t>(value.rows()) * static_cast<size_t>(value.cols()) *
            sizeof(float);
}

TrainEvalSplit split_train_eval_rows(
        const RowMatrixXf& x,
        int max_train,
        int max_eval,
        float eval_frac,
        int seed);
void validate_partition(
        const Groups& groups,
        int d,
        bool require_cover,
        bool allow_empty_group = false);
PartitionKey canonical_partition_key(const Groups& groups);
std::vector<int> remove_one(const std::vector<int>& group, int v);
Structure make_structure(
        const Groups& groups,
        const Bits& bits,
        const BuildContext& ctx,
        const std::string& builder_name);

#if EPQ_ENABLE_STRUCTURE_TRACE
void trace_structure_candidate(
        const BuildContext& ctx,
        const std::string& stage,
        int step,
        const Groups& groups,
        const Bits& bits,
        double j_star,
        const std::string& source);
#else
inline void trace_structure_candidate(
        const BuildContext&,
        const std::string&,
        int,
        const Groups&,
        const Bits&,
        double,
        const std::string&) {}
#endif

struct ProxyContext {
    BuildContext build_ctx;
    RowMatrixXf xt_train;
    RowMatrixXf xt_eval;
    int km_niter = 8;
    int km_nredo = 1;
    int min_points_per_centroid = 4;
    int pca_top_dims = 0;
    int fast_pca_top_dims = 0;
    int seed = 123;
    bool cache_slices = true;
    LruMapCache<DGKey, double, DGKeyHash> d_cache{400000};
    LruMapCache<DGKey, double, DGKeyHash> d_fast_cache{400000};
    LruMapCache<DimsKey, RowMatrixXf, DimsKeyHash> xtr_cache{0};
    LruMapCache<DimsKey, RowMatrixXf, DimsKeyHash> xev_cache{0};
    LruMapCache<DimsKey, ProxyPcaSlice, DimsKeyHash> pca_cache{2048};
    LruMapCache<DimsKey, ProxyPcaSlice, DimsKeyHash> pca_fast_cache{2048};
    ProxyCacheStats cache_stats;
    ProxyWorkStats work_stats;
    ChainTailProfile chain_tail_profile;

    double D(const std::vector<int>& dims, int bits);
    double D_fast(const std::vector<int>& dims, int bits);
    BitAllocResult solve_bits(const Groups& groups, bool allow_partial = false);
};

bool group_stats_env_enabled();
void print_group_proxy_stats(
        std::ostream& os,
        const std::string& quantizer_name,
        const std::string& space_label,
        const Groups& groups,
        const Bits& bits,
        const BuildContext& ctx,
        ProxyContext& proxy);
void print_group_proxy_stats_from_matrix(
        std::ostream& os,
        const std::string& quantizer_name,
        const std::string& space_label,
        const Groups& groups,
        const Bits& bits,
        const RowMatrixXf& xt,
        const BuildContext& ctx,
        int proxy_max_train_rows = 16384,
        int proxy_max_eval_rows = 4096,
        float proxy_eval_frac = 0.2f,
        int proxy_kmeans_niter = 8,
        int proxy_kmeans_nredo = 1,
        int proxy_min_points_per_centroid = 4,
        int seed = 123);

int score_bits_for_group(int d, int B, int proxy_bmax, int fixed_bits);
std::vector<std::vector<std::pair<int, float>>> build_dim_neighbors_by_corr_weighted(
        const RowMatrixXf& xt,
        int knn,
        bool abs_corr,
        int max_rows,
        int seed,
        float edge_tau);
std::vector<std::vector<int>> build_dim_neighbors_by_corr(
        const RowMatrixXf& xt,
        int knn,
        bool abs_corr,
        int max_rows,
        int seed);
Groups singleton_groups(int d);
std::vector<int> greedy_allocate_bits(
        const std::vector<float>& weights,
        const BuildContext& ctx);

std::pair<Groups, Bits> run_grow_stage(
        const RefinedStructureBuilder& cfg,
        ProxyContext& proxy,
        const BuildContext& ctx);
std::pair<Groups, Bits> run_crystallize_stage(
        const RefinedStructureBuilder& cfg,
        ProxyContext& proxy,
        const BuildContext& ctx,
        const Groups& groups,
        const Bits& bits_in);
std::pair<Groups, Bits> run_mbeam_stage(
        const RefinedStructureBuilder& cfg,
        ProxyContext& proxy,
        const BuildContext& ctx,
        const Groups& groups,
        const Bits& bits);
std::pair<Groups, Bits> run_greedy_tail_stage(
        const RefinedStructureBuilder& cfg,
        ProxyContext& proxy,
        const BuildContext& ctx,
        const Groups& groups,
        const Bits& bits);
std::pair<Groups, Bits> run_chain_tail_stage(
        const RefinedStructureBuilder& cfg,
        ProxyContext& proxy,
        const BuildContext& ctx,
        const Groups& groups,
        const Bits& bits);

}  // namespace epq::structure_builder_internal
