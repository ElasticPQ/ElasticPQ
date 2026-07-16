#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <faiss/Index.h>

#include "epq/structure.h"

namespace epq {

struct BuildContext {
    int d = 0;
    int total_bits = 0;
    int min_bits = 0;
    int max_bits = 12;
};

class StructureBuilder {
   public:
    virtual ~StructureBuilder() = default;

    virtual Structure build(faiss::idx_t n, const float* x, const BuildContext& ctx) const = 0;
    virtual std::unique_ptr<StructureBuilder> clone() const = 0;
    virtual std::string name() const = 0;
};

class FixedStructureBuilder final : public StructureBuilder {
   public:
    explicit FixedStructureBuilder(Structure structure);

    const Structure& structure() const noexcept;

    Structure build(faiss::idx_t n, const float* x, const BuildContext& ctx) const override;
    std::unique_ptr<StructureBuilder> clone() const override;
    std::string name() const override;

   private:
    Structure structure_;
};

class BalancedStructureBuilder final : public StructureBuilder {
   public:
    int target_groups = 0;
    int nominal_group_bits = 8;

    Structure build(faiss::idx_t n, const float* x, const BuildContext& ctx) const override;
    std::unique_ptr<StructureBuilder> clone() const override;
    std::string name() const override;
};

class VarianceAwareStructureBuilder final : public StructureBuilder {
   public:
    float alpha_groups = 2.0f;
    int min_groups = 16;
    int max_groups = 0;
    int target_groups = 0;
    int corr_sample_rows = 4096;
    bool abs_correlation = true;
    float size_penalty = 0.01f;
    int seed = 123;

    Structure build(faiss::idx_t n, const float* x, const BuildContext& ctx) const override;
    std::unique_ptr<StructureBuilder> clone() const override;
    std::string name() const override;
};

class RefinedStructureBuilder final : public StructureBuilder {
   public:
    bool use_grow = true;
    bool use_crystallize = true;
    bool use_mbeam = false;
    bool use_greedy_tail = false;
    bool use_chain_tail = true;

    int seed = 123;

    int proxy_max_train_rows = 16384;
    int proxy_max_eval_rows = 4096;
    float proxy_eval_frac = 0.2f;
    int proxy_kmeans_niter = 8;
    int proxy_kmeans_nredo = 1;
    int proxy_min_points_per_centroid = 4;
    bool proxy_cache_slices = true;
    int proxy_max_d_cache = 400000;
    uint64_t proxy_max_slice_cache_bytes = 64ULL << 30;
    int proxy_pca_top_dims = 0;
    int proxy_max_pca_cache = 2048;

    float grow_alpha_groups = 2.0f;
    int grow_min_groups = 16;
    int grow_target_groups = 0;
    int grow_max_groups = 0;
    int grow_corr_adj_k = 16;
    bool grow_corr_adj_abs = true;
    int grow_corr_adj_rows = 4096;
    float grow_edge_tau = 0.0f;
    int grow_dmax = 1024;
    int grow_min_group_size = 1;
    int grow_min_votes = 2;
    float grow_avg_gain_tau = 0.0f;
    bool grow_fill_when_stuck = true;
    int grow_score_bits_fixed = 4;
    int grow_rerank_L = 32;
    int grow_seed_topk = 16;
    bool grow_seed_pair = true;

    int crystallize_dmax = 1024;
    int crystallize_candidates = 128;
    int crystallize_shortlist_factor = 4;
    int crystallize_pool_mult = 16;
    float crystallize_weight_corr = 0.4f;
    float crystallize_weight_long = 0.6f;
    int crystallize_corr_adj_k = 16;
    bool crystallize_corr_adj_abs = true;
    int crystallize_corr_adj_rows = 4096;
    float crystallize_corr_two_hop_ratio = 0.25f;
    int crystallize_corr_two_hop_per_gid = 4;
    float crystallize_long_oversample = 2.0f;
    float crystallize_long_edge_power = 0.5f;
    int crystallize_endpoint_quota = 12;
    int crystallize_proxy_bits = 4;
    double crystallize_struct_tol = 1e-6;
    int crystallize_beam_width = 8;
    int crystallize_beam_topR = 8;
    int crystallize_beam_max_depth = 1000000;
    int crystallize_fast_proxy_top_dims = 0;

    int mbeam_iters = 1000;
    int mbeam_patience = 30;
    double mbeam_eps_improve = 0.0;
    int mbeam_beam_width = 4;
    int mbeam_per_state_eval_topk = 6;
    int mbeam_per_state_shortlist_k = 24;
    int mbeam_donor_topk = 10;
    int mbeam_recv_topk = 10;
    int mbeam_dims_sample_per_group = 10;
    float mbeam_suspicious_alpha = 1.0f;
    int mbeam_n_relocate = 128;
    int mbeam_n_swap_pairs = 48;
    int mbeam_relocate_pair_limit = 2;
    int mbeam_swap_pair_limit = 2;
    int mbeam_shortlist_per_pair = 2;
    double mbeam_max_local_score = 0.0;
    double mbeam_shift_lambda = 1.0;
    int mbeam_seen_window = 5;
    int mbeam_min_novel_children = 0;

    int greedy_tail_iters = 96;
    int greedy_tail_patience = 10;
    double greedy_tail_eps_improve = 0.0;
    int greedy_tail_eval_topk = 3;
    int greedy_tail_shortlist_k = 12;
    int greedy_tail_donor_topk = 8;
    int greedy_tail_recv_topk = 8;
    int greedy_tail_dims_sample_per_group = 6;
    float greedy_tail_suspicious_alpha = 1.0f;
    int greedy_tail_n_relocate = 48;
    int greedy_tail_n_swap_pairs = 16;
    int greedy_tail_relocate_pair_limit = 1;
    int greedy_tail_swap_pair_limit = 1;
    int greedy_tail_shortlist_per_pair = 1;
    double greedy_tail_max_local_score = 0.0;
    double greedy_tail_shift_lambda = 1.0;
    int greedy_tail_seen_window = 6;

    int chain_tail_iters = 96;
    int chain_tail_patience = 10;
    double chain_tail_eps_improve = 0.0;
    int chain_tail_eval_topk = 3;
    int chain_tail_shortlist_k = 12;
    int chain_tail_donor_topk = 8;
    int chain_tail_recv_topk = 8;
    int chain_tail_dims_sample_per_group = 6;
    float chain_tail_suspicious_alpha = 1.0f;
    int chain_tail_n_seed_moves = 48;
    int chain_tail_receiver_topk_per_dim = 3;
    int chain_tail_max_depth = 4;
    double chain_tail_max_local_score = 0.0;
    double chain_tail_prefix_slack = 0.0;
    int chain_tail_seen_window = 6;
    int chain_tail_fast_proxy_top_dims = 0;
    int chain_tail_fast_shortlist_mult = 2;

    Structure build(faiss::idx_t n, const float* x, const BuildContext& ctx) const override;
    std::unique_ptr<StructureBuilder> clone() const override;
    std::string name() const override;
};

}  // namespace epq
