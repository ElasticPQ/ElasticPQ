#include "epq/training_config.h"

#include "epq/index_epq.h"
#include "epq/structure.h"
#include "epq/structure_builder.h"

#include <faiss/utils/distances.h>

#include <fstream>
#include <stdexcept>
#include <string_view>

namespace epq {
namespace {

template <typename T>
void assign_if_present(const nlohmann::json& j, std::string_view key, T& out) {
    const std::string key_str(key);
    if (j.contains(key_str)) {
        out = j.at(key_str).get<T>();
    }
}

const nlohmann::json* find_section(
        const nlohmann::json& j,
        std::initializer_list<std::string_view> names) {
    for (const auto name : names) {
        const std::string key(name);
        if (j.contains(key) && j.at(key).is_object()) {
            return &j.at(key);
        }
    }
    return nullptr;
}

std::filesystem::path resolve_path(
        const std::filesystem::path& raw,
        const std::filesystem::path& base_dir) {
    if (raw.is_absolute() || base_dir.empty()) {
        return raw;
    }
    return base_dir / raw;
}

const nlohmann::json& resolve_builder_payload(
        const nlohmann::json& section,
        std::string_view type) {
    const std::string key(type);
    if (section.contains(key) && section.at(key).is_object()) {
        return section.at(key);
    }
    return section;
}

bool find_bool_option(
        const nlohmann::json& config,
        std::initializer_list<std::string_view> section_names,
        std::initializer_list<std::string_view> keys,
        bool default_value) {
    const nlohmann::json* section = find_section(config, section_names);
    if (section == nullptr) {
        return default_value;
    }
    for (const auto key : keys) {
        const std::string key_str(key);
        if (section->contains(key_str)) {
            return section->at(key_str).get<bool>();
        }
    }
    return default_value;
}

template <typename BuilderT>
std::shared_ptr<StructureBuilder> make_builder_with_common_json(
        const nlohmann::json& payload);

template <>
std::shared_ptr<StructureBuilder> make_builder_with_common_json<BalancedStructureBuilder>(
        const nlohmann::json& payload) {
    auto builder = std::make_shared<BalancedStructureBuilder>();
    assign_if_present(payload, "target_groups", builder->target_groups);
    assign_if_present(payload, "nominal_group_bits", builder->nominal_group_bits);
    return builder;
}

template <>
std::shared_ptr<StructureBuilder> make_builder_with_common_json<VarianceAwareStructureBuilder>(
        const nlohmann::json& payload) {
    auto builder = std::make_shared<VarianceAwareStructureBuilder>();
    assign_if_present(payload, "alpha_groups", builder->alpha_groups);
    assign_if_present(payload, "min_groups", builder->min_groups);
    assign_if_present(payload, "max_groups", builder->max_groups);
    assign_if_present(payload, "target_groups", builder->target_groups);
    assign_if_present(payload, "corr_sample_rows", builder->corr_sample_rows);
    assign_if_present(payload, "abs_correlation", builder->abs_correlation);
    assign_if_present(payload, "size_penalty", builder->size_penalty);
    assign_if_present(payload, "seed", builder->seed);
    return builder;
}

template <>
std::shared_ptr<StructureBuilder> make_builder_with_common_json<RefinedStructureBuilder>(
        const nlohmann::json& payload) {
    auto builder = std::make_shared<RefinedStructureBuilder>();
    assign_if_present(payload, "use_grow", builder->use_grow);
    assign_if_present(payload, "use_crystallize", builder->use_crystallize);
    assign_if_present(payload, "use_mbeam", builder->use_mbeam);
    assign_if_present(payload, "use_greedy_tail", builder->use_greedy_tail);
    assign_if_present(payload, "use_fast_tail", builder->use_greedy_tail);
    assign_if_present(payload, "use_chain_tail", builder->use_chain_tail);
    assign_if_present(payload, "seed", builder->seed);

    assign_if_present(payload, "proxy_max_train_rows", builder->proxy_max_train_rows);
    assign_if_present(payload, "proxy_max_eval_rows", builder->proxy_max_eval_rows);
    assign_if_present(payload, "proxy_eval_frac", builder->proxy_eval_frac);
    assign_if_present(payload, "proxy_kmeans_niter", builder->proxy_kmeans_niter);
    assign_if_present(payload, "proxy_kmeans_nredo", builder->proxy_kmeans_nredo);
    assign_if_present(
            payload,
            "proxy_min_points_per_centroid",
            builder->proxy_min_points_per_centroid);
    assign_if_present(payload, "proxy_cache_slices", builder->proxy_cache_slices);
    assign_if_present(payload, "proxy_max_d_cache", builder->proxy_max_d_cache);
    assign_if_present(
            payload,
            "proxy_max_slice_cache_bytes",
            builder->proxy_max_slice_cache_bytes);
    assign_if_present(payload, "proxy_pca_top_dims", builder->proxy_pca_top_dims);
    assign_if_present(payload, "proxy_max_pca_cache", builder->proxy_max_pca_cache);

    assign_if_present(payload, "grow_alpha_groups", builder->grow_alpha_groups);
    assign_if_present(payload, "grow_min_groups", builder->grow_min_groups);
    assign_if_present(payload, "grow_target_groups", builder->grow_target_groups);
    assign_if_present(payload, "grow_max_groups", builder->grow_max_groups);
    assign_if_present(payload, "grow_corr_adj_k", builder->grow_corr_adj_k);
    assign_if_present(payload, "grow_corr_adj_abs", builder->grow_corr_adj_abs);
    assign_if_present(payload, "grow_corr_adj_rows", builder->grow_corr_adj_rows);
    assign_if_present(payload, "grow_edge_tau", builder->grow_edge_tau);
    assign_if_present(payload, "grow_dmax", builder->grow_dmax);
    assign_if_present(payload, "grow_min_group_size", builder->grow_min_group_size);
    assign_if_present(payload, "grow_min_votes", builder->grow_min_votes);
    assign_if_present(payload, "grow_avg_gain_tau", builder->grow_avg_gain_tau);
    assign_if_present(payload, "grow_fill_when_stuck", builder->grow_fill_when_stuck);
    assign_if_present(payload, "grow_score_bits_fixed", builder->grow_score_bits_fixed);
    assign_if_present(payload, "grow_rerank_L", builder->grow_rerank_L);
    assign_if_present(payload, "grow_seed_topk", builder->grow_seed_topk);
    assign_if_present(payload, "grow_seed_pair", builder->grow_seed_pair);

    assign_if_present(payload, "crystallize_dmax", builder->crystallize_dmax);
    assign_if_present(
            payload, "crystallize_candidates", builder->crystallize_candidates);
    assign_if_present(
            payload,
            "crystallize_shortlist_factor",
            builder->crystallize_shortlist_factor);
    assign_if_present(payload, "crystallize_pool_mult", builder->crystallize_pool_mult);
    assign_if_present(
            payload, "crystallize_weight_corr", builder->crystallize_weight_corr);
    assign_if_present(
            payload, "crystallize_weight_long", builder->crystallize_weight_long);
    assign_if_present(
            payload, "crystallize_corr_adj_k", builder->crystallize_corr_adj_k);
    assign_if_present(
            payload, "crystallize_corr_adj_abs", builder->crystallize_corr_adj_abs);
    assign_if_present(
            payload,
            "crystallize_corr_adj_rows",
            builder->crystallize_corr_adj_rows);
    assign_if_present(
            payload,
            "crystallize_corr_two_hop_ratio",
            builder->crystallize_corr_two_hop_ratio);
    assign_if_present(
            payload,
            "crystallize_corr_two_hop_per_gid",
            builder->crystallize_corr_two_hop_per_gid);
    assign_if_present(
            payload,
            "crystallize_long_oversample",
            builder->crystallize_long_oversample);
    assign_if_present(
            payload,
            "crystallize_long_edge_power",
            builder->crystallize_long_edge_power);
    assign_if_present(
            payload,
            "crystallize_endpoint_quota",
            builder->crystallize_endpoint_quota);
    assign_if_present(
            payload, "crystallize_proxy_bits", builder->crystallize_proxy_bits);
    assign_if_present(
            payload, "crystallize_struct_tol", builder->crystallize_struct_tol);
    assign_if_present(
            payload, "crystallize_beam_width", builder->crystallize_beam_width);
    assign_if_present(
            payload, "crystallize_beam_topR", builder->crystallize_beam_topR);
    assign_if_present(
            payload,
            "crystallize_beam_max_depth",
            builder->crystallize_beam_max_depth);
    assign_if_present(
            payload,
            "crystallize_fast_proxy_top_dims",
            builder->crystallize_fast_proxy_top_dims);

    assign_if_present(payload, "mbeam_iters", builder->mbeam_iters);
    assign_if_present(payload, "mbeam_patience", builder->mbeam_patience);
    assign_if_present(
            payload, "mbeam_eps_improve", builder->mbeam_eps_improve);
    assign_if_present(payload, "mbeam_beam_width", builder->mbeam_beam_width);
    assign_if_present(
            payload,
            "mbeam_per_state_eval_topk",
            builder->mbeam_per_state_eval_topk);
    assign_if_present(
            payload,
            "mbeam_per_state_shortlist_k",
            builder->mbeam_per_state_shortlist_k);
    assign_if_present(payload, "mbeam_donor_topk", builder->mbeam_donor_topk);
    assign_if_present(payload, "mbeam_recv_topk", builder->mbeam_recv_topk);
    assign_if_present(
            payload,
            "mbeam_dims_sample_per_group",
            builder->mbeam_dims_sample_per_group);
    assign_if_present(
            payload,
            "mbeam_suspicious_alpha",
            builder->mbeam_suspicious_alpha);
    assign_if_present(payload, "mbeam_n_relocate", builder->mbeam_n_relocate);
    assign_if_present(payload, "mbeam_n_swap_pairs", builder->mbeam_n_swap_pairs);
    assign_if_present(
            payload,
            "mbeam_relocate_pair_limit",
            builder->mbeam_relocate_pair_limit);
    assign_if_present(
            payload, "mbeam_swap_pair_limit", builder->mbeam_swap_pair_limit);
    assign_if_present(
            payload,
            "mbeam_shortlist_per_pair",
            builder->mbeam_shortlist_per_pair);
    assign_if_present(
            payload, "mbeam_max_local_score", builder->mbeam_max_local_score);
    assign_if_present(payload, "mbeam_shift_lambda", builder->mbeam_shift_lambda);
    assign_if_present(payload, "mbeam_seen_window", builder->mbeam_seen_window);
    assign_if_present(
            payload,
            "mbeam_min_novel_children",
            builder->mbeam_min_novel_children);

    assign_if_present(payload, "greedy_tail_iters", builder->greedy_tail_iters);
    assign_if_present(payload, "fast_tail_iters", builder->greedy_tail_iters);
    assign_if_present(
            payload, "greedy_tail_patience", builder->greedy_tail_patience);
    assign_if_present(
            payload, "fast_tail_patience", builder->greedy_tail_patience);
    assign_if_present(
            payload,
            "greedy_tail_eps_improve",
            builder->greedy_tail_eps_improve);
    assign_if_present(
            payload,
            "fast_tail_eps_improve",
            builder->greedy_tail_eps_improve);
    assign_if_present(
            payload, "greedy_tail_eval_topk", builder->greedy_tail_eval_topk);
    assign_if_present(
            payload, "fast_tail_eval_topk", builder->greedy_tail_eval_topk);
    assign_if_present(
            payload,
            "greedy_tail_shortlist_k",
            builder->greedy_tail_shortlist_k);
    assign_if_present(
            payload,
            "fast_tail_shortlist_k",
            builder->greedy_tail_shortlist_k);
    assign_if_present(
            payload,
            "greedy_tail_donor_topk",
            builder->greedy_tail_donor_topk);
    assign_if_present(
            payload,
            "fast_tail_donor_topk",
            builder->greedy_tail_donor_topk);
    assign_if_present(
            payload, "greedy_tail_recv_topk", builder->greedy_tail_recv_topk);
    assign_if_present(
            payload, "fast_tail_recv_topk", builder->greedy_tail_recv_topk);
    assign_if_present(
            payload,
            "greedy_tail_dims_sample_per_group",
            builder->greedy_tail_dims_sample_per_group);
    assign_if_present(
            payload,
            "fast_tail_dims_sample_per_group",
            builder->greedy_tail_dims_sample_per_group);
    assign_if_present(
            payload,
            "greedy_tail_suspicious_alpha",
            builder->greedy_tail_suspicious_alpha);
    assign_if_present(
            payload,
            "fast_tail_suspicious_alpha",
            builder->greedy_tail_suspicious_alpha);
    assign_if_present(
            payload, "greedy_tail_n_relocate", builder->greedy_tail_n_relocate);
    assign_if_present(
            payload, "fast_tail_n_relocate", builder->greedy_tail_n_relocate);
    assign_if_present(
            payload,
            "greedy_tail_n_swap_pairs",
            builder->greedy_tail_n_swap_pairs);
    assign_if_present(
            payload,
            "fast_tail_n_swap_pairs",
            builder->greedy_tail_n_swap_pairs);
    assign_if_present(
            payload,
            "greedy_tail_relocate_pair_limit",
            builder->greedy_tail_relocate_pair_limit);
    assign_if_present(
            payload,
            "fast_tail_relocate_pair_limit",
            builder->greedy_tail_relocate_pair_limit);
    assign_if_present(
            payload,
            "greedy_tail_swap_pair_limit",
            builder->greedy_tail_swap_pair_limit);
    assign_if_present(
            payload,
            "fast_tail_swap_pair_limit",
            builder->greedy_tail_swap_pair_limit);
    assign_if_present(
            payload,
            "greedy_tail_shortlist_per_pair",
            builder->greedy_tail_shortlist_per_pair);
    assign_if_present(
            payload,
            "fast_tail_shortlist_per_pair",
            builder->greedy_tail_shortlist_per_pair);
    assign_if_present(
            payload,
            "greedy_tail_max_local_score",
            builder->greedy_tail_max_local_score);
    assign_if_present(
            payload,
            "fast_tail_max_local_score",
            builder->greedy_tail_max_local_score);
    assign_if_present(
            payload,
            "greedy_tail_shift_lambda",
            builder->greedy_tail_shift_lambda);
    assign_if_present(
            payload,
            "fast_tail_shift_lambda",
            builder->greedy_tail_shift_lambda);
    assign_if_present(
            payload,
            "greedy_tail_seen_window",
            builder->greedy_tail_seen_window);
    assign_if_present(
            payload,
            "fast_tail_seen_window",
            builder->greedy_tail_seen_window);

    assign_if_present(payload, "chain_tail_iters", builder->chain_tail_iters);
    assign_if_present(
            payload, "chain_tail_patience", builder->chain_tail_patience);
    assign_if_present(
            payload,
            "chain_tail_eps_improve",
            builder->chain_tail_eps_improve);
    assign_if_present(
            payload, "chain_tail_eval_topk", builder->chain_tail_eval_topk);
    assign_if_present(
            payload, "chain_tail_shortlist_k", builder->chain_tail_shortlist_k);
    assign_if_present(
            payload, "chain_tail_donor_topk", builder->chain_tail_donor_topk);
    assign_if_present(
            payload, "chain_tail_recv_topk", builder->chain_tail_recv_topk);
    assign_if_present(
            payload,
            "chain_tail_dims_sample_per_group",
            builder->chain_tail_dims_sample_per_group);
    assign_if_present(
            payload,
            "chain_tail_suspicious_alpha",
            builder->chain_tail_suspicious_alpha);
    assign_if_present(
            payload, "chain_tail_n_seed_moves", builder->chain_tail_n_seed_moves);
    assign_if_present(
            payload,
            "chain_tail_receiver_topk_per_dim",
            builder->chain_tail_receiver_topk_per_dim);
    assign_if_present(
            payload, "chain_tail_max_depth", builder->chain_tail_max_depth);
    assign_if_present(
            payload,
            "chain_tail_max_local_score",
            builder->chain_tail_max_local_score);
    assign_if_present(
            payload,
            "chain_tail_prefix_slack",
            builder->chain_tail_prefix_slack);
    assign_if_present(
            payload, "chain_tail_seen_window", builder->chain_tail_seen_window);
    assign_if_present(
            payload,
            "chain_tail_fast_proxy_top_dims",
            builder->chain_tail_fast_proxy_top_dims);
    assign_if_present(
            payload,
            "chain_tail_fast_shortlist_mult",
            builder->chain_tail_fast_shortlist_mult);
    return builder;
}

}  // namespace

nlohmann::json load_json_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open config JSON: " + path.string());
    }
    nlohmann::json j;
    in >> j;
    if (!j.is_object()) {
        throw std::runtime_error("config JSON root must be an object: " + path.string());
    }
    return j;
}

bool should_auto_reuse_structure(
        const nlohmann::json& config,
        bool default_value) {
    return find_bool_option(
            config,
            {"builder", "structure_builder"},
            {"auto_reuse_structure", "reuse_existing_structure"},
            default_value);
}

bool apply_faiss_runtime_config(const nlohmann::json& config) {
    const nlohmann::json* section = find_section(
            config, {"faiss", "faiss_runtime", "faiss_runtime_config"});
    if (section == nullptr) {
        return false;
    }

    bool changed = false;
    auto assign_runtime_int = [&](std::string_view key, int& out) {
        const std::string key_str(key);
        if (section->contains(key_str)) {
            out = section->at(key_str).get<int>();
            changed = true;
        }
    };

    assign_runtime_int(
            "distance_compute_blas_threshold",
            faiss::distance_compute_blas_threshold);
    assign_runtime_int(
            "distance_compute_blas_query_bs",
            faiss::distance_compute_blas_query_bs);
    assign_runtime_int(
            "distance_compute_blas_database_bs",
            faiss::distance_compute_blas_database_bs);
    return changed;
}

std::shared_ptr<StructureBuilder> make_structure_builder_from_config(
        const nlohmann::json& config,
        const std::filesystem::path& config_dir) {
    const nlohmann::json* section =
            find_section(config, {"builder", "structure_builder"});
    if (section == nullptr) {
        return std::make_shared<RefinedStructureBuilder>();
    }

    const std::string type = section->value("type", "refined");
    const nlohmann::json& payload = resolve_builder_payload(*section, type);

    if (type == "fixed") {
        const auto raw_path =
                payload.contains("structure_path")
                ? std::filesystem::path(payload.at("structure_path").get<std::string>())
                : std::filesystem::path(payload.at("path").get<std::string>());
        auto structure = Structure::load_json(
                resolve_path(raw_path, config_dir).string());
        return std::make_shared<FixedStructureBuilder>(std::move(structure));
    }
    if (type == "balanced") {
        return make_builder_with_common_json<BalancedStructureBuilder>(payload);
    }
    if (type == "variance_aware") {
        return make_builder_with_common_json<VarianceAwareStructureBuilder>(payload);
    }
    if (type == "refined") {
        return make_builder_with_common_json<RefinedStructureBuilder>(payload);
    }

    throw std::runtime_error("unknown structure builder type in config: " + type);
}

void apply_index_training_config(
        IndexEPQ& index,
        const nlohmann::json& config) {
    const nlohmann::json* section =
            find_section(config, {"index", "index_epq"});
    if (section == nullptr) {
        return;
    }

    assign_if_present(*section, "min_bits", index.min_bits);
    assign_if_present(*section, "max_bits", index.max_bits);
    assign_if_present(*section, "kmeans_niter", index.kmeans_niter);
    assign_if_present(*section, "kmeans_nredo", index.kmeans_nredo);
    assign_if_present(
            *section, "use_uneven_transform", index.use_uneven_transform);

    const nlohmann::json* transform =
            find_section(*section, {"transform", "uneven_opq"});
    const nlohmann::json& transform_payload =
            transform != nullptr ? *transform : *section;

    assign_if_present(
            transform_payload, "transform_niter", index.transform_niter);
    assign_if_present(
            transform_payload,
            "transform_kmeans_niter",
            index.transform_kmeans_niter);
    assign_if_present(
            transform_payload,
            "transform_kmeans_nredo",
            index.transform_kmeans_nredo);
    assign_if_present(
            transform_payload, "transform_max_train", index.transform_max_train);
    assign_if_present(
            transform_payload, "transform_max_eval", index.transform_max_eval);
    assign_if_present(
            transform_payload, "transform_eval_frac", index.transform_eval_frac);
    assign_if_present(
            transform_payload, "transform_seed", index.transform_seed);
    assign_if_present(
            transform_payload,
            "transform_init_mode",
            index.transform_init_mode);
    assign_if_present(
            transform_payload,
            "transform_init_seed",
            index.transform_init_seed);
    assign_if_present(
            transform_payload,
            "transform_proxy_max_bits",
            index.transform_proxy_max_bits);
    assign_if_present(
            transform_payload,
            "transform_exact_polish_iters",
            index.transform_exact_polish_iters);
    assign_if_present(
            *section,
            "ivf_query_weighted_sampling",
            index.ivf_query_weighted_sampling);
    assign_if_present(
            *section,
            "ivf_query_weighted_sampling_base_mix",
            index.ivf_query_weighted_sampling_base_mix);
    assign_if_present(
            *section,
            "ivf_query_weighted_sampling_seed",
            index.ivf_query_weighted_sampling_seed);
    assign_if_present(
            *section,
            "ivf_query_weighted_sampling_rank_decay",
            index.ivf_query_weighted_sampling_rank_decay);
    assign_if_present(
            *section,
            "ivf_query_weighted_sampling_within_list_norm_alpha",
            index.ivf_query_weighted_sampling_within_list_norm_alpha);
}

}  // namespace epq
