#pragma once

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>

#include <faiss/VectorTransform.h>
#include <nlohmann/json.hpp>

#include "epq/index_avq.h"
#include "epq/index_bapq.h"
#include "epq/index_epq.h"
#include "epq/structure_builder.h"

#if defined(__linux__)
#include <sys/utsname.h>
#endif

#ifndef EPQ_CMAKE_BUILD_TYPE
#define EPQ_CMAKE_BUILD_TYPE "unknown"
#endif

#ifndef EPQ_FAISS_TARGET_NAME
#define EPQ_FAISS_TARGET_NAME "unknown"
#endif

#ifndef EPQ_CMAKE_CXX_COMPILER_ID
#define EPQ_CMAKE_CXX_COMPILER_ID "unknown"
#endif

#ifndef EPQ_CMAKE_CXX_COMPILER_VERSION
#define EPQ_CMAKE_CXX_COMPILER_VERSION "unknown"
#endif

#ifndef EPQ_ENABLE_STRUCTURE_TRACE
#define EPQ_ENABLE_STRUCTURE_TRACE 0
#endif

namespace epq {

namespace benchmark_metadata {

inline std::string trim(std::string value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

inline std::optional<std::string> getenv_string(const char* name) {
    if (const char* value = std::getenv(name); value != nullptr) {
        return std::string(value);
    }
    return std::nullopt;
}

inline std::optional<std::string> read_prefixed_line(
        const std::filesystem::path& path,
        std::string_view prefix) {
    std::ifstream in(path);
    if (!in) {
        return std::nullopt;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind(prefix, 0) == 0) {
            return trim(line.substr(prefix.size()));
        }
    }
    return std::nullopt;
}

inline int count_cpu_list(std::string_view text) {
    int total = 0;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find(',', start);
        if (end == std::string_view::npos) {
            end = text.size();
        }
        const std::string token = trim(std::string(text.substr(start, end - start)));
        if (!token.empty()) {
            const size_t dash = token.find('-');
            if (dash == std::string::npos) {
                ++total;
            } else {
                const int lo = std::stoi(token.substr(0, dash));
                const int hi = std::stoi(token.substr(dash + 1));
                if (hi >= lo) {
                    total += hi - lo + 1;
                }
            }
        }
        start = end + 1;
    }
    return total;
}

inline std::string utc_now_iso8601() {
    const auto now = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &now);
#else
    gmtime_r(&now, &tm);
#endif
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buffer;
}

inline std::string simd_path() {
    std::string simd = "scalar";
#if defined(__AVX512F__)
    simd = "avx512f";
#elif defined(__AVX2__)
    simd = "avx2";
#elif defined(__AVX__)
    simd = "avx";
#elif defined(__SSE4_2__)
    simd = "sse4.2";
#endif
#if defined(__FMA__)
    simd += "+fma";
#endif
    return simd;
}

inline std::string faiss_simd_hint() {
    const std::string target = EPQ_FAISS_TARGET_NAME;
    if (target.find("avx512") != std::string::npos) {
        return "avx512";
    }
    if (target.find("avx2") != std::string::npos) {
        return "avx2";
    }
    if (target.find("avx") != std::string::npos) {
        return "avx";
    }
    return "generic";
}

inline void copy_if_present(
        const nlohmann::json& src,
        nlohmann::json& dst,
        std::string_view key) {
    const std::string key_str(key);
    if (src.contains(key_str)) {
        dst[key_str] = src.at(key_str);
    }
}

inline const nlohmann::json* find_object(
        const nlohmann::json& src,
        std::string_view key) {
    const std::string key_str(key);
    if (!src.contains(key_str) || !src.at(key_str).is_object()) {
        return nullptr;
    }
    return &src.at(key_str);
}

inline nlohmann::json collect_hardware_metadata() {
    nlohmann::json meta;
    meta["timestamp_utc"] = utc_now_iso8601();
    meta["hardware_concurrency"] = std::thread::hardware_concurrency();

#if defined(__linux__)
    struct utsname uts {};
    if (uname(&uts) == 0) {
        meta["sysname"] = uts.sysname;
        meta["release"] = uts.release;
        meta["machine"] = uts.machine;
    }

    if (const auto cpu_model =
                read_prefixed_line("/proc/cpuinfo", "model name\t:");
        cpu_model.has_value()) {
        meta["cpu_model"] = *cpu_model;
    }
    if (const auto cpu_vendor =
                read_prefixed_line("/proc/cpuinfo", "vendor_id\t:");
        cpu_vendor.has_value()) {
        meta["cpu_vendor"] = *cpu_vendor;
    }
    if (const auto cpuset =
                read_prefixed_line("/proc/self/status", "Cpus_allowed_list:\t");
        cpuset.has_value()) {
        meta["cpuset"] = *cpuset;
        meta["cpuset_cpu_count"] = count_cpu_list(*cpuset);
    }
    if (const auto mems =
                read_prefixed_line("/proc/self/status", "Mems_allowed_list:\t");
        mems.has_value()) {
        meta["mems_allowed"] = *mems;
    }
#endif

    return meta;
}

inline nlohmann::json collect_build_metadata() {
    nlohmann::json meta;
    meta["build_type"] = EPQ_CMAKE_BUILD_TYPE;
    meta["faiss_target"] = EPQ_FAISS_TARGET_NAME;
    meta["compiler_id"] = EPQ_CMAKE_CXX_COMPILER_ID;
    meta["compiler_version"] = EPQ_CMAKE_CXX_COMPILER_VERSION;
    meta["compiler_version_macro"] = __VERSION__;
    meta["cxx_standard"] = __cplusplus;
    meta["compile_simd"] = simd_path();
    meta["faiss_simd_hint"] = faiss_simd_hint();
    meta["avq_enabled"] = static_cast<bool>(EPQ_ENABLE_AVQ);
    meta["structure_debug_level"] = EPQ_STRUCTURE_DEBUG_LEVEL;
    meta["structure_trace_enabled"] = static_cast<bool>(EPQ_ENABLE_STRUCTURE_TRACE);
#if defined(NDEBUG)
    meta["ndebug"] = true;
#else
    meta["ndebug"] = false;
#endif
    return meta;
}

inline nlohmann::json summarize_structure_builder(const epq::StructureBuilder& builder) {
    nlohmann::json meta;
    meta["name"] = builder.name();

    if (const auto* fixed = dynamic_cast<const epq::FixedStructureBuilder*>(&builder);
        fixed != nullptr) {
        meta["type"] = "fixed";
        meta["group_count"] = fixed->structure().group_count();
        meta["total_bits"] = fixed->structure().total_bits;
        return meta;
    }

    if (const auto* balanced =
                dynamic_cast<const epq::BalancedStructureBuilder*>(&builder);
        balanced != nullptr) {
        meta["type"] = "balanced";
        meta["target_groups"] = balanced->target_groups;
        meta["nominal_group_bits"] = balanced->nominal_group_bits;
        return meta;
    }

    if (const auto* variance =
                dynamic_cast<const epq::VarianceAwareStructureBuilder*>(&builder);
        variance != nullptr) {
        meta["type"] = "variance_aware";
        meta["alpha_groups"] = variance->alpha_groups;
        meta["min_groups"] = variance->min_groups;
        meta["max_groups"] = variance->max_groups;
        meta["target_groups"] = variance->target_groups;
        meta["corr_sample_rows"] = variance->corr_sample_rows;
        meta["abs_correlation"] = variance->abs_correlation;
        meta["size_penalty"] = variance->size_penalty;
        meta["seed"] = variance->seed;
        return meta;
    }

    if (const auto* refined =
                dynamic_cast<const epq::RefinedStructureBuilder*>(&builder);
        refined != nullptr) {
        meta["type"] = "refined";
        meta["use_grow"] = refined->use_grow;
        meta["use_crystallize"] = refined->use_crystallize;
        meta["use_mbeam"] = refined->use_mbeam;
        meta["use_greedy_tail"] = refined->use_greedy_tail;
        meta["use_chain_tail"] = refined->use_chain_tail;
        meta["seed"] = refined->seed;
        meta["proxy_max_train_rows"] = refined->proxy_max_train_rows;
        meta["proxy_max_eval_rows"] = refined->proxy_max_eval_rows;
        meta["proxy_eval_frac"] = refined->proxy_eval_frac;
        meta["proxy_kmeans_niter"] = refined->proxy_kmeans_niter;
        meta["proxy_kmeans_nredo"] = refined->proxy_kmeans_nredo;
        meta["proxy_min_points_per_centroid"] =
                refined->proxy_min_points_per_centroid;
        meta["proxy_cache_slices"] = refined->proxy_cache_slices;
        meta["proxy_max_d_cache"] = refined->proxy_max_d_cache;
        meta["proxy_max_slice_cache_bytes"] =
                refined->proxy_max_slice_cache_bytes;
        meta["proxy_pca_top_dims"] = refined->proxy_pca_top_dims;
        meta["proxy_max_pca_cache"] = refined->proxy_max_pca_cache;
        meta["grow_corr_adj_rows"] = refined->grow_corr_adj_rows;
        meta["grow_seed_topk"] = refined->grow_seed_topk;
        meta["grow_seed_pair"] = refined->grow_seed_pair;
        meta["crystallize_corr_adj_rows"] = refined->crystallize_corr_adj_rows;
        meta["crystallize_proxy_bits"] = refined->crystallize_proxy_bits;
        meta["crystallize_beam_width"] = refined->crystallize_beam_width;
        meta["mbeam_iters"] = refined->mbeam_iters;
        meta["mbeam_beam_width"] = refined->mbeam_beam_width;
        meta["greedy_tail_iters"] = refined->greedy_tail_iters;
        meta["chain_tail_iters"] = refined->chain_tail_iters;
        meta["chain_tail_n_seed_moves"] = refined->chain_tail_n_seed_moves;
        meta["chain_tail_max_depth"] = refined->chain_tail_max_depth;
        meta["chain_tail_fast_proxy_top_dims"] =
                refined->chain_tail_fast_proxy_top_dims;
        return meta;
    }

    return meta;
}

inline nlohmann::json summarize_index_epq(
        const epq::IndexEPQ& index,
        std::string_view family = "epq") {
    nlohmann::json meta;
    meta["family"] = family;
    meta["total_bits"] = index.total_bits;
    meta["min_bits"] = index.min_bits;
    meta["max_bits"] = index.max_bits;
    meta["kmeans_niter"] = index.kmeans_niter;
    meta["kmeans_nredo"] = index.kmeans_nredo;
    meta["use_uneven_transform"] = index.use_uneven_transform;
    meta["transform_niter"] = index.transform_niter;
    meta["transform_kmeans_niter"] = index.transform_kmeans_niter;
    meta["transform_kmeans_nredo"] = index.transform_kmeans_nredo;
    meta["transform_max_train"] = index.transform_max_train;
    meta["transform_max_eval"] = index.transform_max_eval;
    meta["transform_eval_frac"] = index.transform_eval_frac;
    meta["transform_seed"] = index.transform_seed;
    meta["transform_init_mode"] = index.transform_init_mode;
    meta["transform_init_seed"] = index.transform_init_seed;
    if (index.transform_profile().used) {
        meta["transform_init_orthogonality_error"] =
                index.transform_profile().init_orthogonality_error;
    }
    meta["transform_proxy_max_bits"] = index.transform_proxy_max_bits;
    meta["transform_exact_polish_iters"] = index.transform_exact_polish_iters;
    meta["ivf_query_weighted_sampling"] = index.ivf_query_weighted_sampling;
    meta["ivf_query_weighted_sampling_base_mix"] =
            index.ivf_query_weighted_sampling_base_mix;
    meta["ivf_query_weighted_sampling_seed"] =
            index.ivf_query_weighted_sampling_seed;
    meta["ivf_query_weighted_sampling_rank_decay"] =
            index.ivf_query_weighted_sampling_rank_decay;
    meta["ivf_query_weighted_sampling_within_list_norm_alpha"] =
            index.ivf_query_weighted_sampling_within_list_norm_alpha;
    if (index.structure_builder) {
        meta["builder"] = summarize_structure_builder(*index.structure_builder);
    }
    return meta;
}

inline nlohmann::json summarize_index_bapq(
        const epq::IndexBAPQ& index,
        std::string_view family = "bapq") {
    nlohmann::json meta;
    meta["family"] = family;
    meta["total_bits"] = index.total_bits;
    meta["subspace_dim"] = index.subspace_dim;
    meta["bmax"] = index.bmax;
    meta["seed"] = index.seed;
    meta["max_train_rows"] = index.max_train_rows;
    meta["pca_max_train_rows"] = index.pca_max_train_rows;
    meta["kmeans_niter"] = index.kmeans_niter;
    meta["kmeans_nredo"] = index.kmeans_nredo;
    meta["query_batch"] = index.query_batch;
    meta["db_chunk"] = index.db_chunk;
    return meta;
}

inline nlohmann::json summarize_index_avq(
        const epq::IndexAVQ& index,
        std::string_view family = "avq") {
    nlohmann::json meta;
    meta["family"] = family;
    meta["total_bits"] = index.total_bits;
    meta["effective_bits"] = index.effective_budget_bits();
    meta["default_num_neighbors"] = index.default_num_neighbors;
    meta["dimensions_per_block"] = index.dimensions_per_block;
    meta["training_threads"] = index.training_threads;
    meta["search_threads"] = index.search_threads;
    meta["search_batch_size"] = index.search_batch_size;
    meta["anisotropic_quantization_threshold"] =
            index.anisotropic_quantization_threshold;
    return meta;
}

inline nlohmann::json default_opq_metadata(int d, int M, int d2) {
    faiss::OPQMatrix opq(d, M, d2);
    nlohmann::json meta;
    meta["d_in"] = d;
    meta["d_out"] = d2;
    meta["M"] = M;
    meta["niter"] = opq.niter;
    meta["niter_pq"] = opq.niter_pq;
    meta["niter_pq_0"] = opq.niter_pq_0;
    meta["max_train_points"] = opq.max_train_points;
    meta["verbose"] = opq.verbose;
    return meta;
}

inline nlohmann::json collect_thread_metadata(
        int requested_threads,
        int effective_threads) {
    nlohmann::json meta;
    meta["requested_threads"] = requested_threads;
    meta["effective_threads"] = effective_threads;
    for (const char* key : {
                 "OMP_NUM_THREADS",
                 "OMP_DYNAMIC",
                 "OPENBLAS_NUM_THREADS",
                 "GOTO_NUM_THREADS",
                 "MKL_NUM_THREADS",
                 "MKL_DYNAMIC",
                 "BLIS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS",
                 "NUMEXPR_NUM_THREADS",
             }) {
        if (const auto value = getenv_string(key); value.has_value()) {
            meta[key] = *value;
        }
    }
    return meta;
}

inline nlohmann::json summarize_epq_config(const nlohmann::json& config) {
    nlohmann::json meta;

    if (const auto* faiss = find_object(config, "faiss"); faiss != nullptr) {
        nlohmann::json faiss_meta;
        copy_if_present(*faiss, faiss_meta, "distance_compute_blas_threshold");
        copy_if_present(*faiss, faiss_meta, "distance_compute_blas_query_bs");
        copy_if_present(*faiss, faiss_meta, "distance_compute_blas_database_bs");
        if (!faiss_meta.empty()) {
            meta["faiss"] = std::move(faiss_meta);
        }
    }

    if (const auto* builder = find_object(config, "builder"); builder != nullptr) {
        nlohmann::json builder_meta;
        const std::string builder_type = builder->value("type", "unknown");
        builder_meta["type"] = builder_type;
        copy_if_present(*builder, builder_meta, "auto_reuse_structure");

        const auto* payload = builder;
        if (const auto* typed = find_object(*builder, builder_type); typed != nullptr) {
            payload = typed;
        }
        nlohmann::json params;
        for (const char* key : {
                     "use_grow",
                     "use_crystallize",
                     "use_mbeam",
                     "use_greedy_tail",
                     "use_chain_tail",
                     "seed",
                     "proxy_max_train_rows",
                     "proxy_max_eval_rows",
                     "proxy_eval_frac",
                     "proxy_kmeans_niter",
                     "proxy_kmeans_nredo",
                     "proxy_min_points_per_centroid",
                     "proxy_cache_slices",
                     "proxy_max_d_cache",
                     "proxy_max_slice_cache_bytes",
                     "proxy_pca_top_dims",
                     "grow_corr_adj_rows",
                     "crystallize_corr_adj_rows",
                     "crystallize_proxy_bits",
                     "crystallize_beam_width",
                     "mbeam_iters",
                     "mbeam_beam_width",
                     "greedy_tail_iters",
                     "chain_tail_iters",
                     "chain_tail_n_seed_moves",
                 }) {
            copy_if_present(*payload, params, key);
        }
        if (!params.empty()) {
            builder_meta["params"] = std::move(params);
        }
        meta["builder"] = std::move(builder_meta);
    }

    if (const auto* index = find_object(config, "index"); index != nullptr) {
        nlohmann::json index_meta;
        for (const char* key : {
                     "kmeans_niter",
                     "kmeans_nredo",
                 }) {
            copy_if_present(*index, index_meta, key);
        }
        if (const auto* transform = find_object(*index, "transform");
            transform != nullptr) {
            nlohmann::json transform_meta;
            for (const char* key : {
                         "transform_niter",
                         "transform_kmeans_niter",
                         "transform_kmeans_nredo",
                         "transform_max_train",
                         "transform_max_eval",
                         "transform_eval_frac",
                         "transform_seed",
                         "transform_init_mode",
                         "transform_init_seed",
                         "transform_proxy_max_bits",
                         "transform_exact_polish_iters",
                     }) {
                copy_if_present(*transform, transform_meta, key);
            }
            if (!transform_meta.empty()) {
                index_meta["transform"] = std::move(transform_meta);
            }
        }
        if (!index_meta.empty()) {
            meta["index"] = std::move(index_meta);
        }
    }

    return meta;
}

inline nlohmann::json build_common_benchmark_metadata(
        const nlohmann::json& run_meta,
        const nlohmann::json& dataset_meta,
        int requested_threads,
        int effective_threads,
        const nlohmann::json* config = nullptr) {
    nlohmann::json meta;
    meta["run"] = run_meta;
    meta["hardware"] = collect_hardware_metadata();
    meta["build"] = collect_build_metadata();
    meta["threads"] = collect_thread_metadata(requested_threads, effective_threads);
    meta["dataset"] = dataset_meta;
    if (config != nullptr) {
        meta["config"] = summarize_epq_config(*config);
    }
    return meta;
}

inline void print_common_benchmark_metadata(
        std::ostream& os,
        const nlohmann::json& meta) {
    for (const char* key : {"run", "hardware", "build", "threads", "dataset", "config"}) {
        if (meta.contains(key) && !meta.at(key).empty()) {
            os << "meta." << key << ' ' << meta.at(key).dump() << '\n';
        }
    }
}

}  // namespace benchmark_metadata

}  // namespace epq
