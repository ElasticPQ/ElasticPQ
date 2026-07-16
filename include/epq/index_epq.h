#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <faiss/Index.h>
#include <faiss/impl/io.h>

#include "epq/structure.h"
#include "epq/structure_builder.h"

namespace epq {

using RowMatrixXf =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ColMatrixXf =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

enum class SearchMode {
    kADC,
    kSDC,
};

struct TrainingStats {
    double structure_time = 0.0;
    double preparation_time = 0.0;
    double codebook_time = 0.0;
    double total_time = 0.0;
};

struct CodebookProfile {
    int group_index = 0;
    int ndims = 0;
    int nbits = 0;
    int ksub = 0;
    int train_rows = 0;
    double seconds = 0.0;
};

struct TransformIterationProfile {
    int iteration = 0;
    int train_rows = 0;
    int eval_rows = 0;
    bool proxy_stage = false;
    double codebook_time = 0.0;
    double quantize_time = 0.0;
    double procrustes_time = 0.0;
    double eval_time = 0.0;
    double objective = 0.0;
    bool objective_is_eval = false;
    double total_time = 0.0;
};

struct TransformProfile {
    bool used = false;
    std::string init_mode = "identity";
    int init_seed = 123;
    double init_orthogonality_error = 0.0;
    int train_rows = 0;
    int eval_rows = 0;
    int proxy_max_bits = 0;
    int exact_polish_iters = 0;
    int proxy_iterations = 0;
    int exact_iterations = 0;
    int iterations_run = 0;
    double total_time = 0.0;
    bool has_final_holdout = false;
    double final_holdout_mse = 0.0;
    double final_holdout_seconds = 0.0;
    std::vector<TransformIterationProfile> iterations;
};

struct RuntimeProfile {
    int last_add_rows = 0;
    double last_add_total_time = 0.0;
    double last_add_transform_time = 0.0;
    double last_add_assign_time = 0.0;

    SearchMode last_search_mode = SearchMode::kADC;
    int last_search_queries = 0;
    int last_search_k = 0;
    double last_search_total_time = 0.0;
    double last_search_transform_time = 0.0;
    double last_search_lut_time = 0.0;
    double last_search_scan_time = 0.0;
};

struct SearchParametersEPQ : faiss::SearchParameters {
    SearchMode mode = SearchMode::kADC;
};

struct GroupSpan {
    int begin = 0;
    int size = 0;
    bool contiguous = false;
};

class IndexEPQ : public faiss::Index {
   public:
    explicit IndexEPQ(
            int d = 0,
            int total_bits = 0,
            std::shared_ptr<StructureBuilder> structure_builder = nullptr);

    int total_bits;
    int min_bits = 0;
    int max_bits = 12;

    int kmeans_niter = 25;
    int kmeans_nredo = 1;

    bool use_uneven_transform = true;
    int transform_niter = 0;
    int transform_kmeans_niter = 15;
    int transform_kmeans_nredo = 1;
    int transform_max_train = 65536;
    int transform_max_eval = 16384;
    float transform_eval_frac = 0.2f;
    int transform_seed = 123;
    std::string transform_init_mode = "identity";
    int transform_init_seed = 123;
    int transform_proxy_max_bits = 8;
    int transform_exact_polish_iters = 1;

    bool ivf_query_weighted_sampling = false;
    float ivf_query_weighted_sampling_base_mix = 0.2f;
    int ivf_query_weighted_sampling_seed = 123;
    std::string ivf_query_weighted_sampling_rank_decay = "none";
    float ivf_query_weighted_sampling_within_list_norm_alpha = 0.0f;

    std::shared_ptr<StructureBuilder> structure_builder;

    void train(faiss::idx_t n, const float* x) override;
    void add(faiss::idx_t n, const float* x) override;
    void add_with_ids(faiss::idx_t n, const float* x, const faiss::idx_t* xids) override;
    void search(
            faiss::idx_t n,
            const float* x,
            faiss::idx_t k,
            float* distances,
            faiss::idx_t* labels,
            const faiss::SearchParameters* params = nullptr) const override;
    void reset() override;
    void reconstruct(faiss::idx_t key, float* recons) const override;

    size_t sa_code_size() const override;
    void sa_encode(faiss::idx_t n, const float* x, uint8_t* bytes) const override;
    void sa_decode(faiss::idx_t n, const uint8_t* bytes, float* x) const override;
    size_t adc_lut_size() const noexcept;
    size_t ivf_lut_build_work() const noexcept;
    size_t ivf_default_lut_min_list_size() const noexcept;
    void transform_vector(const float* x, float* out) const;
    void compute_adc_lut_from_transformed(const float* transformed_x, float* lut) const;
    void unpack_code_assignments(const uint8_t* code, uint16_t* assignments) const;
    float adc_distance_from_assignments(
            const uint16_t* assignments,
            const float* lut) const;
    float exact_distance_from_assignments_transformed(
            const uint16_t* assignments,
            const float* transformed_x) const;
    float exact_distance_from_packed_code_transformed(
            const uint8_t* code,
            const float* transformed_x) const;
    float adc_distance_from_packed_code(const uint8_t* code, const float* lut) const;

    const Structure& structure() const noexcept;
    const std::vector<std::vector<int>>& active_groups() const noexcept;
    const std::vector<RowMatrixXf>& codebooks() const noexcept;
    const TrainingStats& training_stats() const noexcept;
    const std::vector<CodebookProfile>& codebook_profiles() const noexcept;
    const TransformProfile& transform_profile() const noexcept;
    const RuntimeProfile& runtime_profile() const noexcept;
    SearchMode default_search_mode() const noexcept;
    void set_default_search_mode(SearchMode mode) noexcept;
    void serialize_payload(faiss::IOWriter& writer) const;
    size_t serialized_payload_bytes() const;

    std::vector<uint16_t> compute_assignments(faiss::idx_t n, const float* x) const;
    void decode_assignments(faiss::idx_t n, const uint16_t* assignments, float* x) const;

   private:
    SearchMode default_search_mode_ = SearchMode::kADC;
    Structure structure_;
    std::vector<std::vector<int>> active_groups_;
    std::vector<std::vector<int>> contiguous_groups_;
    std::vector<GroupSpan> active_group_spans_;
    bool all_active_groups_contiguous_ = false;
    std::vector<int> perm_;
    std::vector<int> inv_perm_;
    RowMatrixXf rotation_;
    bool has_transform_ = false;

    std::vector<RowMatrixXf> codebooks_;
    std::vector<ColMatrixXf> lut_codebooks_;
    std::vector<std::vector<float>> centroid_norms_;
    std::vector<RowMatrixXf> sdc_tables_;
    TrainingStats training_stats_;
    std::vector<CodebookProfile> codebook_profiles_;
    TransformProfile transform_profile_;
    mutable RuntimeProfile runtime_profile_;

    std::vector<uint16_t> database_codes_;
    std::vector<uint16_t> database_codes_by_group_;
    size_t database_code_capacity_ = 0;
    std::vector<uint8_t> packed_codes_;

    BuildContext make_build_context() const;
    void validate_runtime_config() const;
    std::vector<uint16_t> compute_assignments_impl(faiss::idx_t n, const float* x) const;
    void refresh_active_group_layout();
    void build_sdc_tables();
};

}  // namespace epq
