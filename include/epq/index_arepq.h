#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <faiss/Index.h>

#include "epq/index_epq.h"

namespace epq {

struct TailMemoryStats {
    size_t payload_code_bytes = 0;
    size_t resident_flat_code_bytes = 0;
    size_t serialized_codebook_bytes = 0;
    size_t reconstruction_codebook_bytes = 0;
    size_t transform_copy_bytes = 0;
    size_t norm_table_entries = 0;
    size_t norm_table_bytes = 0;
    size_t product_tail_table_entries = 0;
    size_t product_tail_table_bytes = 0;
    size_t tail_pair_table_entries = 0;
    size_t tail_pair_table_bytes = 0;
    size_t query_lut_entries_per_query = 0;
    size_t query_lut_bytes_per_query = 0;

    size_t serialized_tail_bytes() const noexcept;
    size_t resident_search_model_bytes() const noexcept;
    size_t resident_auxiliary_table_bytes() const noexcept;
    size_t resident_model_bytes() const noexcept;
};

class IndexAREPQ final : public faiss::Index {
   public:
    explicit IndexAREPQ(
            int d = 0,
            int total_bits = 0,
            int tail_bits = 8,
            int tail_stages = 1,
            std::shared_ptr<StructureBuilder> structure_builder = nullptr);

    int total_bits = 0;
    int tail_bits = 8;
    int tail_stages = 1;
    int main_bits = 0;
    int tail_ksub = 0;

    int icm_iters = 2;
    bool final_main_reassign = false;
    bool skip_stable_tail_reassign = true;
    int tail_alt_iters = 1;
    float tail_alt_update_weight = 0.5f;
    int tail_kmeans_niter = 25;
    int tail_kmeans_nredo = 1;
    int tail_beam_candidates = 1;
    int add_batch_rows = 100000;
    int search_query_batch = 4;
    int search_db_chunk = 65536;

    IndexEPQ& main_index() noexcept;
    const IndexEPQ& main_index() const noexcept;

    int component_count() const;
    int effective_budget_bits() const noexcept;
    const TrainingStats& training_stats() const noexcept;
    double tail_train_time() const noexcept;
    double tail_alt_initial_mse() const noexcept;
    double tail_alt_best_mse() const noexcept;
    double tail_alt_final_mse() const noexcept;

    void train(faiss::idx_t n, const float* x) override;
    void add(faiss::idx_t n, const float* x) override;
    void search(
            faiss::idx_t n,
            const float* x,
            faiss::idx_t k,
            float* distances,
            faiss::idx_t* labels,
            const faiss::SearchParameters* params = nullptr) const override;
    void reset() override;
    void reconstruct(faiss::idx_t key, float* recons) const override;
    void reconstruct_rows(
            const std::vector<faiss::idx_t>& ids,
            RowMatrixXf& out) const;

    size_t sa_code_size() const override;
    void sa_encode(faiss::idx_t n, const float* x, uint8_t* bytes) const override;
    void sa_decode(faiss::idx_t n, const uint8_t* bytes, float* x) const override;
    size_t adc_lut_size() const noexcept;
    void transform_vector(const float* x, float* out) const;
    void compute_adc_lut_from_transformed(const float* transformed_x, float* lut) const;
    float adc_distance_from_packed_code(const uint8_t* code, const float* lut) const;

    size_t serialized_payload_bytes() const;
    TailMemoryStats tail_memory_stats() const noexcept;

   private:
    void validate_config() const;
    void encode_batch_joint(
            const RowMatrixXf& x_batch,
            std::vector<uint16_t>& main_codes,
            std::vector<std::vector<uint16_t>>& tail_codes) const;
    void encode_transformed_batch_joint(
            const RowMatrixXf& y,
            const std::vector<uint16_t>& initial_main_codes,
            std::vector<uint16_t>& main_codes,
            std::vector<std::vector<uint16_t>>& tail_codes) const;
    void decode_tail_sum(
            const std::vector<std::vector<uint16_t>>& tail_codes,
            RowMatrixXf& out) const;
    void decode_tail_sum_except(
            const std::vector<std::vector<uint16_t>>& tail_codes,
            int excluded_stage,
            RowMatrixXf& out) const;
    void build_tail_auxiliary_tables();
    void update_tail_codebooks_from_assignments(
            const RowMatrixXf& y,
            const std::vector<uint16_t>& main_codes,
            const std::vector<std::vector<uint16_t>>& tail_codes,
            float update_weight);
    double additive_mse(
            const RowMatrixXf& y,
            const std::vector<uint16_t>& main_codes,
            const std::vector<std::vector<uint16_t>>& tail_codes) const;
    void run_tail_alternating_optimization(
            const RowMatrixXf& train_y,
            const std::vector<uint16_t>& initial_main_codes);
    void refine_single_tail_beam(
            const RowMatrixXf& y,
            std::vector<uint16_t>& main_codes,
            std::vector<std::vector<uint16_t>>& tail_codes) const;
    void add_tail_terms_to_chunk(
            faiss::idx_t qb,
            faiss::idx_t b0,
            faiss::idx_t csz,
            const std::vector<std::vector<float>>& tail_luts,
            std::vector<float>& dist_chunk) const;

    IndexEPQ main_;
    TrainingStats training_stats_;
    double tail_train_time_ = 0.0;
    double tail_alt_initial_mse_ = std::numeric_limits<double>::quiet_NaN();
    double tail_alt_best_mse_ = std::numeric_limits<double>::quiet_NaN();
    double tail_alt_final_mse_ = std::numeric_limits<double>::quiet_NaN();
    RowMatrixXf transform_matrix_;
    std::vector<RowMatrixXf> tail_codebooks_y_;
    std::vector<RowMatrixXf> tail_codebooks_original_;
    std::vector<std::vector<float>> tail_norms_;
    std::vector<std::vector<std::vector<float>>> cross_tables_;
    std::vector<std::vector<std::vector<float>>> tail_pair_tables_;
    std::vector<uint16_t> main_codes_by_group_;
    std::vector<std::vector<uint16_t>> tail_codes_by_stage_;
};

}  // namespace epq
