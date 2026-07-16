#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <faiss/Index.h>
#include <nlohmann/json.hpp>

namespace epq {

struct VAQTrainingStats {
    double structure_time = 0.0;
    double preparation_time = 0.0;
    double codebook_time = 0.0;
    double total_time = 0.0;
};

// Faiss-compatible adapter around TheDatumOrg/VAQ. The upstream model learns
// the PCA, variance-aware bit allocation, and centroids. This class supplies
// compact variable-width codes and the codec operations needed by IVF.
class IndexVAQ : public faiss::Index {
   public:
    explicit IndexVAQ(
            int d = 0,
            int total_bits = 0,
            int subspaces = 0,
            int min_bits_per_subspace = 1,
            int max_bits_per_subspace = 8,
            float variance_fraction = 1.0f);
    ~IndexVAQ() override;

    IndexVAQ(const IndexVAQ&) = delete;
    IndexVAQ& operator=(const IndexVAQ&) = delete;

    int total_bits = 0;
    int subspaces = 0;
    int min_bits_per_subspace = 1;
    int max_bits_per_subspace = 8;
    float variance_fraction = 1.0f;

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
    size_t sa_code_size() const override;
    void sa_encode(faiss::idx_t n, const float* x, uint8_t* bytes) const override;
    void sa_decode(faiss::idx_t n, const uint8_t* bytes, float* x) const override;

    int component_count() const noexcept;
    const std::vector<int>& bit_allocation() const;
    const VAQTrainingStats& training_stats() const noexcept;
    size_t serialized_payload_bytes() const;
    size_t adc_lut_size() const noexcept;
    void transform_vector(const float* x, float* out) const;
    void compute_adc_lut_from_transformed(
            const float* query_transformed,
            float* lut) const;
    float adc_distance_from_packed_code(
            const uint8_t* code,
            const float* lut) const;
    nlohmann::json metadata() const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::vector<uint8_t> codes_;
    VAQTrainingStats stats_;

    void validate_config() const;
};

}  // namespace epq
