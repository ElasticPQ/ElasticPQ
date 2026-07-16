#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <faiss/Index.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/io.h>

namespace epq {

using RowMatrixXf =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct BAPQTrainingStats {
    double structure_time = 0.0;
    double preparation_time = 0.0;
    double codebook_time = 0.0;
    double total_time = 0.0;
};

class IndexBAPQ : public faiss::Index {
   public:
    explicit IndexBAPQ(int d = 0, int total_bits = 0, int subspace_dim = 4);

    int total_bits = 0;
    int subspace_dim = 4;
    int bmax = 12;

    int seed = 123;
    int max_train_rows = 200000;
    int pca_max_train_rows = 200000;
    int kmeans_niter = 20;
    int kmeans_nredo = 1;
    int query_batch = 8;
    int db_chunk = 65536;

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
    void reconstruct_rows(
            const std::vector<faiss::idx_t>& ids,
            RowMatrixXf& out) const;
    void rerank_candidates(
            const float* query,
            const faiss::idx_t* candidate_ids,
            size_t ncandidates,
            faiss::idx_t k,
            float* distances,
            faiss::idx_t* labels) const;

    int component_count() const noexcept;
    int active_component_count() const noexcept;
    const std::vector<int>& nbits_per_group() const noexcept;
    const std::vector<int>& group_sizes() const noexcept;
    const BAPQTrainingStats& training_stats() const noexcept;

    size_t theoretical_code_bytes() const noexcept;
    size_t codebook_bytes() const noexcept;
    size_t transform_bytes() const noexcept;
    void serialize_payload(faiss::IOWriter& writer) const;
    size_t serialized_payload_bytes() const;
    size_t adc_lut_size() const noexcept;
    void compute_adc_lut(const float* query, float* lut) const;
    void transform_vector(const float* x, float* out) const;
    void compute_adc_lut_from_transformed(
            const float* query_transformed,
            float* lut) const;
    float adc_distance_from_packed_code(
            const uint8_t* code,
            const float* lut) const;

   private:
    struct GroupModel {
        int begin = 0;
        int size = 0;
        int nbits = 0;
        int ksub = 1;
        bool active = false;
        int active_slot = -1;
        RowMatrixXf codebook;
        std::vector<float> centroid_norms;
    };

    int component_count_ = 0;
    std::vector<int> nbits_per_group_;
    std::vector<int> group_sizes_;
    std::vector<GroupModel> groups_;
    std::vector<int> active_groups_;
    std::unique_ptr<faiss::PCAMatrix> pca_;
    BAPQTrainingStats training_stats_;

    // Active codes in group-major layout: [active_group][vector_id].
    std::vector<uint16_t> codes_;
    size_t code_capacity_ = 0;

    void validate_config() const;
    RowMatrixXf sample_rows(
            const RowMatrixXf& x,
            int max_rows,
            uint32_t seed_offset) const;
    RowMatrixXf apply_transform(const RowMatrixXf& x) const;
    void apply_transform_noalloc(
            faiss::idx_t n,
            const float* x,
            float* out) const;
    void assign_active_group_codes_transformed(
            faiss::idx_t n,
            const float* x_transformed,
            size_t active_group_index,
            uint16_t* dst) const;
};

}  // namespace epq
