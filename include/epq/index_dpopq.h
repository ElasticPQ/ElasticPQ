#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <faiss/Index.h>
#include <nlohmann/json.hpp>

namespace epq {

using RowMatrixXf =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct DPOPQTrainingStats {
    double structure_time = 0.0;
    double preparation_time = 0.0;
    double codebook_time = 0.0;
    double total_time = 0.0;
};

class IndexDPOPQ : public faiss::Index {
   public:
    explicit IndexDPOPQ(int d = 0, int total_bits = 0);

    int total_bits = 0;
    int kmeans_niter = 25;
    int kmeans_nredo = 1;
    int dp_max_units = 0;
    bool block_alignment = false;

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
    const DPOPQTrainingStats& training_stats() const noexcept;
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
    int M_ = 0;
    Eigen::RowVectorXf mean_;
    RowMatrixXf pca_rotation_;
    RowMatrixXf base_rotation_;
    RowMatrixXf rotation_;
    RowMatrixXf inverse_transform_;
    std::vector<float> pca_eigenvalues_;
    std::vector<double> pca_partition_values_;
    std::vector<int> partition_units_;
    std::vector<float> eigenvalues_;
    std::vector<double> partition_values_;
    std::vector<float> transform_scales_;
    std::vector<int> pc_order_;
    std::vector<int> group_offsets_;
    std::vector<RowMatrixXf> codebooks_;
    DPOPQTrainingStats stats_;
    double partition_cost_ = 0.0;
    double partition_units_scale_ = 1.0;
    int partition_units_sum_ = 0;
    bool partition_units_exact_ = true;

    void validate_config() const;
    RowMatrixXf as_matrix(faiss::idx_t n, const float* x) const;
    void train_pca_rotation(const RowMatrixXf& xt);
    void prepare_partition_weights();
    std::vector<int> choose_balanced_subset_dp(
            const std::vector<int>& items,
            int take,
            int target_units) const;
    std::vector<std::vector<int>> partition_recursive(
            const std::vector<int>& items,
            int groups) const;
    void solve_dp_partition();
    void configure_block_alignment();
    RowMatrixXf apply_transform(const RowMatrixXf& x) const;
    static RowMatrixXf train_kmeans(
            const RowMatrixXf& x,
            int k,
            int niter,
            int nredo);
    static float l2_distance(const float* a, const float* b, int dim);
};

}  // namespace epq
