#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <faiss/Index.h>

namespace epq {

using RowMatrixXf =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct AVQTrainingStats {
    double structure_time = 0.0;
    double preparation_time = 0.0;
    double codebook_time = 0.0;
    double total_time = 0.0;
};

class IndexAVQ : public faiss::Index {
   public:
    explicit IndexAVQ(int d = 0, int total_bits = 0);
    ~IndexAVQ() override;

    IndexAVQ(const IndexAVQ&) = delete;
    IndexAVQ& operator=(const IndexAVQ&) = delete;

    int total_bits = 0;
    int default_num_neighbors = 1000;
    int dimensions_per_block = 0;
    int training_threads = 0;
    int search_threads = 0;
    int search_batch_size = 256;
    float anisotropic_quantization_threshold = 0.2f;

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

    const AVQTrainingStats& training_stats() const noexcept;
    int effective_budget_bits() const noexcept;
    const RowMatrixXf& database() const noexcept;

   private:
    struct Impl;

    int resolve_dimensions_per_block() const;
    int resolve_effective_budget_bits(int dims_per_block) const;

    AVQTrainingStats training_stats_;
    RowMatrixXf train_sample_;
    RowMatrixXf database_;
    int effective_budget_bits_ = 0;
    Impl* impl_ = nullptr;
};

}  // namespace epq
