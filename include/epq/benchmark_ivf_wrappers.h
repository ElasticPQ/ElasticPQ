#pragma once

#include <cstdint>

#include <faiss/IndexIVFAdditiveQuantizer.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/IndexPreTransform.h>

namespace epq {

struct TimedIVFPQTrainStats {
    double coarse_train_time = 0.0;
    double encoder_train_time = 0.0;
    double total_train_time = 0.0;
};

struct TimedIVFRaBitQTrainStats {
    double coarse_train_time = 0.0;
    double encoder_train_time = 0.0;
    double total_train_time = 0.0;
};

struct TimedIVFAQTrainStats {
    double coarse_train_time = 0.0;
    double encoder_train_time = 0.0;
    double total_train_time = 0.0;
};

struct TimedPreTransformTrainStats {
    double transform_train_time = 0.0;
    double transform_apply_time = 0.0;
    double sub_index_train_time = 0.0;
    double total_train_time = 0.0;
};

class TimedIndexIVFPQ final : public faiss::IndexIVFPQ {
   public:
    using faiss::IndexIVFPQ::IndexIVFPQ;

    void train(faiss::idx_t n, const float* x) override;
    const TimedIVFPQTrainStats& train_stats() const;

   private:
    TimedIVFPQTrainStats stats_;
};

class TimedIndexIVFRaBitQ final : public faiss::IndexIVFRaBitQ {
   public:
    using faiss::IndexIVFRaBitQ::IndexIVFRaBitQ;

    void train(faiss::idx_t n, const float* x) override;
    const TimedIVFRaBitQTrainStats& train_stats() const;

   private:
    TimedIVFRaBitQTrainStats stats_;
};

class TimedIndexIVFResidualQuantizer final
        : public faiss::IndexIVFResidualQuantizer {
   public:
    using faiss::IndexIVFResidualQuantizer::IndexIVFResidualQuantizer;

    void train(faiss::idx_t n, const float* x) override;
    const TimedIVFAQTrainStats& train_stats() const;

   private:
    TimedIVFAQTrainStats stats_;
};

class TimedIndexIVFLocalSearchQuantizer final
        : public faiss::IndexIVFLocalSearchQuantizer {
   public:
    using faiss::IndexIVFLocalSearchQuantizer::IndexIVFLocalSearchQuantizer;

    void train(faiss::idx_t n, const float* x) override;
    const TimedIVFAQTrainStats& train_stats() const;

   private:
    TimedIVFAQTrainStats stats_;
};

class TimedIndexPreTransform final : public faiss::IndexPreTransform {
   public:
    using faiss::IndexPreTransform::IndexPreTransform;

    void train(faiss::idx_t n, const float* x) override;
    const TimedPreTransformTrainStats& train_stats() const;

   private:
    TimedPreTransformTrainStats stats_;
};

uint8_t resolve_rabitq_nb_bits(int d, int total_bits);

}  // namespace epq
