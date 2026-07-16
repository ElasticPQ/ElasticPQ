#include "epq/benchmark_ivf_wrappers.h"

#include <faiss/VectorTransform.h>
#include <faiss/utils/utils.h>

#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>

namespace {

template <typename IndexT, typename StatsT>
void train_timed_ivf(IndexT& index, StatsT& stats, faiss::idx_t n, const float* x) {
    const auto total_t0 = std::chrono::steady_clock::now();
    if (index.verbose) {
        std::printf("Training level-1 quantizer\n");
    }

    const auto coarse_t0 = std::chrono::steady_clock::now();
    index.train_q1(n, x, index.verbose, index.metric_type);
    stats.coarse_train_time =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - coarse_t0)
                    .count();

    if (index.verbose) {
        std::printf("Training IVF residual\n");
    }

    faiss::idx_t max_nt = index.train_encoder_num_vectors();
    if (max_nt <= 0) {
        max_nt = static_cast<faiss::idx_t>((size_t)1 << 35);
    }

    faiss::TransformedVectors tv(
            x,
            faiss::fvecs_maybe_subsample(index.d, (size_t*)&n, max_nt, x, index.verbose));

    const auto encoder_t0 = std::chrono::steady_clock::now();
    if (index.by_residual) {
        std::vector<faiss::idx_t> assign(static_cast<size_t>(n));
        index.quantizer->assign(n, tv.x, assign.data());

        std::vector<float> residuals(
                static_cast<size_t>(n) * static_cast<size_t>(index.d));
        index.quantizer->compute_residual_n(n, tv.x, residuals.data(), assign.data());

        index.train_encoder(n, residuals.data(), assign.data());
    } else {
        index.train_encoder(n, tv.x, nullptr);
    }
    stats.encoder_train_time =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - encoder_t0)
                    .count();

    index.is_trained = true;
    stats.total_train_time =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - total_t0)
                    .count();
}

}  // namespace

namespace epq {

void TimedIndexIVFPQ::train(faiss::idx_t n, const float* x) {
    train_timed_ivf(*this, stats_, n, x);
}

const TimedIVFPQTrainStats& TimedIndexIVFPQ::train_stats() const {
    return stats_;
}

void TimedIndexIVFRaBitQ::train(faiss::idx_t n, const float* x) {
    train_timed_ivf(*this, stats_, n, x);
}

const TimedIVFRaBitQTrainStats& TimedIndexIVFRaBitQ::train_stats() const {
    return stats_;
}

void TimedIndexIVFResidualQuantizer::train(faiss::idx_t n, const float* x) {
    train_timed_ivf(*this, stats_, n, x);
}

const TimedIVFAQTrainStats& TimedIndexIVFResidualQuantizer::train_stats() const {
    return stats_;
}

void TimedIndexIVFLocalSearchQuantizer::train(faiss::idx_t n, const float* x) {
    train_timed_ivf(*this, stats_, n, x);
}

const TimedIVFAQTrainStats& TimedIndexIVFLocalSearchQuantizer::train_stats() const {
    return stats_;
}

uint8_t resolve_rabitq_nb_bits(int d, int total_bits) {
    faiss::RaBitQuantizer probe(static_cast<size_t>(d), faiss::METRIC_L2, 1);
    uint8_t chosen = 1;
    for (uint8_t nb_bits = 1; nb_bits <= 9; ++nb_bits) {
        const size_t code_bits =
                probe.compute_code_size(static_cast<size_t>(d), nb_bits) * 8;
        if (code_bits <= static_cast<size_t>(total_bits)) {
            chosen = nb_bits;
        }
    }
    return chosen;
}

void TimedIndexPreTransform::train(faiss::idx_t n, const float* x) {
    const auto total_t0 = std::chrono::steady_clock::now();
    int last_untrained = 0;
    if (!index->is_trained) {
        last_untrained = static_cast<int>(chain.size());
    } else {
        for (int i = static_cast<int>(chain.size()) - 1; i >= 0; --i) {
            if (!chain[static_cast<size_t>(i)]->is_trained) {
                last_untrained = i;
                break;
            }
        }
    }
    const float* prev_x = x;
    std::unique_ptr<const float[]> del;

    if (verbose) {
        std::printf(
                "IndexPreTransform::train: training chain 0 to %d\n",
                last_untrained);
    }

    for (int i = 0; i <= last_untrained; ++i) {
        if (i < static_cast<int>(chain.size())) {
            faiss::VectorTransform* ltrans = chain[static_cast<size_t>(i)];
            if (!ltrans->is_trained) {
                if (verbose) {
                    std::printf(
                            "   Training chain component %d/%zu\n",
                            i,
                            chain.size());
                    if (auto* opqm = dynamic_cast<faiss::OPQMatrix*>(ltrans)) {
                        opqm->verbose = true;
                    }
                }
                const auto t0 = std::chrono::steady_clock::now();
                ltrans->train(n, prev_x);
                stats_.transform_train_time += std::chrono::duration<double>(
                                                       std::chrono::steady_clock::now() -
                                                       t0)
                                                       .count();
            }
        } else {
            if (verbose) {
                std::printf("   Training sub-index\n");
            }
            const auto t0 = std::chrono::steady_clock::now();
            index->train(n, prev_x);
            stats_.sub_index_train_time += std::chrono::duration<double>(
                                                   std::chrono::steady_clock::now() - t0)
                                                   .count();
        }
        if (i == last_untrained) {
            break;
        }
        if (verbose) {
            std::printf("   Applying transform %d/%zu\n", i, chain.size());
        }

        const auto apply_t0 = std::chrono::steady_clock::now();
        float* xt = chain[static_cast<size_t>(i)]->apply(n, prev_x);
        stats_.transform_apply_time += std::chrono::duration<double>(
                                               std::chrono::steady_clock::now() -
                                               apply_t0)
                                               .count();

        if (prev_x != x) {
            del.reset();
        }

        prev_x = xt;
        del.reset(xt);
    }

    is_trained = true;
    stats_.total_train_time =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - total_t0)
                    .count();
}

const TimedPreTransformTrainStats& TimedIndexPreTransform::train_stats() const {
    return stats_;
}

}  // namespace epq
