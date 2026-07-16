#include "epq/index_epq.h"
#include "epq/serialization_size.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/LU>
#include <Eigen/QR>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/Heap.h>

namespace epq {
namespace {

extern "C" {

#ifndef FINTEGER
#define FINTEGER int
#endif

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);

int sgesvd_(
        const char* jobu,
        const char* jobvt,
        FINTEGER* m,
        FINTEGER* n,
        float* a,
        FINTEGER* lda,
        float* s,
        float* u,
        FINTEGER* ldu,
        float* vt,
        FINTEGER* ldvt,
        float* work,
        FINTEGER* lwork,
        FINTEGER* info);

}  // extern "C"

struct BlockRange {
    int begin = 0;
    int end = 0;
};

struct TrainEvalSplit {
    RowMatrixXf fit;
    RowMatrixXf eval;
    bool has_eval = false;
};

std::vector<int> inverse_permutation(const std::vector<int>& perm) {
    std::vector<int> inv(perm.size(), -1);
    for (size_t i = 0; i < perm.size(); ++i) {
        inv[static_cast<size_t>(perm[i])] = static_cast<int>(i);
    }
    return inv;
}

RowMatrixXf make_haar_orthogonal(int d, int seed) {
    ColMatrixXf gaussian(d, d);
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (Eigen::Index col = 0; col < gaussian.cols(); ++col) {
        for (Eigen::Index row = 0; row < gaussian.rows(); ++row) {
            gaussian(row, col) = normal(rng);
        }
    }

    Eigen::HouseholderQR<ColMatrixXf> qr(gaussian);
    ColMatrixXf q =
            qr.householderQ() * ColMatrixXf::Identity(d, d);
    const auto packed_qr = qr.matrixQR();
    for (int col = 0; col < d; ++col) {
        if (packed_qr(col, col) < 0.0f) {
            q.col(col) *= -1.0f;
        }
    }
    return RowMatrixXf(q);
}

RowMatrixXf make_initial_rotation(
        int d,
        const std::vector<int>& perm,
        const std::string& mode,
        int seed) {
    if (mode == "identity") {
        return RowMatrixXf::Identity(d, d);
    }

    const RowMatrixXf haar = make_haar_orthogonal(d, seed);
    if (mode == "haar_r") {
        return haar;
    }
    if (mode == "matched_physical") {
        if (perm.size() != static_cast<size_t>(d)) {
            throw std::invalid_argument(
                    "matched_physical transform initialization requires a full permutation");
        }
        RowMatrixXf rotation(d, d);
        for (int row = 0; row < d; ++row) {
            rotation.row(row) = haar.row(perm[static_cast<size_t>(row)]);
        }
        return rotation;
    }
    throw std::invalid_argument(
            "unknown transform_init_mode: " + mode);
}

double orthogonality_error(const RowMatrixXf& rotation) {
    const RowMatrixXf gram = rotation.transpose() * rotation;
    return static_cast<double>(
            (gram - RowMatrixXf::Identity(gram.rows(), gram.cols())).norm());
}

std::vector<BlockRange> make_blocks(const Structure& structure) {
    std::vector<BlockRange> blocks;
    blocks.reserve(structure.group_count());
    int offset = 0;
    for (const auto& group : structure.groups) {
        const int next = offset + static_cast<int>(group.dims.size());
        blocks.push_back({offset, next});
        offset = next;
    }
    return blocks;
}

Structure make_proxy_structure(const Structure& structure, int max_bits) {
    Structure proxy = structure;
    if (max_bits <= 0) {
        return proxy;
    }
    proxy.total_bits = 0;
    for (auto& group : proxy.groups) {
        group.nbits = std::min(group.nbits, max_bits);
        proxy.total_bits += group.nbits;
    }
    proxy.validate();
    return proxy;
}

RowMatrixXf gather_columns(
        const RowMatrixXf& x,
        const std::vector<int>& dims) {
    RowMatrixXf out(x.rows(), static_cast<Eigen::Index>(dims.size()));
    for (size_t j = 0; j < dims.size(); ++j) {
        out.col(static_cast<Eigen::Index>(j)) = x.col(dims[j]);
    }
    return out;
}

GroupSpan contiguous_span(const std::vector<int>& dims) {
    GroupSpan span;
    if (dims.empty()) {
        span.contiguous = true;
        return span;
    }
    span.begin = dims.front();
    span.size = static_cast<int>(dims.size());
    span.contiguous = true;
    for (size_t j = 1; j < dims.size(); ++j) {
        if (dims[j] != span.begin + static_cast<int>(j)) {
            span.contiguous = false;
            break;
        }
    }
    return span;
}

RowMatrixXf permute_columns(
        const RowMatrixXf& x,
        const std::vector<int>& perm) {
    RowMatrixXf out(x.rows(), x.cols());
    for (size_t j = 0; j < perm.size(); ++j) {
        out.col(static_cast<Eigen::Index>(j)) = x.col(perm[j]);
    }
    return out;
}

void unpermute_columns(
        const RowMatrixXf& x_perm,
        const std::vector<int>& perm,
        Eigen::Ref<RowMatrixXf> out) {
    for (size_t j = 0; j < perm.size(); ++j) {
        out.col(perm[j]) = x_perm.col(static_cast<Eigen::Index>(j));
    }
}

TrainEvalSplit split_rows(
        const RowMatrixXf& x,
        int max_train,
        int max_eval,
        float eval_frac,
        int seed) {
    TrainEvalSplit split;
    if (x.rows() <= 1) {
        split.fit = x;
        split.eval = x;
        split.has_eval = false;
        return split;
    }

    std::vector<int> ids(static_cast<size_t>(x.rows()));
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(ids.begin(), ids.end(), rng);

    int want_train = max_train > 0 ? std::min<int>(max_train, x.rows()) : x.rows();
    int want_eval = 0;
    if (max_eval > 0) {
        want_eval = std::min<int>(max_eval, std::max<int>(0, x.rows() - want_train));
    } else if (eval_frac > 0.0f) {
        want_eval = std::min<int>(
                std::max(1, static_cast<int>(std::lround(eval_frac * x.rows()))),
                std::max<int>(0, x.rows() - 1));
    }

    if (want_train + want_eval > x.rows()) {
        want_eval = std::max<int>(0, x.rows() - want_train);
    }
    if (want_train <= 0) {
        want_train = x.rows();
        want_eval = 0;
    }

    split.fit.resize(want_train, x.cols());
    for (int i = 0; i < want_train; ++i) {
        split.fit.row(i) = x.row(ids[static_cast<size_t>(i)]);
    }

    if (want_eval > 0) {
        split.eval.resize(want_eval, x.cols());
        for (int i = 0; i < want_eval; ++i) {
            split.eval.row(i) = x.row(ids[static_cast<size_t>(want_train + i)]);
        }
        split.has_eval = true;
    } else {
        split.eval = split.fit;
        split.has_eval = false;
    }
    return split;
}

RowMatrixXf orthogonal_procrustes(
        const RowMatrixXf& x,
        const RowMatrixXf& y) {
    using ColMatrixXf =
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    FAISS_THROW_IF_NOT_MSG(
            x.rows() == y.rows() && x.cols() == y.cols(),
            "orthogonal_procrustes requires equal-shaped matrices");

    const FINTEGER n = static_cast<FINTEGER>(x.rows());
    const FINTEGER d = static_cast<FINTEGER>(x.cols());
    if (n == 0 || d == 0) {
        return RowMatrixXf();
    }

    ColMatrixXf xty(d, d);
    {
        const float one = 1.0f;
        float zero = 0.0f;
        sgemm_("Not",
               "Transposed",
               const_cast<FINTEGER*>(&d),
               const_cast<FINTEGER*>(&d),
               const_cast<FINTEGER*>(&n),
               &one,
               x.data(),
               const_cast<FINTEGER*>(&d),
               y.data(),
               const_cast<FINTEGER*>(&d),
               &zero,
               xty.data(),
               const_cast<FINTEGER*>(&d));
    }

    ColMatrixXf u(d, d);
    ColMatrixXf vt(d, d);
    std::vector<float> sing(static_cast<size_t>(d));
    FINTEGER lwork = -1;
    FINTEGER info = -1;
    float workq = 0.0f;
    sgesvd_("All",
            "All",
            const_cast<FINTEGER*>(&d),
            const_cast<FINTEGER*>(&d),
            xty.data(),
            const_cast<FINTEGER*>(&d),
            sing.data(),
            u.data(),
            const_cast<FINTEGER*>(&d),
            vt.data(),
            const_cast<FINTEGER*>(&d),
            &workq,
            &lwork,
            &info);
    FAISS_THROW_IF_NOT_FMT(
            info == 0,
            "sgesvd workspace query failed in orthogonal_procrustes: info=%d",
            int(info));
    lwork = static_cast<FINTEGER>(workq);
    std::vector<float> work(static_cast<size_t>(lwork));
    sgesvd_("All",
            "All",
            const_cast<FINTEGER*>(&d),
            const_cast<FINTEGER*>(&d),
            xty.data(),
            const_cast<FINTEGER*>(&d),
            sing.data(),
            u.data(),
            const_cast<FINTEGER*>(&d),
            vt.data(),
            const_cast<FINTEGER*>(&d),
            work.data(),
            &lwork,
            &info);
    FAISS_THROW_IF_NOT_FMT(
            info == 0,
            "sgesvd failed in orthogonal_procrustes: info=%d",
            int(info));

    ColMatrixXf r = u * vt;
    if (r.determinant() < 0.0f) {
        u.col(u.cols() - 1) *= -1.0f;
        r.noalias() = u * vt;
    }
    return RowMatrixXf(r);
}

RowMatrixXf train_kmeans(
        const RowMatrixXf& x,
        int k,
        int niter,
        int nredo) {
    if (x.rows() <= 0 || x.cols() <= 0) {
        throw std::invalid_argument("epq::IndexEPQ: cannot train k-means on empty matrix");
    }
    const int effective_k = std::min<int>(k, x.rows());
    faiss::ClusteringParameters cp;
    cp.niter = niter;
    cp.nredo = nredo;
    cp.verbose = false;
    cp.min_points_per_centroid = 1;
    faiss::Clustering clustering(x.cols(), effective_k, cp);
    faiss::IndexFlatL2 assign_index(x.cols());
    clustering.train(x.rows(), x.data(), assign_index);

    RowMatrixXf centroids(k, x.cols());
    Eigen::Map<const RowMatrixXf> trained(
            clustering.centroids.data(),
            effective_k,
            x.cols());
    centroids.topRows(effective_k) = trained;
    for (int i = effective_k; i < k; ++i) {
        centroids.row(i) = trained.row((effective_k - 1 + i) % effective_k);
    }
    return centroids;
}

std::vector<faiss::idx_t> assign_codebook(
        const RowMatrixXf& x,
        const RowMatrixXf& codebook,
        std::vector<float>* distances = nullptr) {
    faiss::IndexFlatL2 assign_index(codebook.cols());
    assign_index.add(codebook.rows(), codebook.data());
    std::vector<faiss::idx_t> labels(static_cast<size_t>(x.rows()));
    std::vector<float> local_distances(static_cast<size_t>(x.rows()));
    assign_index.search(
            x.rows(),
            x.data(),
            1,
            local_distances.data(),
            labels.data());
    if (distances != nullptr) {
        *distances = std::move(local_distances);
    }
    return labels;
}

std::vector<float> centroid_norms(const RowMatrixXf& codebook) {
    std::vector<float> norms(static_cast<size_t>(codebook.rows()), 0.0f);
    for (Eigen::Index i = 0; i < codebook.rows(); ++i) {
        norms[static_cast<size_t>(i)] = codebook.row(i).squaredNorm();
    }
    return norms;
}

size_t grow_code_capacity(size_t current, size_t required) {
    size_t next = current == 0 ? size_t{1024}
                               : std::max(current + current / 2, current + size_t{1});
    while (next < required) {
        next = std::max(next + next / 2, next + size_t{1});
    }
    return next;
}

std::vector<RowMatrixXf> train_codebooks(
        const RowMatrixXf& x,
        const Structure& structure,
        const std::vector<std::vector<int>>& groups,
        int niter,
        int nredo,
        std::vector<CodebookProfile>* profiles = nullptr) {
    std::vector<RowMatrixXf> codebooks;
    codebooks.reserve(structure.group_count());
    if (profiles != nullptr) {
        profiles->clear();
        profiles->reserve(structure.group_count());
    }
    for (size_t gi = 0; gi < structure.group_count(); ++gi) {
        const auto group_t0 = std::chrono::steady_clock::now();
        const auto& dims = groups[gi];
        RowMatrixXf sub = gather_columns(x, dims);
        const int ksub = static_cast<int>(structure.groups[gi].ksub());
        if (structure.groups[gi].nbits == 0 || ksub <= 1 || sub.cols() == 0) {
            RowMatrixXf centroid(1, sub.cols());
            centroid.row(0) = sub.colwise().mean();
            codebooks.push_back(std::move(centroid));
        } else {
            codebooks.push_back(train_kmeans(sub, ksub, niter, nredo));
        }
        if (profiles != nullptr) {
            profiles->push_back(CodebookProfile{
                    .group_index = static_cast<int>(gi),
                    .ndims = static_cast<int>(dims.size()),
                    .nbits = structure.groups[gi].nbits,
                    .ksub = ksub,
                    .train_rows = static_cast<int>(sub.rows()),
                    .seconds = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - group_t0)
                                       .count(),
            });
        }
    }
    return codebooks;
}

RowMatrixXf quantize_with_codebooks(
        const RowMatrixXf& x,
        const Structure& structure,
        const std::vector<std::vector<int>>& groups,
        const std::vector<RowMatrixXf>& codebooks) {
    RowMatrixXf out(x.rows(), x.cols());
    out.setZero();
    for (size_t gi = 0; gi < structure.group_count(); ++gi) {
        const auto& dims = groups[gi];
        RowMatrixXf sub = gather_columns(x, dims);
        const auto& codebook = codebooks[gi];
        if (codebook.rows() == 1) {
            for (int row = 0; row < out.rows(); ++row) {
                for (size_t j = 0; j < dims.size(); ++j) {
                    out(row, dims[j]) = codebook(0, static_cast<Eigen::Index>(j));
                }
            }
            continue;
        }
        const auto labels = assign_codebook(sub, codebook);
        for (int row = 0; row < out.rows(); ++row) {
            const faiss::idx_t label = labels[static_cast<size_t>(row)];
            for (size_t j = 0; j < dims.size(); ++j) {
                out(row, dims[j]) = codebook(label, static_cast<Eigen::Index>(j));
            }
        }
    }
    return out;
}

float reconstruction_mse_with_codebooks(
        const RowMatrixXf& y_ref,
        const Structure& structure,
        const std::vector<BlockRange>& blocks,
        const std::vector<RowMatrixXf>& codebooks) {
    RowMatrixXf y_hat(y_ref.rows(), y_ref.cols());
    y_hat.setZero();
    for (size_t gi = 0; gi < structure.group_count(); ++gi) {
        const BlockRange block = blocks[gi];
        const RowMatrixXf sub =
                y_ref.block(0, block.begin, y_ref.rows(), block.end - block.begin);
        const auto labels = assign_codebook(sub, codebooks[gi]);
        for (int row = 0; row < y_ref.rows(); ++row) {
            y_hat.block(
                         row,
                         block.begin,
                         1,
                         block.end - block.begin) =
                    codebooks[gi].row(labels[static_cast<size_t>(row)]);
        }
    }
    return (y_ref - y_hat).array().square().mean();
}

int run_rotation_stage(
        const TrainEvalSplit& split,
        const Structure& structure,
        const std::vector<BlockRange>& blocks,
        RowMatrixXf& rotation,
        int max_iter,
        int kmeans_niter,
        int kmeans_nredo,
        bool use_eval_objective,
        bool proxy_stage,
        int patience,
        float min_delta,
        TransformProfile* profile) {
    if (max_iter <= 0) {
        return 0;
    }

    float best = std::numeric_limits<float>::infinity();
    int bad = 0;
    int ran = 0;
    for (int iter = 0; iter < max_iter; ++iter) {
        const auto iter_t0 = std::chrono::steady_clock::now();
        const RowMatrixXf y_fit = split.fit * rotation;
        const auto codebook_t0 = std::chrono::steady_clock::now();
        std::vector<RowMatrixXf> codebooks;
        codebooks.reserve(structure.group_count());
        for (size_t gi = 0; gi < structure.group_count(); ++gi) {
            const auto& group = structure.groups[gi];
            const BlockRange block = blocks[gi];
            RowMatrixXf sub =
                    y_fit.block(0, block.begin, y_fit.rows(), block.end - block.begin);
            const int ksub = static_cast<int>(group.ksub());
            if (group.nbits == 0 || ksub <= 1 || sub.cols() == 0) {
                RowMatrixXf centroid(1, sub.cols());
                centroid.row(0) = sub.colwise().mean();
                codebooks.push_back(std::move(centroid));
            } else {
                codebooks.push_back(
                        train_kmeans(sub, ksub, kmeans_niter, kmeans_nredo));
            }
        }
        const double codebook_time = std::chrono::duration<double>(
                                             std::chrono::steady_clock::now() - codebook_t0)
                                             .count();

        const auto quant_t0 = std::chrono::steady_clock::now();
        RowMatrixXf y_hat(y_fit.rows(), y_fit.cols());
        y_hat.setZero();
        for (size_t gi = 0; gi < structure.group_count(); ++gi) {
            const BlockRange block = blocks[gi];
            const RowMatrixXf sub =
                    y_fit.block(0, block.begin, y_fit.rows(), block.end - block.begin);
            const auto labels = assign_codebook(sub, codebooks[gi]);
            for (int row = 0; row < y_fit.rows(); ++row) {
                y_hat.block(
                             row,
                             block.begin,
                             1,
                             block.end - block.begin) =
                        codebooks[gi].row(labels[static_cast<size_t>(row)]);
            }
        }
        const double quantize_time = std::chrono::duration<double>(
                                             std::chrono::steady_clock::now() - quant_t0)
                                             .count();

        const auto proc_t0 = std::chrono::steady_clock::now();
        RowMatrixXf next = orthogonal_procrustes(split.fit, y_hat);
        const double procrustes_time = std::chrono::duration<double>(
                                               std::chrono::steady_clock::now() - proc_t0)
                                               .count();
        rotation = std::move(next);

        const auto eval_t0 = std::chrono::steady_clock::now();
        double objective = 0.0;
        bool objective_is_eval = false;
        double eval_time = 0.0;
        if (use_eval_objective) {
            const RowMatrixXf y_ref =
                    split.has_eval ? split.eval * rotation : y_fit;
            objective = reconstruction_mse_with_codebooks(
                    y_ref, structure, blocks, codebooks);
            eval_time = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - eval_t0)
                                .count();
            objective_is_eval = true;
            if (best - objective > min_delta) {
                best = static_cast<float>(objective);
                bad = 0;
            } else {
                ++bad;
            }
        } else {
            objective = (y_fit - y_hat).array().square().mean();
        }

        if (profile != nullptr) {
            profile->iterations.push_back(TransformIterationProfile{
                    .iteration = static_cast<int>(profile->iterations.size()) + 1,
                    .train_rows = static_cast<int>(split.fit.rows()),
                    .eval_rows = profile->eval_rows,
                    .proxy_stage = proxy_stage,
                    .codebook_time = codebook_time,
                    .quantize_time = quantize_time,
                    .procrustes_time = procrustes_time,
                    .eval_time = eval_time,
                    .objective = objective,
                    .objective_is_eval = objective_is_eval,
                    .total_time = std::chrono::duration<double>(
                                          std::chrono::steady_clock::now() - iter_t0)
                                          .count(),
            });
        }
        ++ran;
        if (use_eval_objective && bad >= patience) {
            break;
        }
    }
    return ran;
}

RowMatrixXf train_uneven_rotation(
        const RowMatrixXf& x_perm,
        const Structure& structure,
        const std::vector<int>& perm,
        int niter,
        int kmeans_niter,
        int kmeans_nredo,
        int max_train,
        int max_eval,
        float eval_frac,
        int seed,
        const std::string& init_mode,
        int init_seed,
        int proxy_max_bits,
        int exact_polish_iters,
        TransformProfile* profile = nullptr) {
    const auto split = split_rows(x_perm, max_train, max_eval, eval_frac, seed);
    RowMatrixXf rotation = make_initial_rotation(
            static_cast<int>(x_perm.cols()),
            perm,
            init_mode,
            init_seed);
    if (profile != nullptr) {
        profile->used = true;
        profile->init_mode = init_mode;
        profile->init_seed = init_seed;
        profile->init_orthogonality_error = orthogonality_error(rotation);
        profile->train_rows = static_cast<int>(split.fit.rows());
        profile->eval_rows = split.has_eval ? static_cast<int>(split.eval.rows()) : 0;
        profile->proxy_max_bits = proxy_max_bits;
        profile->exact_polish_iters = exact_polish_iters;
        profile->proxy_iterations = 0;
        profile->exact_iterations = 0;
        profile->iterations.clear();
        profile->iterations_run = 0;
        profile->total_time = 0.0;
    }
    const auto total_t0 = std::chrono::steady_clock::now();

    const int max_iter = niter > 0 ? niter : 128;
    const int patience = 5;
    const float min_delta = 1e-7f;

    const bool use_eval_objective = (niter <= 0);
    const bool use_proxy = proxy_max_bits > 0 && proxy_max_bits < structure.max_nbits();
    if (use_proxy) {
        const Structure proxy_structure = make_proxy_structure(structure, proxy_max_bits);
        const auto proxy_blocks = make_blocks(proxy_structure);
        const int proxy_iters = run_rotation_stage(
                split,
                proxy_structure,
                proxy_blocks,
                rotation,
                max_iter,
                kmeans_niter,
                kmeans_nredo,
                use_eval_objective,
                /*proxy_stage=*/true,
                patience,
                min_delta,
                profile);
        if (profile != nullptr) {
            profile->proxy_iterations = proxy_iters;
        }
    } else {
        const auto blocks = make_blocks(structure);
        const int exact_iters = run_rotation_stage(
                split,
                structure,
                blocks,
                rotation,
                max_iter,
                kmeans_niter,
                kmeans_nredo,
                use_eval_objective,
                /*proxy_stage=*/false,
                patience,
                min_delta,
                profile);
        if (profile != nullptr) {
            profile->exact_iterations = exact_iters;
        }
    }

    if (use_proxy && exact_polish_iters > 0) {
        const auto blocks = make_blocks(structure);
        const int exact_iters = run_rotation_stage(
                split,
                structure,
                blocks,
                rotation,
                exact_polish_iters,
                kmeans_niter,
                kmeans_nredo,
                use_eval_objective,
                /*proxy_stage=*/false,
                patience,
                min_delta,
                profile);
        if (profile != nullptr) {
            profile->exact_iterations = exact_iters;
        }
    }

    // Report a selection-safe final-codec diagnostic without changing the
    // trained rotation or the production codebooks.  Exact codebooks are fit
    // only on the rotation fit split and evaluated once on its disjoint
    // holdout split.  This lets architecture sweeps select a candidate without
    // consulting database reconstruction or query-retrieval test metrics.
    if (profile != nullptr && split.has_eval) {
        const auto holdout_t0 = std::chrono::steady_clock::now();
        const RowMatrixXf y_fit = split.fit * rotation;
        const RowMatrixXf y_eval = split.eval * rotation;
        const auto groups = structure.contiguous_groups();
        const auto codebooks = train_codebooks(
                y_fit,
                structure,
                groups,
                kmeans_niter,
                kmeans_nredo);
        profile->final_holdout_mse = reconstruction_mse_with_codebooks(
                y_eval, structure, make_blocks(structure), codebooks);
        profile->final_holdout_seconds = std::chrono::duration<double>(
                                                 std::chrono::steady_clock::now() -
                                                 holdout_t0)
                                                 .count();
        profile->has_final_holdout = true;
    }

    if (profile != nullptr) {
        profile->iterations_run = static_cast<int>(profile->iterations.size());
        profile->total_time = std::chrono::duration<double>(
                                      std::chrono::steady_clock::now() - total_t0)
                                      .count();
    }
    return rotation;
}

void pack_assignments_row(
        const Structure& structure,
        const uint16_t* assignments,
        uint8_t* bytes) {
    std::memset(bytes, 0, structure.code_size());
    int bit_offset = 0;
    for (size_t gi = 0; gi < structure.group_count(); ++gi) {
        const int nbits = structure.groups[gi].nbits;
        uint32_t value = assignments[gi];
        int written = 0;
        while (written < nbits) {
            const int byte_index = bit_offset / 8;
            const int bit_in_byte = bit_offset % 8;
            const int take = std::min(nbits - written, 8 - bit_in_byte);
            const uint32_t mask = (uint32_t{1} << take) - 1U;
            bytes[byte_index] |= static_cast<uint8_t>(
                    ((value >> written) & mask) << bit_in_byte);
            written += take;
            bit_offset += take;
        }
    }
}

void unpack_assignments_row(
        const Structure& structure,
        const uint8_t* bytes,
        uint16_t* assignments) {
    int bit_offset = 0;
    for (size_t gi = 0; gi < structure.group_count(); ++gi) {
        const int nbits = structure.groups[gi].nbits;
        uint32_t value = 0;
        int consumed = 0;
        while (consumed < nbits) {
            const int byte_index = bit_offset / 8;
            const int bit_in_byte = bit_offset % 8;
            const int take = std::min(nbits - consumed, 8 - bit_in_byte);
            const uint32_t mask = (uint32_t{1} << take) - 1U;
            value |=
                    ((static_cast<uint32_t>(bytes[byte_index]) >> bit_in_byte) &
                     mask)
                    << consumed;
            consumed += take;
            bit_offset += take;
        }
        assignments[gi] = static_cast<uint16_t>(value);
    }
}

SearchMode resolve_search_mode(
        SearchMode fallback,
        const faiss::SearchParameters* params) {
    if (params == nullptr) {
        return fallback;
    }
    const auto* epq_params = dynamic_cast<const SearchParametersEPQ*>(params);
    if (epq_params == nullptr) {
        return fallback;
    }
    return epq_params->mode;
}

constexpr int kDefaultSearchQueryBatch = 10;
constexpr int kDefaultSearchDbChunk = 65536;

int read_positive_env_or_default(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || (end != nullptr && *end != '\0') || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max()) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

int search_query_batch() {
    static const int value = read_positive_env_or_default(
            "EPQ_SEARCH_QUERY_BATCH", kDefaultSearchQueryBatch);
    return value;
}

int search_db_chunk() {
    static const int value =
            read_positive_env_or_default("EPQ_SEARCH_DB_CHUNK", kDefaultSearchDbChunk);
    return value;
}

struct SearchScratch {
    std::vector<float> heap_dist;
    std::vector<faiss::idx_t> heap_ids;
    std::vector<float> dist_chunk;
    std::vector<float> reservoir_dist;
    std::vector<faiss::idx_t> reservoir_ids;
};

template <bool Add>
inline void accumulate_code_lut_row(
        float* row,
        const uint16_t* codes,
        const float* lut,
        faiss::idx_t csz) {
    faiss::idx_t j = 0;
    for (; j + 3 < csz; j += 4) {
        const size_t j0 = static_cast<size_t>(j);
        const size_t j1 = static_cast<size_t>(j + 1);
        const size_t j2 = static_cast<size_t>(j + 2);
        const size_t j3 = static_cast<size_t>(j + 3);
        if constexpr (Add) {
            row[j0] += lut[codes[j0]];
            row[j1] += lut[codes[j1]];
            row[j2] += lut[codes[j2]];
            row[j3] += lut[codes[j3]];
        } else {
            row[j0] = lut[codes[j0]];
            row[j1] = lut[codes[j1]];
            row[j2] = lut[codes[j2]];
            row[j3] = lut[codes[j3]];
        }
    }
    for (; j < csz; ++j) {
        const size_t jj = static_cast<size_t>(j);
        if constexpr (Add) {
            row[jj] += lut[codes[jj]];
        } else {
            row[jj] = lut[codes[jj]];
        }
    }
}

template <typename Reservoir>
inline void reservoir_add_row(
        Reservoir& reservoir,
        const float* row,
        faiss::idx_t base,
        faiss::idx_t csz) {
    float threshold = reservoir.threshold;
    faiss::idx_t j = 0;
    for (; j + 3 < csz; j += 4) {
        const size_t j0 = static_cast<size_t>(j);
        const size_t j1 = static_cast<size_t>(j + 1);
        const size_t j2 = static_cast<size_t>(j + 2);
        const size_t j3 = static_cast<size_t>(j + 3);
        const float d0 = row[j0];
        if (d0 < threshold) {
            reservoir.add(d0, base + j);
            threshold = reservoir.threshold;
        }
        const float d1 = row[j1];
        if (d1 < threshold) {
            reservoir.add(d1, base + j + 1);
            threshold = reservoir.threshold;
        }
        const float d2 = row[j2];
        if (d2 < threshold) {
            reservoir.add(d2, base + j + 2);
            threshold = reservoir.threshold;
        }
        const float d3 = row[j3];
        if (d3 < threshold) {
            reservoir.add(d3, base + j + 3);
            threshold = reservoir.threshold;
        }
    }
    for (; j < csz; ++j) {
        const float dis = row[static_cast<size_t>(j)];
        if (dis < threshold) {
            reservoir.add(dis, base + j);
            threshold = reservoir.threshold;
        }
    }
}

}  // namespace

IndexEPQ::IndexEPQ(
        int d,
        int total_bits,
        std::shared_ptr<StructureBuilder> structure_builder)
        : faiss::Index(d, faiss::METRIC_L2),
          total_bits(total_bits),
          structure_builder(std::move(structure_builder)) {
    is_trained = false;
    if (!this->structure_builder) {
        this->structure_builder =
                std::make_shared<RefinedStructureBuilder>();
    }
}

BuildContext IndexEPQ::make_build_context() const {
    return BuildContext{
            .d = d,
            .total_bits = total_bits,
            .min_bits = min_bits,
            .max_bits = max_bits,
    };
}

void IndexEPQ::validate_runtime_config() const {
    if (d <= 0) {
        throw std::invalid_argument("epq::IndexEPQ: d must be positive");
    }
    if (total_bits < 0) {
        throw std::invalid_argument("epq::IndexEPQ: total_bits must be non-negative");
    }
    if (min_bits < 0 || max_bits < min_bits) {
        throw std::invalid_argument("epq::IndexEPQ: invalid bit bounds");
    }
    if (metric_type != faiss::METRIC_L2) {
        throw std::invalid_argument("epq::IndexEPQ: only METRIC_L2 is supported");
    }
    if (transform_init_mode != "identity" &&
        transform_init_mode != "haar_r" &&
        transform_init_mode != "matched_physical") {
        throw std::invalid_argument(
                "epq::IndexEPQ: transform_init_mode must be identity, haar_r, or matched_physical");
    }
}

void IndexEPQ::train(faiss::idx_t n, const float* x) {
    validate_runtime_config();
    FAISS_THROW_IF_NOT_MSG(n > 0, "epq::IndexEPQ::train requires non-empty training data");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "epq::IndexEPQ::train requires non-null input");

    const auto total_t0 = std::chrono::steady_clock::now();
    const Eigen::Map<const RowMatrixXf> xt(x, n, d);
    const auto structure_t0 = std::chrono::steady_clock::now();
    structure_ = structure_builder->build(n, x, make_build_context());
    structure_.validate(min_bits, max_bits);
    const auto structure_t1 = std::chrono::steady_clock::now();

    contiguous_groups_ = structure_.contiguous_groups();
    has_transform_ = false;
    perm_.clear();
    inv_perm_.clear();
    rotation_.resize(0, 0);

    RowMatrixXf train_x = xt;
    active_groups_.clear();
    training_stats_ = {};
    codebook_profiles_.clear();
    transform_profile_ = {};
    runtime_profile_ = {};
    training_stats_.structure_time =
            std::chrono::duration<double>(structure_t1 - structure_t0).count();
    if (use_uneven_transform) {
        const auto prep_t0 = std::chrono::steady_clock::now();
        if (transform_init_mode == "matched_physical") {
            // Canonicalize the reparameterization so every membership uses the
            // same physical W and the same floating-point accumulation order.
            // In exact arithmetic, using the learned permutation with
            // R=P^T W is equivalent, but different GEMM reduction orders can
            // perturb k-means enough to send the non-convex training down a
            // different path.
            perm_.resize(static_cast<size_t>(d));
            std::iota(perm_.begin(), perm_.end(), 0);
        } else {
            perm_ = structure_.flatten_dims();
        }
        inv_perm_ = inverse_permutation(perm_);
        const RowMatrixXf x_perm = permute_columns(xt, perm_);
        rotation_ = train_uneven_rotation(
                x_perm,
                structure_,
                perm_,
                transform_niter,
                transform_kmeans_niter,
                transform_kmeans_nredo,
                transform_max_train,
                transform_max_eval,
                transform_eval_frac,
                transform_seed,
                transform_init_mode,
                transform_init_seed,
                transform_proxy_max_bits,
                transform_exact_polish_iters,
                &transform_profile_);
        train_x = x_perm * rotation_;
        active_groups_ = contiguous_groups_;
        has_transform_ = true;
        const auto prep_t1 = std::chrono::steady_clock::now();
        training_stats_.preparation_time =
                std::chrono::duration<double>(prep_t1 - prep_t0).count();
    } else {
        active_groups_.reserve(structure_.group_count());
        for (const auto& group : structure_.groups) {
            active_groups_.push_back(group.dims);
        }
    }
    refresh_active_group_layout();

    const auto codebook_t0 = std::chrono::steady_clock::now();
    codebooks_ = train_codebooks(
            train_x,
            structure_,
            active_groups_,
            kmeans_niter,
            kmeans_nredo,
            &codebook_profiles_);
    lut_codebooks_.clear();
    lut_codebooks_.reserve(codebooks_.size());
    for (const auto& codebook : codebooks_) {
        lut_codebooks_.emplace_back(codebook);
    }
    centroid_norms_.clear();
    centroid_norms_.reserve(codebooks_.size());
    for (const auto& codebook : codebooks_) {
        centroid_norms_.push_back(centroid_norms(codebook));
    }
    build_sdc_tables();
    const auto codebook_t1 = std::chrono::steady_clock::now();
    training_stats_.codebook_time =
            std::chrono::duration<double>(codebook_t1 - codebook_t0).count();

    database_codes_.clear();
    database_codes_by_group_.clear();
    database_code_capacity_ = 0;
    packed_codes_.clear();
    ntotal = 0;
    is_trained = true;
    training_stats_.total_time =
            std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - total_t0)
                    .count();
}

std::vector<uint16_t> IndexEPQ::compute_assignments_impl(
        faiss::idx_t n,
        const float* x) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "epq::IndexEPQ is not trained");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "epq::IndexEPQ requires non-null input");

    const Eigen::Map<const RowMatrixXf> input(x, n, d);
    RowMatrixXf work;
    if (has_transform_) {
        work = permute_columns(input, perm_) * rotation_;
    } else {
        work = input;
    }

    const size_t m = structure_.group_count();
    std::vector<uint16_t> assignments(static_cast<size_t>(n) * m, 0);
    for (size_t gi = 0; gi < m; ++gi) {
        RowMatrixXf sub = gather_columns(work, active_groups_[gi]);
        const auto labels = assign_codebook(sub, codebooks_[gi]);
        for (faiss::idx_t row = 0; row < n; ++row) {
            assignments[static_cast<size_t>(row) * m + gi] =
                    static_cast<uint16_t>(labels[static_cast<size_t>(row)]);
        }
    }
    return assignments;
}

std::vector<uint16_t> IndexEPQ::compute_assignments(
        faiss::idx_t n,
        const float* x) const {
    return compute_assignments_impl(n, x);
}

void IndexEPQ::add(faiss::idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "epq::IndexEPQ must be trained before add");
    if (n <= 0) {
        return;
    }

    const auto add_t0 = std::chrono::steady_clock::now();
    runtime_profile_.last_add_rows = static_cast<int>(n);
    const auto assignments = compute_assignments_impl(n, x);
    const auto assign_t1 = std::chrono::steady_clock::now();
    const size_t old_total = static_cast<size_t>(ntotal);
    const size_t new_total = old_total + static_cast<size_t>(n);
    const size_t m = structure_.group_count();
    if (new_total > database_code_capacity_) {
        const size_t new_capacity =
                grow_code_capacity(database_code_capacity_, new_total);
        std::vector<uint16_t> next_codes_by_group(
                m * new_capacity,
                uint16_t{0});
        #pragma omp parallel for if (m > 1)
        for (faiss::idx_t gi = 0; gi < static_cast<faiss::idx_t>(m); ++gi) {
            if (old_total == 0) {
                continue;
            }
            std::memcpy(
                    next_codes_by_group.data() +
                            static_cast<size_t>(gi) * new_capacity,
                    database_codes_by_group_.data() +
                            static_cast<size_t>(gi) * database_code_capacity_,
                    old_total * sizeof(uint16_t));
        }
        database_codes_by_group_ = std::move(next_codes_by_group);
        database_code_capacity_ = new_capacity;
    } else if (database_codes_by_group_.size() != m * database_code_capacity_) {
        database_codes_by_group_.resize(m * database_code_capacity_);
    }

    database_codes_.resize(new_total * m);
    std::copy(
            assignments.begin(),
            assignments.end(),
            database_codes_.begin() + static_cast<std::ptrdiff_t>(old_total * m));
    #pragma omp parallel for if (m > 1)
    for (faiss::idx_t gi = 0; gi < static_cast<faiss::idx_t>(m); ++gi) {
        uint16_t* dst = database_codes_by_group_.data() +
                static_cast<size_t>(gi) * database_code_capacity_ + old_total;
        for (faiss::idx_t row = 0; row < n; ++row) {
            dst[static_cast<size_t>(row)] = assignments[static_cast<size_t>(row) * m +
                    static_cast<size_t>(gi)];
        }
    }

    const size_t code_size = structure_.code_size();
    packed_codes_.resize(new_total * code_size);
    for (faiss::idx_t i = 0; i < n; ++i) {
        pack_assignments_row(
                structure_,
                assignments.data() + static_cast<size_t>(i) * m,
                packed_codes_.data() + (old_total + static_cast<size_t>(i)) * code_size);
    }
    ntotal += n;
    const auto add_t1 = std::chrono::steady_clock::now();
    runtime_profile_.last_add_total_time =
            std::chrono::duration<double>(add_t1 - add_t0).count();
    runtime_profile_.last_add_assign_time =
            std::chrono::duration<double>(assign_t1 - add_t0).count();
    runtime_profile_.last_add_transform_time = 0.0;
}

void IndexEPQ::add_with_ids(
        faiss::idx_t,
        const float*,
        const faiss::idx_t*) {
    FAISS_THROW_MSG("epq::IndexEPQ does not support add_with_ids");
}

void IndexEPQ::search(
        faiss::idx_t n,
        const float* x,
        faiss::idx_t k,
        float* distances,
        faiss::idx_t* labels,
        const faiss::SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "epq::IndexEPQ must be trained before search");
    FAISS_THROW_IF_NOT_MSG(ntotal > 0, "epq::IndexEPQ has no database vectors");
    FAISS_THROW_IF_NOT_MSG(k > 0, "epq::IndexEPQ::search requires k > 0");

    const SearchMode mode = resolve_search_mode(default_search_mode_, params);
    const faiss::IDSelector* selector = params != nullptr ? params->sel : nullptr;
    const size_t m = structure_.group_count();
    const auto search_t0 = std::chrono::steady_clock::now();
    runtime_profile_.last_search_mode = mode;
    runtime_profile_.last_search_queries = static_cast<int>(n);
    runtime_profile_.last_search_k = static_cast<int>(k);
    runtime_profile_.last_search_transform_time = 0.0;
    runtime_profile_.last_search_lut_time = 0.0;
    runtime_profile_.last_search_scan_time = 0.0;

    if (mode == SearchMode::kADC) {
        const auto transform_t0 = std::chrono::steady_clock::now();
        const Eigen::Map<const RowMatrixXf> input(x, n, d);
        RowMatrixXf work;
        if (has_transform_) {
            work = permute_columns(input, perm_) * rotation_;
        } else {
            work = input;
        }
        const auto transform_t1 = std::chrono::steady_clock::now();
        runtime_profile_.last_search_transform_time =
                std::chrono::duration<double>(transform_t1 - transform_t0).count();

        const auto lut_t0 = std::chrono::steady_clock::now();
        const faiss::idx_t k_eff = std::min(k, ntotal);
        for (faiss::idx_t i = 0; i < n * k; ++i) {
            distances[i] = std::numeric_limits<float>::infinity();
            labels[i] = -1;
        }
        if (k_eff == 0) {
            runtime_profile_.last_search_total_time =
                    std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - search_t0)
                            .count();
            return;
        }
        const auto lut_t1 = std::chrono::steady_clock::now();
        runtime_profile_.last_search_lut_time =
                std::chrono::duration<double>(lut_t1 - lut_t0).count();

        const auto scan_t0 = std::chrono::steady_clock::now();
        const int qbatch = std::max(1, search_query_batch());
        const int chunk_limit = std::max(1024, search_db_chunk());
        const size_t reservoir_capacity =
                (2 * static_cast<size_t>(k_eff) + size_t{15}) & ~size_t{15};
        #pragma omp parallel for schedule(dynamic)
        for (faiss::idx_t q0 = 0; q0 < n; q0 += qbatch) {
            const faiss::idx_t qb = std::min<faiss::idx_t>(qbatch, n - q0);
            thread_local SearchScratch scratch;
            std::vector<std::vector<float>> luts(m);
            for (size_t gi = 0; gi < m; ++gi) {
                const auto& dims = active_groups_[gi];
                const auto& codebook = codebooks_[gi];
                const auto& norms = centroid_norms_[gi];
                auto& lut = luts[gi];
                lut.resize(static_cast<size_t>(qb) *
                        static_cast<size_t>(codebook.rows()));
                for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                    const float* qrow =
                            work.row(static_cast<Eigen::Index>(q0 + qi)).data();
                    float qnorm = 0.0f;
                    for (const int dim : dims) {
                        qnorm += qrow[dim] * qrow[dim];
                    }
                    float* lut_q = lut.data() +
                            static_cast<size_t>(qi) *
                                    static_cast<size_t>(codebook.rows());
                    for (int code = 0; code < codebook.rows(); ++code) {
                        const float* cent = codebook.row(code).data();
                        float dot = 0.0f;
                        for (size_t j = 0; j < dims.size(); ++j) {
                            dot += qrow[dims[j]] * cent[j];
                        }
                        lut_q[static_cast<size_t>(code)] =
                                qnorm + norms[static_cast<size_t>(code)] -
                                2.0f * dot;
                    }
                }
            }

            using Reservoir =
                    faiss::ReservoirTopN<faiss::CMax<float, faiss::idx_t>>;
            std::vector<Reservoir> reservoirs;
            if (selector == nullptr) {
                scratch.reservoir_dist.resize(
                        static_cast<size_t>(qb) * reservoir_capacity);
                scratch.reservoir_ids.resize(
                        static_cast<size_t>(qb) * reservoir_capacity);
                reservoirs.reserve(static_cast<size_t>(qb));
                for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                    reservoirs.emplace_back(
                            static_cast<size_t>(k_eff),
                            reservoir_capacity,
                            scratch.reservoir_dist.data() +
                                    static_cast<size_t>(qi) * reservoir_capacity,
                            scratch.reservoir_ids.data() +
                                    static_cast<size_t>(qi) * reservoir_capacity);
                }
            } else {
                scratch.heap_dist.resize(
                        static_cast<size_t>(qb) * static_cast<size_t>(k_eff));
                scratch.heap_ids.resize(
                        static_cast<size_t>(qb) * static_cast<size_t>(k_eff));
                for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                    float* hdist = scratch.heap_dist.data() +
                            static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
                    faiss::idx_t* hids = scratch.heap_ids.data() +
                            static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
                    std::fill(
                            hdist,
                            hdist + k_eff,
                            std::numeric_limits<float>::infinity());
                    std::fill(hids, hids + k_eff, faiss::idx_t{-1});
                    faiss::maxheap_heapify(k_eff, hdist, hids);
                }
            }

            for (faiss::idx_t b0 = 0; b0 < ntotal; b0 += chunk_limit) {
                const faiss::idx_t csz =
                        std::min<faiss::idx_t>(chunk_limit, ntotal - b0);
                scratch.dist_chunk.resize(
                        static_cast<size_t>(qb) * static_cast<size_t>(csz));
                if (m > 0) {
                    {
                        const size_t gi = 0;
                        const uint16_t* codes_group =
                                database_codes_by_group_.data() +
                                gi * database_code_capacity_ +
                                static_cast<size_t>(b0);
                        const auto& lut = luts[gi];
                        for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                            float* row = scratch.dist_chunk.data() +
                                    static_cast<size_t>(qi) *
                                            static_cast<size_t>(csz);
                            const float* lut_q = lut.data() +
                                    static_cast<size_t>(qi) *
                                            static_cast<size_t>(codebooks_[gi].rows());
                            accumulate_code_lut_row<false>(
                                    row, codes_group, lut_q, csz);
                        }
                    }
                }
                for (size_t gi = 1; gi < m; ++gi) {
                    const uint16_t* codes_group =
                            database_codes_by_group_.data() +
                            gi * database_code_capacity_ +
                            static_cast<size_t>(b0);
                    const auto& lut = luts[gi];
                    for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                        float* row = scratch.dist_chunk.data() +
                                static_cast<size_t>(qi) *
                                        static_cast<size_t>(csz);
                        const float* lut_q = lut.data() +
                                static_cast<size_t>(qi) *
                                        static_cast<size_t>(codebooks_[gi].rows());
                        accumulate_code_lut_row<true>(
                                row, codes_group, lut_q, csz);
                    }
                }

                if (selector == nullptr) {
                    for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                        auto& reservoir = reservoirs[static_cast<size_t>(qi)];
                        const float* row = scratch.dist_chunk.data() +
                                static_cast<size_t>(qi) * static_cast<size_t>(csz);
                        reservoir_add_row(reservoir, row, b0, csz);
                    }
                } else {
                    for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                        float* hdist = scratch.heap_dist.data() +
                                static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
                        faiss::idx_t* hids = scratch.heap_ids.data() +
                                static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
                        const float* row = scratch.dist_chunk.data() +
                                static_cast<size_t>(qi) * static_cast<size_t>(csz);
                        for (faiss::idx_t j = 0; j < csz; ++j) {
                            const faiss::idx_t db = b0 + j;
                            if (!selector->is_member(db)) {
                                continue;
                            }
                            const float dis = row[static_cast<size_t>(j)];
                            if (dis < hdist[0]) {
                                faiss::maxheap_replace_top(
                                        k_eff,
                                        hdist,
                                        hids,
                                        dis,
                                        db);
                            }
                        }
                    }
                }
            }

            if (selector == nullptr) {
                for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                    reservoirs[static_cast<size_t>(qi)].to_result(
                            distances +
                                    static_cast<size_t>(q0 + qi) *
                                            static_cast<size_t>(k),
                            labels +
                                    static_cast<size_t>(q0 + qi) *
                                            static_cast<size_t>(k));
                }
            } else {
                for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                    float* hdist = scratch.heap_dist.data() +
                            static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
                    faiss::idx_t* hids = scratch.heap_ids.data() +
                            static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
                    faiss::maxheap_reorder(k_eff, hdist, hids);
                    std::memcpy(
                            distances +
                                    static_cast<size_t>(q0 + qi) *
                                            static_cast<size_t>(k),
                            hdist,
                            static_cast<size_t>(k_eff) * sizeof(float));
                    std::memcpy(
                            labels +
                                    static_cast<size_t>(q0 + qi) *
                                            static_cast<size_t>(k),
                            hids,
                            static_cast<size_t>(k_eff) * sizeof(faiss::idx_t));
                }
            }
        }
        const auto scan_t1 = std::chrono::steady_clock::now();
        runtime_profile_.last_search_scan_time =
                std::chrono::duration<double>(scan_t1 - scan_t0).count();
        runtime_profile_.last_search_total_time =
                std::chrono::duration<double>(scan_t1 - search_t0).count();
        return;
    }

    const auto code_t0 = std::chrono::steady_clock::now();
    const auto query_codes = compute_assignments_impl(n, x);
    const auto code_t1 = std::chrono::steady_clock::now();
    runtime_profile_.last_search_transform_time =
            std::chrono::duration<double>(code_t1 - code_t0).count();
    const auto scan_t0 = std::chrono::steady_clock::now();
    const faiss::idx_t k_eff = std::min(k, ntotal);
    for (faiss::idx_t i = 0; i < n * k; ++i) {
        distances[i] = std::numeric_limits<float>::infinity();
        labels[i] = -1;
    }
    if (k_eff == 0) {
        runtime_profile_.last_search_total_time =
                std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - search_t0)
                        .count();
        return;
    }
    const int qbatch = std::max(1, search_query_batch());
    const int chunk_limit = std::max(1024, search_db_chunk());
    const size_t reservoir_capacity =
            (2 * static_cast<size_t>(k_eff) + size_t{15}) & ~size_t{15};
    #pragma omp parallel for schedule(dynamic)
    for (faiss::idx_t q0 = 0; q0 < n; q0 += qbatch) {
        const faiss::idx_t qb = std::min<faiss::idx_t>(qbatch, n - q0);
        thread_local SearchScratch scratch;
        using Reservoir =
                faiss::ReservoirTopN<faiss::CMax<float, faiss::idx_t>>;
        std::vector<Reservoir> reservoirs;
        if (selector == nullptr) {
            scratch.reservoir_dist.resize(
                    static_cast<size_t>(qb) * reservoir_capacity);
            scratch.reservoir_ids.resize(
                    static_cast<size_t>(qb) * reservoir_capacity);
            reservoirs.reserve(static_cast<size_t>(qb));
            for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                reservoirs.emplace_back(
                        static_cast<size_t>(k_eff),
                        reservoir_capacity,
                        scratch.reservoir_dist.data() +
                                static_cast<size_t>(qi) * reservoir_capacity,
                        scratch.reservoir_ids.data() +
                                static_cast<size_t>(qi) * reservoir_capacity);
            }
        } else {
            scratch.heap_dist.resize(
                    static_cast<size_t>(qb) * static_cast<size_t>(k_eff));
            scratch.heap_ids.resize(
                    static_cast<size_t>(qb) * static_cast<size_t>(k_eff));
            for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                float* hdist = scratch.heap_dist.data() +
                        static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
                faiss::idx_t* hids = scratch.heap_ids.data() +
                        static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
                std::fill(
                        hdist,
                        hdist + k_eff,
                        std::numeric_limits<float>::infinity());
                std::fill(hids, hids + k_eff, faiss::idx_t{-1});
                faiss::maxheap_heapify(k_eff, hdist, hids);
            }
        }

        for (faiss::idx_t b0 = 0; b0 < ntotal; b0 += chunk_limit) {
            const faiss::idx_t csz =
                    std::min<faiss::idx_t>(chunk_limit, ntotal - b0);
            scratch.dist_chunk.resize(
                    static_cast<size_t>(qb) * static_cast<size_t>(csz));
            if (m > 0) {
                const size_t gi = 0;
                const uint16_t* codes_group =
                        database_codes_by_group_.data() +
                        gi * database_code_capacity_ +
                        static_cast<size_t>(b0);
                const auto& sdc = sdc_tables_[gi];
                for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                    float* row = scratch.dist_chunk.data() +
                            static_cast<size_t>(qi) * static_cast<size_t>(csz);
                    const uint16_t qcode =
                            query_codes[static_cast<size_t>(q0 + qi) * m + gi];
                    const float* sdc_row =
                            sdc.row(static_cast<Eigen::Index>(qcode)).data();
                    accumulate_code_lut_row<false>(
                            row, codes_group, sdc_row, csz);
                }
            }
            for (size_t gi = 1; gi < m; ++gi) {
                const uint16_t* codes_group =
                        database_codes_by_group_.data() +
                        gi * database_code_capacity_ +
                        static_cast<size_t>(b0);
                const auto& sdc = sdc_tables_[gi];
                for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                    float* row = scratch.dist_chunk.data() +
                            static_cast<size_t>(qi) * static_cast<size_t>(csz);
                    const uint16_t qcode =
                            query_codes[static_cast<size_t>(q0 + qi) * m + gi];
                    const float* sdc_row =
                            sdc.row(static_cast<Eigen::Index>(qcode)).data();
                    accumulate_code_lut_row<true>(
                            row, codes_group, sdc_row, csz);
                }
            }

            if (selector == nullptr) {
                for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                    auto& reservoir = reservoirs[static_cast<size_t>(qi)];
                    const float* row = scratch.dist_chunk.data() +
                            static_cast<size_t>(qi) * static_cast<size_t>(csz);
                    reservoir_add_row(reservoir, row, b0, csz);
                }
            } else {
                for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                    float* hdist = scratch.heap_dist.data() +
                            static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
                    faiss::idx_t* hids = scratch.heap_ids.data() +
                            static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
                    const float* row = scratch.dist_chunk.data() +
                            static_cast<size_t>(qi) * static_cast<size_t>(csz);
                    for (faiss::idx_t j = 0; j < csz; ++j) {
                        const faiss::idx_t db = b0 + j;
                        if (!selector->is_member(db)) {
                            continue;
                        }
                        const float dis = row[static_cast<size_t>(j)];
                        if (dis < hdist[0]) {
                            faiss::maxheap_replace_top(
                                    k_eff,
                                    hdist,
                                    hids,
                                    dis,
                                    db);
                        }
                    }
                }
            }
        }

        if (selector == nullptr) {
            for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                reservoirs[static_cast<size_t>(qi)].to_result(
                        distances + static_cast<size_t>(q0 + qi) * static_cast<size_t>(k),
                        labels + static_cast<size_t>(q0 + qi) * static_cast<size_t>(k));
            }
        } else {
            for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                float* hdist = scratch.heap_dist.data() +
                        static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
                faiss::idx_t* hids = scratch.heap_ids.data() +
                        static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
                faiss::maxheap_reorder(k_eff, hdist, hids);
                std::memcpy(
                        distances + static_cast<size_t>(q0 + qi) * static_cast<size_t>(k),
                        hdist,
                        static_cast<size_t>(k_eff) * sizeof(float));
                std::memcpy(
                        labels + static_cast<size_t>(q0 + qi) * static_cast<size_t>(k),
                        hids,
                        static_cast<size_t>(k_eff) * sizeof(faiss::idx_t));
            }
        }
    }
    const auto scan_t1 = std::chrono::steady_clock::now();
    runtime_profile_.last_search_scan_time =
            std::chrono::duration<double>(scan_t1 - scan_t0).count();
    runtime_profile_.last_search_total_time =
            std::chrono::duration<double>(scan_t1 - search_t0).count();
}

void IndexEPQ::reset() {
    database_codes_.clear();
    database_codes_by_group_.clear();
    database_code_capacity_ = 0;
    packed_codes_.clear();
    ntotal = 0;
}

void IndexEPQ::decode_assignments(
        faiss::idx_t n,
        const uint16_t* assignments,
        float* x) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "epq::IndexEPQ is not trained");
    const size_t m = structure_.group_count();
    RowMatrixXf work(n, d);
    work.setZero();
    for (faiss::idx_t row = 0; row < n; ++row) {
        const size_t row_offset = static_cast<size_t>(row) * m;
        for (size_t gi = 0; gi < m; ++gi) {
            const uint16_t code = assignments[row_offset + gi];
            const auto& dims = active_groups_[gi];
            const auto& codebook = codebooks_[gi];
            for (size_t j = 0; j < dims.size(); ++j) {
                work(row, dims[j]) = codebook(code, static_cast<Eigen::Index>(j));
            }
        }
    }

    Eigen::Map<RowMatrixXf> out(x, n, d);
    if (has_transform_) {
        const RowMatrixXf x_perm = work * rotation_.transpose();
        out.setZero();
        unpermute_columns(x_perm, perm_, out);
    } else {
        out = work;
    }
}

void IndexEPQ::reconstruct(faiss::idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(key >= 0 && key < ntotal, "epq::IndexEPQ: reconstruct key out of range");
    decode_assignments(1, database_codes_.data() + static_cast<size_t>(key) * structure_.group_count(), recons);
}

size_t IndexEPQ::sa_code_size() const {
    return structure_.code_size();
}

void IndexEPQ::sa_encode(
        faiss::idx_t n,
        const float* x,
        uint8_t* bytes) const {
    const auto assignments = compute_assignments_impl(n, x);
    const size_t m = structure_.group_count();
    const size_t code_size = structure_.code_size();
    for (faiss::idx_t row = 0; row < n; ++row) {
        pack_assignments_row(
                structure_,
                assignments.data() + static_cast<size_t>(row) * m,
                bytes + static_cast<size_t>(row) * code_size);
    }
}

void IndexEPQ::sa_decode(
        faiss::idx_t n,
        const uint8_t* bytes,
        float* x) const {
    const size_t m = structure_.group_count();
    const size_t code_size = structure_.code_size();
    std::vector<uint16_t> assignments(static_cast<size_t>(n) * m, 0);
    for (faiss::idx_t row = 0; row < n; ++row) {
        unpack_assignments_row(
                structure_,
                bytes + static_cast<size_t>(row) * code_size,
                assignments.data() + static_cast<size_t>(row) * m);
    }
    decode_assignments(n, assignments.data(), x);
}

size_t IndexEPQ::adc_lut_size() const noexcept {
    size_t total = 0;
    for (const auto& codebook : codebooks_) {
        total += static_cast<size_t>(codebook.rows());
    }
    return total;
}

size_t IndexEPQ::ivf_lut_build_work() const noexcept {
    size_t total = 0;
    const size_t group_count =
            std::min(active_groups_.size(), codebooks_.size());
    for (size_t gi = 0; gi < group_count; ++gi) {
        total += active_groups_[gi].size() *
                static_cast<size_t>(codebooks_[gi].rows());
    }
    return total;
}

size_t IndexEPQ::ivf_default_lut_min_list_size() const noexcept {
    // Exact decode is prohibitively expensive in high dimension, so keep the
    // LUT path unconditional there unless the caller explicitly overrides it.
    if (d > 256) {
        return 0;
    }
    const size_t build_work = ivf_lut_build_work();
    if (build_work == 0 || d <= 0) {
        return 0;
    }
    // Keep the default conservative: on current low-dimensional IVF workloads,
    // exact decode is only useful on fairly small lists.
    const size_t denom = 32 * static_cast<size_t>(d);
    const size_t threshold = build_work / std::max<size_t>(size_t{1}, denom);
    return std::clamp<size_t>(threshold, 128, 512);
}

void IndexEPQ::refresh_active_group_layout() {
    active_group_spans_.clear();
    active_group_spans_.reserve(active_groups_.size());
    all_active_groups_contiguous_ = true;
    for (const auto& dims : active_groups_) {
        const GroupSpan span = contiguous_span(dims);
        active_group_spans_.push_back(span);
        all_active_groups_contiguous_ =
                all_active_groups_contiguous_ && span.contiguous;
    }
}

void IndexEPQ::transform_vector(const float* x, float* out) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "epq::IndexEPQ is not trained");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "epq::IndexEPQ requires non-null input");
    FAISS_THROW_IF_NOT_MSG(out != nullptr, "epq::IndexEPQ requires non-null output");

    Eigen::Map<RowMatrixXf> out_map(out, 1, d);
    if (!has_transform_) {
        std::memcpy(out, x, sizeof(float) * static_cast<size_t>(d));
        return;
    }

    const Eigen::Map<const RowMatrixXf> input(x, 1, d);
    out_map = permute_columns(input, perm_) * rotation_;
}

void IndexEPQ::compute_adc_lut_from_transformed(
        const float* transformed_x,
        float* lut) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "epq::IndexEPQ is not trained");
    FAISS_THROW_IF_NOT_MSG(
            transformed_x != nullptr, "epq::IndexEPQ requires transformed query");
    FAISS_THROW_IF_NOT_MSG(lut != nullptr, "epq::IndexEPQ requires non-null LUT output");

    size_t lut_offset = 0;
    if (all_active_groups_contiguous_) {
        for (size_t gi = 0; gi < active_groups_.size(); ++gi) {
            const auto& lut_codebook = lut_codebooks_[gi];
            const auto& norms = centroid_norms_[gi];
            const GroupSpan& span = active_group_spans_[gi];
            Eigen::Map<Eigen::VectorXf> lut_block(
                    lut + lut_offset, lut_codebook.rows());
            const Eigen::Map<const Eigen::VectorXf> norms_map(
                    norms.data(), static_cast<Eigen::Index>(norms.size()));
            const Eigen::Map<const Eigen::VectorXf> qsub(
                    transformed_x + span.begin, span.size);
            const float qnorm = qsub.squaredNorm();
            lut_block.noalias() = lut_codebook * qsub;
            lut_block *= -2.0f;
            lut_block.array() += norms_map.array() + qnorm;
            lut_offset += static_cast<size_t>(lut_codebook.rows());
        }
        return;
    }

    for (size_t gi = 0; gi < active_groups_.size(); ++gi) {
        const auto& dims = active_groups_[gi];
        const auto& codebook = codebooks_[gi];
        const auto& lut_codebook = lut_codebooks_[gi];
        const auto& norms = centroid_norms_[gi];
        Eigen::Map<Eigen::VectorXf> lut_block(
                lut + lut_offset, lut_codebook.rows());
        const Eigen::Map<const Eigen::VectorXf> norms_map(
                norms.data(), static_cast<Eigen::Index>(norms.size()));
        const GroupSpan& span = active_group_spans_[gi];
        if (span.contiguous) {
            const Eigen::Map<const Eigen::VectorXf> qsub(
                    transformed_x + span.begin, span.size);
            const float qnorm = qsub.squaredNorm();
            lut_block.noalias() = lut_codebook * qsub;
            lut_block *= -2.0f;
            lut_block.array() += norms_map.array() + qnorm;
        } else {
            const float* qrow = transformed_x;
            float qnorm = 0.0f;
            for (const int dim : dims) {
                qnorm += qrow[dim] * qrow[dim];
            }
            for (int code = 0; code < codebook.rows(); ++code) {
                const float* cent = codebook.row(code).data();
                float dot = 0.0f;
                for (size_t j = 0; j < dims.size(); ++j) {
                    dot += qrow[dims[j]] * cent[j];
                }
                lut[lut_offset + static_cast<size_t>(code)] =
                        qnorm + norms[static_cast<size_t>(code)] - 2.0f * dot;
            }
        }
        lut_offset += static_cast<size_t>(codebook.rows());
    }
}

void IndexEPQ::unpack_code_assignments(
        const uint8_t* code,
        uint16_t* assignments) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "epq::IndexEPQ is not trained");
    FAISS_THROW_IF_NOT_MSG(code != nullptr, "epq::IndexEPQ requires non-null code");
    FAISS_THROW_IF_NOT_MSG(
            assignments != nullptr,
            "epq::IndexEPQ requires non-null assignments");
    unpack_assignments_row(structure_, code, assignments);
}

float IndexEPQ::adc_distance_from_assignments(
        const uint16_t* assignments,
        const float* lut) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "epq::IndexEPQ is not trained");
    FAISS_THROW_IF_NOT_MSG(
            assignments != nullptr,
            "epq::IndexEPQ requires non-null assignments");
    FAISS_THROW_IF_NOT_MSG(lut != nullptr, "epq::IndexEPQ requires non-null LUT");

    float distance = 0.0f;
    size_t lut_offset = 0;
    for (size_t gi = 0; gi < structure_.group_count(); ++gi) {
        distance += lut[lut_offset + static_cast<size_t>(assignments[gi])];
        lut_offset += static_cast<size_t>(codebooks_[gi].rows());
    }
    return distance;
}

float IndexEPQ::exact_distance_from_assignments_transformed(
        const uint16_t* assignments,
        const float* transformed_x) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "epq::IndexEPQ is not trained");
    FAISS_THROW_IF_NOT_MSG(
            assignments != nullptr,
            "epq::IndexEPQ requires non-null assignments");
    FAISS_THROW_IF_NOT_MSG(
            transformed_x != nullptr,
            "epq::IndexEPQ requires non-null transformed query");

    float distance = 0.0f;
    for (size_t gi = 0; gi < structure_.group_count(); ++gi) {
        const auto& codebook = codebooks_[gi];
        const float* cent =
                codebook.row(static_cast<Eigen::Index>(assignments[gi])).data();
        const GroupSpan& span = active_group_spans_[gi];
        if (span.contiguous) {
            const float* qsub = transformed_x + span.begin;
            for (int j = 0; j < span.size; ++j) {
                const float diff = qsub[j] - cent[j];
                distance += diff * diff;
            }
        } else {
            const auto& dims = active_groups_[gi];
            for (size_t j = 0; j < dims.size(); ++j) {
                const float diff = transformed_x[dims[j]] - cent[j];
                distance += diff * diff;
            }
        }
    }
    return distance;
}

float IndexEPQ::exact_distance_from_packed_code_transformed(
        const uint8_t* code,
        const float* transformed_x) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "epq::IndexEPQ is not trained");
    FAISS_THROW_IF_NOT_MSG(code != nullptr, "epq::IndexEPQ requires non-null code");
    FAISS_THROW_IF_NOT_MSG(
            transformed_x != nullptr,
            "epq::IndexEPQ requires non-null transformed query");

    float distance = 0.0f;
    int bit_offset = 0;
    for (size_t gi = 0; gi < structure_.group_count(); ++gi) {
        const int nbits = structure_.groups[gi].nbits;
        uint32_t packed = 0;
        int consumed = 0;
        while (consumed < nbits) {
            const int byte_index = bit_offset / 8;
            const int bit_in_byte = bit_offset % 8;
            const int take = std::min(nbits - consumed, 8 - bit_in_byte);
            const uint32_t mask = (uint32_t{1} << take) - 1U;
            packed |=
                    ((static_cast<uint32_t>(code[byte_index]) >> bit_in_byte) &
                     mask)
                    << consumed;
            consumed += take;
            bit_offset += take;
        }
        const auto& codebook = codebooks_[gi];
        const float* cent = codebook.row(static_cast<Eigen::Index>(packed)).data();
        const GroupSpan& span = active_group_spans_[gi];
        if (span.contiguous) {
            const float* qsub = transformed_x + span.begin;
            for (int j = 0; j < span.size; ++j) {
                const float diff = qsub[j] - cent[j];
                distance += diff * diff;
            }
        } else {
            const auto& dims = active_groups_[gi];
            for (size_t j = 0; j < dims.size(); ++j) {
                const float diff = transformed_x[dims[j]] - cent[j];
                distance += diff * diff;
            }
        }
    }
    return distance;
}

float IndexEPQ::adc_distance_from_packed_code(const uint8_t* code, const float* lut) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "epq::IndexEPQ is not trained");
    FAISS_THROW_IF_NOT_MSG(code != nullptr, "epq::IndexEPQ requires non-null code");
    FAISS_THROW_IF_NOT_MSG(lut != nullptr, "epq::IndexEPQ requires non-null LUT");

    float distance = 0.0f;
    uint32_t current = 0;
    int bits_avail = 0;
    size_t byte_pos = 0;
    size_t lut_offset = 0;
    for (size_t gi = 0; gi < structure_.group_count(); ++gi) {
        const int nbits = structure_.groups[gi].nbits;
        uint32_t packed = 0;
        int bits_read = 0;
        while (bits_read < nbits) {
            if (bits_avail == 0) {
                current = code[byte_pos++];
                bits_avail = 8;
            }
            const int take = std::min(bits_avail, nbits - bits_read);
            const uint32_t mask = (uint32_t{1} << take) - 1U;
            packed |= (current & mask) << bits_read;
            current >>= take;
            bits_avail -= take;
            bits_read += take;
        }
        distance += lut[lut_offset + static_cast<size_t>(packed)];
        lut_offset += static_cast<size_t>(codebooks_[gi].rows());
    }
    return distance;
}

const Structure& IndexEPQ::structure() const noexcept {
    return structure_;
}

const std::vector<std::vector<int>>& IndexEPQ::active_groups() const noexcept {
    return active_groups_;
}

const std::vector<RowMatrixXf>& IndexEPQ::codebooks() const noexcept {
    return codebooks_;
}

const TrainingStats& IndexEPQ::training_stats() const noexcept {
    return training_stats_;
}

const std::vector<CodebookProfile>& IndexEPQ::codebook_profiles() const noexcept {
    return codebook_profiles_;
}

const TransformProfile& IndexEPQ::transform_profile() const noexcept {
    return transform_profile_;
}

const RuntimeProfile& IndexEPQ::runtime_profile() const noexcept {
    return runtime_profile_;
}

SearchMode IndexEPQ::default_search_mode() const noexcept {
    return default_search_mode_;
}

void IndexEPQ::set_default_search_mode(SearchMode mode) noexcept {
    default_search_mode_ = mode;
}

void IndexEPQ::serialize_payload(faiss::IOWriter& writer) const {
    static constexpr char kMagic[] = "EPQv1";
    write_vector_data(writer, kMagic, sizeof(kMagic));
    write_scalar<int32_t>(writer, d);
    write_scalar<int32_t>(writer, total_bits);
    write_scalar<int64_t>(writer, ntotal);
    write_scalar<uint8_t>(writer, has_transform_ ? 1 : 0);
    write_scalar<uint64_t>(writer, static_cast<uint64_t>(structure_.format_version));
    write_string(writer, structure_.to_json().dump());
    write_vector<int>(writer, perm_);
    write_vector<int>(writer, inv_perm_);
    write_scalar<int64_t>(writer, rotation_.rows());
    write_scalar<int64_t>(writer, rotation_.cols());
    write_vector_data(
            writer,
            rotation_.data(),
            static_cast<size_t>(rotation_.size()));
    write_scalar<uint64_t>(writer, static_cast<uint64_t>(codebooks_.size()));
    for (const auto& codebook : codebooks_) {
        write_scalar<int64_t>(writer, codebook.rows());
        write_scalar<int64_t>(writer, codebook.cols());
        write_vector_data(
                writer,
                codebook.data(),
                static_cast<size_t>(codebook.size()));
    }
    write_scalar<uint64_t>(writer, static_cast<uint64_t>(packed_codes_.size()));
    write_vector_data(writer, packed_codes_.data(), packed_codes_.size());
}

size_t IndexEPQ::serialized_payload_bytes() const {
    CountingIOWriter writer;
    serialize_payload(writer);
    return writer.bytes_written;
}

void IndexEPQ::build_sdc_tables() {
    sdc_tables_.clear();
    sdc_tables_.reserve(codebooks_.size());
    for (const auto& codebook : codebooks_) {
        RowMatrixXf table(codebook.rows(), codebook.rows());
        for (int i = 0; i < codebook.rows(); ++i) {
            for (int j = 0; j < codebook.rows(); ++j) {
                table(i, j) =
                        (codebook.row(i) - codebook.row(j)).squaredNorm();
            }
        }
        sdc_tables_.push_back(std::move(table));
    }
}

}  // namespace epq
