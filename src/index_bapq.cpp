#include "epq/index_bapq.h"
#include "epq/serialization_size.h"
#include "structure_builder_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/index_io.h>
#include <faiss/utils/Heap.h>

namespace epq {
namespace sbi = structure_builder_internal;
namespace {

struct CacheEntry {
    bool ready = false;
    float mse = 0.0f;
    RowMatrixXf codebook;
};

RowMatrixXf make_matrix_view(const float* x, faiss::idx_t n, int d) {
    Eigen::Map<const RowMatrixXf> mapped(x, static_cast<Eigen::Index>(n), d);
    return mapped;
}

float squared_row_norm(const float* x, int d) {
    float acc = 0.0f;
    for (int i = 0; i < d; ++i) {
        acc += x[i] * x[i];
    }
    return acc;
}

std::vector<float> centroid_norms(const RowMatrixXf& codebook) {
    std::vector<float> norms(static_cast<size_t>(codebook.rows()), 0.0f);
    for (Eigen::Index i = 0; i < codebook.rows(); ++i) {
        norms[static_cast<size_t>(i)] = codebook.row(i).squaredNorm();
    }
    return norms;
}

CacheEntry train_codebook_entry(
        const RowMatrixXf& sub,
        int bits,
        int niter,
        int nredo,
        int seed) {
    if (sub.rows() <= 0 || sub.cols() <= 0) {
        throw std::invalid_argument("IndexBAPQ: empty training slice");
    }

    CacheEntry entry;
    const int k = bits > 0 ? (1 << bits) : 1;
    if (k <= 1) {
        entry.codebook.resize(1, sub.cols());
        entry.codebook.row(0) = sub.colwise().mean();
        const RowMatrixXf centered =
                sub.rowwise() - entry.codebook.row(0);
        entry.mse = centered.array().square().rowwise().sum().mean();
        entry.ready = true;
        return entry;
    }

    const int effective_k = std::min<int>(k, sub.rows());
    faiss::ClusteringParameters cp;
    cp.niter = niter;
    cp.nredo = nredo;
    cp.verbose = false;
    cp.min_points_per_centroid = 1;
    faiss::Clustering clustering(sub.cols(), effective_k, cp);
    clustering.seed = seed;
    faiss::IndexFlatL2 assign_index(sub.cols());
    clustering.train(sub.rows(), sub.data(), assign_index);

    entry.codebook.resize(k, sub.cols());
    Eigen::Map<const RowMatrixXf> trained(
            clustering.centroids.data(),
            effective_k,
            sub.cols());
    entry.codebook.topRows(effective_k) = trained;
    for (int i = effective_k; i < k; ++i) {
        entry.codebook.row(i) = trained.row((effective_k - 1 + i) % effective_k);
    }

    faiss::IndexFlatL2 eval_index(sub.cols());
    eval_index.add(k, entry.codebook.data());
    std::vector<float> distances(static_cast<size_t>(sub.rows()));
    std::vector<faiss::idx_t> labels(static_cast<size_t>(sub.rows()));
    eval_index.search(
            sub.rows(),
            sub.data(),
            1,
            distances.data(),
            labels.data());
    entry.mse = std::accumulate(distances.begin(), distances.end(), 0.0) /
            static_cast<double>(distances.size());
    entry.ready = true;
    return entry;
}

size_t packed_code_size_bytes(int total_bits) {
    return static_cast<size_t>((total_bits + 7) / 8);
}

size_t grow_code_capacity(size_t current, size_t required) {
    size_t next = current == 0 ? size_t{1024}
                               : std::max(current + current / 2, current + size_t{1});
    while (next < required) {
        next = std::max(next + next / 2, next + size_t{1});
    }
    return next;
}

}  // namespace

IndexBAPQ::IndexBAPQ(int d_in, int total_bits_in, int subspace_dim_in)
        : faiss::Index(d_in, faiss::METRIC_L2),
          total_bits(total_bits_in),
          subspace_dim(subspace_dim_in) {
    is_trained = false;
}

void IndexBAPQ::validate_config() const {
    if (d <= 0) {
        throw std::invalid_argument("IndexBAPQ: d must be positive");
    }
    if (total_bits < 0) {
        throw std::invalid_argument("IndexBAPQ: total_bits must be non-negative");
    }
    if (subspace_dim <= 0) {
        throw std::invalid_argument("IndexBAPQ: subspace_dim must be positive");
    }
    if (bmax <= 0 || bmax > 15) {
        throw std::invalid_argument("IndexBAPQ: bmax must be in [1, 15]");
    }
    const int M = (d + subspace_dim - 1) / subspace_dim;
    if (total_bits > M * bmax) {
        throw std::invalid_argument("IndexBAPQ: infeasible total_bits for bmax");
    }
}

RowMatrixXf IndexBAPQ::sample_rows(
        const RowMatrixXf& x,
        int max_rows,
        uint32_t seed_offset) const {
    if (max_rows <= 0 || x.rows() <= max_rows) {
        return x;
    }
    std::vector<int> ids(static_cast<size_t>(x.rows()));
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 rng(seed + seed_offset);
    std::shuffle(ids.begin(), ids.end(), rng);
    RowMatrixXf out(max_rows, x.cols());
    for (int i = 0; i < max_rows; ++i) {
        out.row(i) = x.row(ids[static_cast<size_t>(i)]);
    }
    return out;
}

RowMatrixXf IndexBAPQ::apply_transform(const RowMatrixXf& x) const {
    if (!pca_) {
        throw std::runtime_error("IndexBAPQ: PCA transform is not trained");
    }
    RowMatrixXf out(x.rows(), x.cols());
    pca_->apply_noalloc(x.rows(), x.data(), out.data());
    return out;
}

void IndexBAPQ::apply_transform_noalloc(
        faiss::idx_t n,
        const float* x,
        float* out) const {
    if (!pca_) {
        throw std::runtime_error("IndexBAPQ: PCA transform is not trained");
    }
    pca_->apply_noalloc(n, x, out);
}

void IndexBAPQ::assign_active_group_codes_transformed(
        faiss::idx_t n,
        const float* x_transformed,
        size_t active_group_index,
        uint16_t* dst) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT_MSG(x_transformed != nullptr, "IndexBAPQ: assignment requires transformed input");
    FAISS_THROW_IF_NOT_MSG(dst != nullptr, "IndexBAPQ: assignment requires output buffer");
    FAISS_THROW_IF_NOT_MSG(
            active_group_index < active_groups_.size(),
            "IndexBAPQ: active group index out of range");
    if (n <= 0) {
        return;
    }

    const auto& group = groups_[static_cast<size_t>(active_groups_[active_group_index])];
    if (group.ksub <= 1) {
        std::fill(dst, dst + static_cast<size_t>(n), uint16_t{0});
        return;
    }

    const float* codebook = group.codebook.data();
    const int size = group.size;
    const size_t stride = static_cast<size_t>(d);

    for (faiss::idx_t row = 0; row < n; ++row) {
        const float* xrow =
                x_transformed + static_cast<size_t>(row) * stride + group.begin;
        float best_score = std::numeric_limits<float>::infinity();
        uint16_t best_code = 0;

        for (int c = 0; c < group.ksub; ++c) {
            const float* cent = codebook + static_cast<size_t>(c) * static_cast<size_t>(size);
            float dot = 0.0f;
            switch (size) {
                case 1:
                    dot = xrow[0] * cent[0];
                    break;
                case 2:
                    dot = xrow[0] * cent[0] + xrow[1] * cent[1];
                    break;
                case 3:
                    dot = xrow[0] * cent[0] + xrow[1] * cent[1] +
                            xrow[2] * cent[2];
                    break;
                case 4:
                    dot = xrow[0] * cent[0] + xrow[1] * cent[1] +
                            xrow[2] * cent[2] + xrow[3] * cent[3];
                    break;
                default:
                    for (int j = 0; j < size; ++j) {
                        dot += xrow[j] * cent[j];
                    }
                    break;
            }
            const float score =
                    group.centroid_norms[static_cast<size_t>(c)] - 2.0f * dot;
            if (score < best_score) {
                best_score = score;
                best_code = static_cast<uint16_t>(c);
            }
        }
        dst[static_cast<size_t>(row)] = best_code;
    }
}

void IndexBAPQ::train(faiss::idx_t n, const float* x) {
    validate_config();
    if (n <= 0 || x == nullptr) {
        throw std::invalid_argument("IndexBAPQ: training data is empty");
    }

    const auto total_t0 = std::chrono::steady_clock::now();
    const RowMatrixXf xt = make_matrix_view(x, n, d);

    component_count_ = (d + subspace_dim - 1) / subspace_dim;
    groups_.assign(static_cast<size_t>(component_count_), {});
    group_sizes_.assign(static_cast<size_t>(component_count_), 0);
    int offset = 0;
    for (int gi = 0; gi < component_count_; ++gi) {
        const int size = std::min(subspace_dim, d - offset);
        groups_[static_cast<size_t>(gi)].begin = offset;
        groups_[static_cast<size_t>(gi)].size = size;
        group_sizes_[static_cast<size_t>(gi)] = size;
        offset += size;
    }

    const RowMatrixXf pca_fit = sample_rows(xt, pca_max_train_rows, 17);
    pca_ = std::make_unique<faiss::PCAMatrix>(d, d, 0.0f, false);
    pca_->train(pca_fit.rows(), pca_fit.data());
    RowMatrixXf xt_pca = apply_transform(xt);
    RowMatrixXf xt_sample = sample_rows(xt_pca, max_train_rows, 23);

    std::vector<int> bits(static_cast<size_t>(component_count_), 0);
    std::vector<float> errors(static_cast<size_t>(component_count_), 0.0f);
    std::vector<std::vector<CacheEntry>> cache(
            static_cast<size_t>(component_count_),
            std::vector<CacheEntry>(static_cast<size_t>(bmax + 1)));

    double codebook_seconds = 0.0;
    auto get_entry = [&](int gi, int b) -> const CacheEntry& {
        auto& entry =
                cache[static_cast<size_t>(gi)][static_cast<size_t>(b)];
        if (entry.ready) {
            return entry;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const auto& group = groups_[static_cast<size_t>(gi)];
        RowMatrixXf sub =
                xt_sample.block(0, group.begin, xt_sample.rows(), group.size);
        entry = train_codebook_entry(
                sub,
                b,
                kmeans_niter,
                kmeans_nredo,
                seed + 10007 * gi + 7919 * b);
        codebook_seconds += std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - t0)
                                    .count();
        return entry;
    };

    double total_error = 0.0;
    for (int gi = 0; gi < component_count_; ++gi) {
        const auto& entry = get_entry(gi, 0);
        errors[static_cast<size_t>(gi)] = entry.mse;
        total_error += entry.mse;
    }

    for (int step = 0; step < total_bits; ++step) {
        int best_group = -1;
        double best_total = std::numeric_limits<double>::infinity();
        for (int gi = 0; gi < component_count_; ++gi) {
            if (bits[static_cast<size_t>(gi)] >= bmax) {
                continue;
            }
            const int next_bits = bits[static_cast<size_t>(gi)] + 1;
            const auto& entry = get_entry(gi, next_bits);
            const double candidate_total =
                    total_error - errors[static_cast<size_t>(gi)] + entry.mse;
            if (candidate_total < best_total) {
                best_total = candidate_total;
                best_group = gi;
            }
        }
        if (best_group < 0) {
            throw std::runtime_error("IndexBAPQ: greedy bit allocation failed");
        }
        bits[static_cast<size_t>(best_group)] += 1;
        const auto& entry =
                get_entry(best_group, bits[static_cast<size_t>(best_group)]);
        total_error =
                total_error - errors[static_cast<size_t>(best_group)] + entry.mse;
        errors[static_cast<size_t>(best_group)] = entry.mse;
    }

    nbits_per_group_ = bits;
    active_groups_.clear();
    for (int gi = 0; gi < component_count_; ++gi) {
        auto& group = groups_[static_cast<size_t>(gi)];
        group.nbits = bits[static_cast<size_t>(gi)];
        group.ksub = group.nbits > 0 ? (1 << group.nbits) : 1;
        group.codebook =
                cache[static_cast<size_t>(gi)][static_cast<size_t>(group.nbits)]
                        .codebook;
        group.centroid_norms = centroid_norms(group.codebook);
        group.active = group.nbits > 0;
        group.active_slot = -1;
        if (group.active) {
            group.active_slot = static_cast<int>(active_groups_.size());
            active_groups_.push_back(gi);
        }
    }

    if (sbi::group_stats_env_enabled()) {
        sbi::Groups proxy_groups;
        proxy_groups.reserve(static_cast<size_t>(component_count_));
        for (const auto& group : groups_) {
            std::vector<int> dims;
            dims.reserve(static_cast<size_t>(group.size));
            for (int j = 0; j < group.size; ++j) {
                dims.push_back(group.begin + j);
            }
            proxy_groups.push_back(std::move(dims));
        }
        const BuildContext ctx{
                .d = d,
                .total_bits = total_bits,
                .min_bits = 0,
                .max_bits = bmax,
        };
        sbi::print_group_proxy_stats_from_matrix(
                std::cout,
                "BAPQ",
                "pca-space",
                proxy_groups,
                nbits_per_group_,
                xt_pca,
                ctx);
    }

    training_stats_.structure_time = 0.0;
    training_stats_.codebook_time = codebook_seconds;
    training_stats_.total_time = std::chrono::duration<double>(
                                         std::chrono::steady_clock::now() -
                                         total_t0)
                                         .count();
    training_stats_.preparation_time =
            std::max(0.0, training_stats_.total_time - codebook_seconds);

    codes_.clear();
    code_capacity_ = 0;
    ntotal = 0;
    is_trained = true;
}

void IndexBAPQ::add(faiss::idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    if (n <= 0) {
        return;
    }

    static thread_local std::vector<float> x_pca_storage;
    x_pca_storage.resize(static_cast<size_t>(n) * static_cast<size_t>(d));
    apply_transform_noalloc(n, x, x_pca_storage.data());
    const float* x_pca = x_pca_storage.data();

    const size_t active_m = active_groups_.size();
    const faiss::idx_t old_ntotal = ntotal;
    const size_t new_ntotal = static_cast<size_t>(old_ntotal + n);
    if (new_ntotal > code_capacity_) {
        const size_t new_capacity =
                grow_code_capacity(code_capacity_, new_ntotal);
        std::vector<uint16_t> next_codes(active_m * new_capacity, uint16_t{0});
        #pragma omp parallel for if (active_m > 1)
        for (faiss::idx_t ai = 0; ai < static_cast<faiss::idx_t>(active_m); ++ai) {
            if (old_ntotal == 0) {
                continue;
            }
            std::memcpy(
                    next_codes.data() + static_cast<size_t>(ai) * new_capacity,
                    codes_.data() +
                            static_cast<size_t>(ai) * code_capacity_,
                    static_cast<size_t>(old_ntotal) * sizeof(uint16_t));
        }
        codes_ = std::move(next_codes);
        code_capacity_ = new_capacity;
    } else if (codes_.size() != active_m * code_capacity_) {
        codes_.resize(active_m * code_capacity_);
    }

    #pragma omp parallel for if (active_m > 1)
    for (faiss::idx_t ai = 0; ai < static_cast<faiss::idx_t>(active_m); ++ai) {
        uint16_t* dst = codes_.data() +
                static_cast<size_t>(ai) * code_capacity_ +
                static_cast<size_t>(old_ntotal);
        assign_active_group_codes_transformed(n, x_pca, ai, dst);
    }

    ntotal = static_cast<faiss::idx_t>(new_ntotal);
}

void IndexBAPQ::search(
        faiss::idx_t n,
        const float* x,
        faiss::idx_t k,
        float* distances,
        faiss::idx_t* labels,
        const faiss::SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(!params, "IndexBAPQ: search params are not supported");
    FAISS_THROW_IF_NOT(is_trained);
    if (n <= 0 || k <= 0) {
        return;
    }

    const faiss::idx_t k_eff = std::min(k, ntotal);
    for (faiss::idx_t i = 0; i < n * k; ++i) {
        distances[i] = std::numeric_limits<float>::infinity();
        labels[i] = -1;
    }
    if (ntotal == 0 || k_eff == 0) {
        return;
    }

    RowMatrixXf x_pca(n, d);
    apply_transform_noalloc(n, x, x_pca.data());

    const size_t active_m = active_groups_.size();
    const int qbatch = std::max(1, query_batch);
    const int chunk_limit = std::max(1024, db_chunk);

#pragma omp parallel for schedule(dynamic)
    for (faiss::idx_t q0 = 0; q0 < n; q0 += qbatch) {
        const faiss::idx_t qb = std::min<faiss::idx_t>(qbatch, n - q0);
        std::vector<std::vector<float>> luts(active_m);
        for (size_t ai = 0; ai < active_m; ++ai) {
            const auto& group = groups_[static_cast<size_t>(active_groups_[ai])];
            auto& lut = luts[ai];
            lut.resize(static_cast<size_t>(qb) * static_cast<size_t>(group.ksub));
            for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                const float* qv = x_pca.row(static_cast<Eigen::Index>(q0 + qi)).data() +
                        group.begin;
                const float qnorm = squared_row_norm(qv, group.size);
                for (int c = 0; c < group.ksub; ++c) {
                    const float* cent =
                            group.codebook.row(c).data();
                    float dot = 0.0f;
                    for (int j = 0; j < group.size; ++j) {
                        dot += qv[j] * cent[j];
                    }
                    lut[static_cast<size_t>(qi) * static_cast<size_t>(group.ksub) +
                            static_cast<size_t>(c)] =
                            qnorm + group.centroid_norms[static_cast<size_t>(c)] -
                            2.0f * dot;
                }
            }
        }

        std::vector<float> heap_dist(static_cast<size_t>(qb) * static_cast<size_t>(k_eff));
        std::vector<faiss::idx_t> heap_ids(
                static_cast<size_t>(qb) * static_cast<size_t>(k_eff));
        for (faiss::idx_t qi = 0; qi < qb; ++qi) {
            float* hdist =
                    heap_dist.data() + static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
            faiss::idx_t* hids =
                    heap_ids.data() + static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
            std::fill(hdist, hdist + k_eff, std::numeric_limits<float>::infinity());
            std::fill(hids, hids + k_eff, faiss::idx_t{-1});
            faiss::maxheap_heapify(k_eff, hdist, hids);
        }

        std::vector<float> dist_chunk;
        for (faiss::idx_t b0 = 0; b0 < ntotal; b0 += chunk_limit) {
            const faiss::idx_t csz =
                    std::min<faiss::idx_t>(chunk_limit, ntotal - b0);
            dist_chunk.assign(
                    static_cast<size_t>(qb) * static_cast<size_t>(csz),
                    0.0f);
            for (size_t ai = 0; ai < active_m; ++ai) {
                const auto& group = groups_[static_cast<size_t>(active_groups_[ai])];
                const uint16_t* codes_group =
                        codes_.data() +
                        ai * code_capacity_ +
                        static_cast<size_t>(b0);
                const auto& lut = luts[ai];
                for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                    float* row =
                            dist_chunk.data() +
                            static_cast<size_t>(qi) * static_cast<size_t>(csz);
                    const float* lut_q =
                            lut.data() +
                            static_cast<size_t>(qi) * static_cast<size_t>(group.ksub);
                    for (faiss::idx_t j = 0; j < csz; ++j) {
                        row[static_cast<size_t>(j)] +=
                                lut_q[codes_group[static_cast<size_t>(j)]];
                    }
                }
            }

            for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                float* hdist =
                        heap_dist.data() +
                        static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
                faiss::idx_t* hids =
                        heap_ids.data() +
                        static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
                const float* row =
                        dist_chunk.data() +
                        static_cast<size_t>(qi) * static_cast<size_t>(csz);
                for (faiss::idx_t j = 0; j < csz; ++j) {
                    const float dis = row[static_cast<size_t>(j)];
                    if (dis < hdist[0]) {
                        faiss::maxheap_replace_top(
                                k_eff,
                                hdist,
                                hids,
                                dis,
                                b0 + j);
                    }
                }
            }
        }

        for (faiss::idx_t qi = 0; qi < qb; ++qi) {
            float* hdist =
                    heap_dist.data() + static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
            faiss::idx_t* hids =
                    heap_ids.data() + static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
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

void IndexBAPQ::reset() {
    ntotal = 0;
}

void IndexBAPQ::reconstruct_rows(
        const std::vector<faiss::idx_t>& ids,
        RowMatrixXf& out) const {
    FAISS_THROW_IF_NOT(is_trained);
    RowMatrixXf out_pca(static_cast<Eigen::Index>(ids.size()), d);
    out_pca.setZero();

    for (size_t gi = 0; gi < groups_.size(); ++gi) {
        const auto& group = groups_[gi];
        if (group.active) {
            const uint16_t* codes_group = codes_.data() +
                    static_cast<size_t>(group.active_slot) *
                            code_capacity_;
            for (size_t row = 0; row < ids.size(); ++row) {
                const faiss::idx_t id = ids[row];
                FAISS_THROW_IF_NOT(id >= 0 && id < ntotal);
                out_pca.block(
                               static_cast<Eigen::Index>(row),
                               group.begin,
                               1,
                               group.size) =
                        group.codebook.row(
                                codes_group[static_cast<size_t>(id)]);
            }
        } else {
            for (size_t row = 0; row < ids.size(); ++row) {
                out_pca.block(
                               static_cast<Eigen::Index>(row),
                               group.begin,
                               1,
                               group.size) = group.codebook.row(0);
            }
        }
    }

    out.resize(static_cast<Eigen::Index>(ids.size()), d);
    pca_->reverse_transform(
            static_cast<faiss::idx_t>(ids.size()),
            out_pca.data(),
            out.data());
}

void IndexBAPQ::reconstruct(faiss::idx_t key, float* recons) const {
    std::vector<faiss::idx_t> ids = {key};
    RowMatrixXf out;
    reconstruct_rows(ids, out);
    std::memcpy(recons, out.data(), static_cast<size_t>(d) * sizeof(float));
}

size_t IndexBAPQ::sa_code_size() const {
    return packed_code_size_bytes(total_bits);
}

size_t IndexBAPQ::adc_lut_size() const noexcept {
    size_t total = 0;
    for (const int group_id : active_groups_) {
        total += static_cast<size_t>(groups_[static_cast<size_t>(group_id)].ksub);
    }
    return total;
}

void IndexBAPQ::compute_adc_lut(const float* query, float* lut) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "IndexBAPQ: compute_adc_lut requires trained index");
    FAISS_THROW_IF_NOT_MSG(query != nullptr, "IndexBAPQ: compute_adc_lut requires query");
    FAISS_THROW_IF_NOT_MSG(lut != nullptr, "IndexBAPQ: compute_adc_lut requires output buffer");

    std::vector<float> query_pca(static_cast<size_t>(d));
    transform_vector(query, query_pca.data());
    compute_adc_lut_from_transformed(query_pca.data(), lut);
}

void IndexBAPQ::transform_vector(const float* x, float* out) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "IndexBAPQ: transform_vector requires trained index");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "IndexBAPQ: transform_vector requires input");
    FAISS_THROW_IF_NOT_MSG(out != nullptr, "IndexBAPQ: transform_vector requires output");
    apply_transform_noalloc(1, x, out);
}

void IndexBAPQ::compute_adc_lut_from_transformed(
        const float* query_transformed,
        float* lut) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "IndexBAPQ: compute_adc_lut_from_transformed requires trained index");
    FAISS_THROW_IF_NOT_MSG(
            query_transformed != nullptr,
            "IndexBAPQ: compute_adc_lut_from_transformed requires query");
    FAISS_THROW_IF_NOT_MSG(
            lut != nullptr,
            "IndexBAPQ: compute_adc_lut_from_transformed requires output buffer");

    size_t lut_offset = 0;
    for (size_t ai = 0; ai < active_groups_.size(); ++ai) {
        const auto& group = groups_[static_cast<size_t>(active_groups_[ai])];
        const float* qv = query_transformed + group.begin;
        const float qnorm = squared_row_norm(qv, group.size);
        for (int c = 0; c < group.ksub; ++c) {
            const float* cent = group.codebook.row(c).data();
            float dot = 0.0f;
            for (int j = 0; j < group.size; ++j) {
                dot += qv[j] * cent[j];
            }
            lut[lut_offset + static_cast<size_t>(c)] =
                    qnorm + group.centroid_norms[static_cast<size_t>(c)] -
                    2.0f * dot;
        }
        lut_offset += static_cast<size_t>(group.ksub);
    }
}

float IndexBAPQ::adc_distance_from_packed_code(
        const uint8_t* code,
        const float* lut) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "IndexBAPQ: adc_distance_from_packed_code requires trained index");
    FAISS_THROW_IF_NOT_MSG(code != nullptr, "IndexBAPQ: adc_distance_from_packed_code requires code");
    FAISS_THROW_IF_NOT_MSG(lut != nullptr, "IndexBAPQ: adc_distance_from_packed_code requires LUT");

    float dist = 0.0f;
    uint32_t current = 0;
    int bits_avail = 0;
    size_t byte_pos = 0;
    size_t lut_offset = 0;

    for (size_t ai = 0; ai < active_groups_.size(); ++ai) {
        const auto& group = groups_[static_cast<size_t>(active_groups_[ai])];
        uint32_t packed = 0;
        int bits_written = 0;
        while (bits_written < group.nbits) {
            if (bits_avail == 0) {
                current = code[byte_pos++];
                bits_avail = 8;
            }
            const int take = std::min(bits_avail, group.nbits - bits_written);
            const uint32_t mask = (uint32_t{1} << take) - 1U;
            packed |= (current & mask) << bits_written;
            current >>= take;
            bits_avail -= take;
            bits_written += take;
        }
        dist += lut[lut_offset + static_cast<size_t>(packed)];
        lut_offset += static_cast<size_t>(group.ksub);
    }
    return dist;
}

void IndexBAPQ::sa_encode(
        faiss::idx_t n,
        const float* x,
        uint8_t* bytes) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "IndexBAPQ: sa_encode requires trained index");
    if (n <= 0) {
        return;
    }

    static thread_local std::vector<float> x_pca_storage;
    static thread_local std::vector<uint16_t> row_codes;
    x_pca_storage.resize(static_cast<size_t>(n) * static_cast<size_t>(d));
    apply_transform_noalloc(n, x, x_pca_storage.data());
    const float* x_pca = x_pca_storage.data();
    const size_t code_size = sa_code_size();

    const size_t active_m = active_groups_.size();
    row_codes.resize(active_m * static_cast<size_t>(n));
    uint16_t* row_codes_data = row_codes.data();
    #pragma omp parallel for if (active_m > 1)
    for (faiss::idx_t ai = 0; ai < static_cast<faiss::idx_t>(active_m); ++ai) {
        assign_active_group_codes_transformed(
                n,
                x_pca,
                static_cast<size_t>(ai),
                row_codes_data + static_cast<size_t>(ai) * static_cast<size_t>(n));
    }

    for (faiss::idx_t row = 0; row < n; ++row) {
        uint8_t current = 0;
        int bits_filled = 0;
        uint8_t* dst = bytes + static_cast<size_t>(row) * code_size;
        size_t out_pos = 0;
        for (size_t ai = 0; ai < active_m; ++ai) {
            const auto& group = groups_[static_cast<size_t>(active_groups_[ai])];
            uint32_t code = static_cast<uint32_t>(
                    row_codes[ai * static_cast<size_t>(n) + static_cast<size_t>(row)]);
            int remaining = group.nbits;
            while (remaining > 0) {
                const int take = std::min(8 - bits_filled, remaining);
                const uint8_t mask =
                        static_cast<uint8_t>((uint32_t{1} << take) - 1);
                current |= static_cast<uint8_t>(code & mask) << bits_filled;
                code >>= take;
                remaining -= take;
                bits_filled += take;
                if (bits_filled == 8) {
                    dst[out_pos++] = current;
                    current = 0;
                    bits_filled = 0;
                }
            }
        }
        if (bits_filled > 0) {
            dst[out_pos++] = current;
        }
        FAISS_THROW_IF_NOT(out_pos == code_size);
    }
}

void IndexBAPQ::sa_decode(
        faiss::idx_t n,
        const uint8_t* bytes,
        float* x) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "IndexBAPQ: sa_decode requires trained index");
    if (n <= 0) {
        return;
    }

    RowMatrixXf out_pca(n, d);
    out_pca.setZero();
    const size_t code_size = sa_code_size();
    for (faiss::idx_t row = 0; row < n; ++row) {
        const uint8_t* src = bytes + static_cast<size_t>(row) * code_size;
        size_t byte_pos = 0;
        uint32_t current = code_size > 0 ? src[0] : 0;
        int bits_avail = code_size > 0 ? 8 : 0;
        for (size_t ai = 0; ai < active_groups_.size(); ++ai) {
            const auto& group = groups_[static_cast<size_t>(active_groups_[ai])];
            uint32_t code = 0;
            int bits_written = 0;
            while (bits_written < group.nbits) {
                if (bits_avail == 0) {
                    ++byte_pos;
                    FAISS_THROW_IF_NOT(byte_pos < code_size);
                    current = src[byte_pos];
                    bits_avail = 8;
                }
                const int take = std::min(bits_avail, group.nbits - bits_written);
                const uint32_t mask = (uint32_t{1} << take) - 1;
                code |= (current & mask) << bits_written;
                current >>= take;
                bits_avail -= take;
                bits_written += take;
            }
            const auto& codebook = group.codebook;
            out_pca.block(
                           static_cast<Eigen::Index>(row),
                           group.begin,
                           1,
                           group.size) = codebook.row(static_cast<Eigen::Index>(code));
        }
        for (const auto& group : groups_) {
            if (group.active) {
                continue;
            }
            out_pca.block(
                           static_cast<Eigen::Index>(row),
                           group.begin,
                           1,
                           group.size) = group.codebook.row(0);
        }
    }

    Eigen::Map<RowMatrixXf> out(x, n, d);
    pca_->reverse_transform(n, out_pca.data(), out.data());
}

void IndexBAPQ::rerank_candidates(
        const float* query,
        const faiss::idx_t* candidate_ids,
        size_t ncandidates,
        faiss::idx_t k,
        float* distances,
        faiss::idx_t* labels) const {
    FAISS_THROW_IF_NOT(is_trained);
    if (k <= 0) {
        return;
    }
    const faiss::idx_t k_eff = std::min<faiss::idx_t>(k, ncandidates);
    std::fill(distances, distances + k, std::numeric_limits<float>::infinity());
    std::fill(labels, labels + k, faiss::idx_t{-1});
    if (ncandidates == 0 || k_eff == 0) {
        return;
    }

    std::vector<float> query_pca(static_cast<size_t>(d));
    apply_transform_noalloc(1, query, query_pca.data());

    std::vector<std::vector<float>> luts(active_groups_.size());
    for (size_t ai = 0; ai < active_groups_.size(); ++ai) {
        const auto& group = groups_[static_cast<size_t>(active_groups_[ai])];
        auto& lut = luts[ai];
        lut.resize(static_cast<size_t>(group.ksub));
        const float* qv = query_pca.data() + group.begin;
        const float qnorm = squared_row_norm(qv, group.size);
        for (int c = 0; c < group.ksub; ++c) {
            const float* cent = group.codebook.row(c).data();
            float dot = 0.0f;
            for (int j = 0; j < group.size; ++j) {
                dot += qv[j] * cent[j];
            }
            lut[static_cast<size_t>(c)] =
                    qnorm + group.centroid_norms[static_cast<size_t>(c)] -
                    2.0f * dot;
        }
    }

    faiss::maxheap_heapify(k_eff, distances, labels);
    for (size_t i = 0; i < ncandidates; ++i) {
        const faiss::idx_t id = candidate_ids[i];
        FAISS_THROW_IF_NOT(id >= 0 && id < ntotal);
        float dist = 0.0f;
        for (size_t ai = 0; ai < active_groups_.size(); ++ai) {
            const auto& group = groups_[static_cast<size_t>(active_groups_[ai])];
            const uint16_t code =
                    codes_[ai * code_capacity_ +
                           static_cast<size_t>(id)];
            dist += luts[ai][static_cast<size_t>(code)];
        }
        if (dist < distances[0]) {
            faiss::maxheap_replace_top(k_eff, distances, labels, dist, id);
        }
    }
    faiss::maxheap_reorder(k_eff, distances, labels);
}

int IndexBAPQ::component_count() const noexcept {
    return component_count_;
}

int IndexBAPQ::active_component_count() const noexcept {
    return static_cast<int>(active_groups_.size());
}

const std::vector<int>& IndexBAPQ::nbits_per_group() const noexcept {
    return nbits_per_group_;
}

const std::vector<int>& IndexBAPQ::group_sizes() const noexcept {
    return group_sizes_;
}

const BAPQTrainingStats& IndexBAPQ::training_stats() const noexcept {
    return training_stats_;
}

size_t IndexBAPQ::theoretical_code_bytes() const noexcept {
    return (static_cast<size_t>(ntotal) * static_cast<size_t>(total_bits) + 7) / 8;
}

size_t IndexBAPQ::codebook_bytes() const noexcept {
    size_t total = 0;
    for (const auto& group : groups_) {
        total += static_cast<size_t>(group.codebook.size()) * sizeof(float);
    }
    return total;
}

size_t IndexBAPQ::transform_bytes() const noexcept {
    if (!pca_) {
        return 0;
    }
    return static_cast<size_t>(d) * static_cast<size_t>(d + 1) * sizeof(float);
}

void IndexBAPQ::serialize_payload(faiss::IOWriter& writer) const {
    static constexpr char kMagic[] = "BAPQv1";
    write_vector_data(writer, kMagic, sizeof(kMagic));
    write_scalar<int32_t>(writer, d);
    write_scalar<int32_t>(writer, total_bits);
    write_scalar<int32_t>(writer, subspace_dim);
    write_scalar<int32_t>(writer, bmax);
    write_scalar<int64_t>(writer, ntotal);
    write_scalar<uint64_t>(writer, static_cast<uint64_t>(groups_.size()));
    write_scalar<uint64_t>(writer, static_cast<uint64_t>(active_groups_.size()));
    for (const auto& group : groups_) {
        write_scalar<int32_t>(writer, group.begin);
        write_scalar<int32_t>(writer, group.size);
        write_scalar<int32_t>(writer, group.nbits);
        write_scalar<int32_t>(writer, group.ksub);
        write_scalar<uint8_t>(writer, group.active ? 1 : 0);
        write_scalar<int32_t>(writer, group.active_slot);
        write_scalar<int64_t>(writer, group.codebook.rows());
        write_scalar<int64_t>(writer, group.codebook.cols());
        write_vector_data(
                writer,
                group.codebook.data(),
                static_cast<size_t>(group.codebook.size()));
    }
    write_vector<int>(writer, active_groups_);
    write_vector<int>(writer, nbits_per_group_);
    write_vector<int>(writer, group_sizes_);
    write_scalar<uint8_t>(writer, pca_ ? 1 : 0);
    if (pca_) {
        faiss::write_VectorTransform(pca_.get(), &writer);
    }
    const size_t expected_bytes = theoretical_code_bytes();
    write_scalar<uint64_t>(writer, static_cast<uint64_t>(expected_bytes));
    uint8_t current = 0;
    int bits_filled = 0;
    size_t produced = 0;
    for (faiss::idx_t row = 0; row < ntotal; ++row) {
        for (size_t ai = 0; ai < active_groups_.size(); ++ai) {
            const auto& group = groups_[static_cast<size_t>(active_groups_[ai])];
            uint32_t code =
                    static_cast<uint32_t>(
                            codes_[ai * code_capacity_ +
                                   static_cast<size_t>(row)]);
            int remaining = group.nbits;
            while (remaining > 0) {
                const int take = std::min(8 - bits_filled, remaining);
                const uint8_t mask =
                        static_cast<uint8_t>((uint32_t{1} << take) - 1);
                current |= static_cast<uint8_t>(code & mask) << bits_filled;
                code >>= take;
                remaining -= take;
                bits_filled += take;
                if (bits_filled == 8) {
                    write_vector_data(writer, &current, 1);
                    produced += 1;
                    current = 0;
                    bits_filled = 0;
                }
            }
        }
    }
    if (bits_filled > 0) {
        write_vector_data(writer, &current, 1);
        produced += 1;
    }
    FAISS_THROW_IF_NOT(produced == expected_bytes);
}

size_t IndexBAPQ::serialized_payload_bytes() const {
    CountingIOWriter writer;
    serialize_payload(writer);
    return writer.bytes_written;
}

}  // namespace epq
