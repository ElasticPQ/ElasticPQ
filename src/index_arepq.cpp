#include "epq/index_arepq.h"
#include "epq/serialization_size.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>

namespace epq {
namespace {

RowMatrixXf train_residual_tail_codebook(
        const RowMatrixXf& x,
        int k,
        int niter,
        int nredo) {
    if (x.rows() <= 0 || x.cols() <= 0) {
        throw std::invalid_argument(
                "IndexAREPQ: residual tail k-means requires non-empty data");
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

std::vector<uint16_t> assign_residual_tail(
        const RowMatrixXf& residuals,
        const RowMatrixXf& codebook) {
    faiss::IndexFlatL2 assign_index(codebook.cols());
    assign_index.add(codebook.rows(), codebook.data());
    std::vector<float> distances(static_cast<size_t>(residuals.rows()));
    std::vector<faiss::idx_t> labels(static_cast<size_t>(residuals.rows()));
    assign_index.search(
            residuals.rows(),
            residuals.data(),
            1,
            distances.data(),
            labels.data());
    std::vector<uint16_t> out(static_cast<size_t>(residuals.rows()));
    for (Eigen::Index i = 0; i < residuals.rows(); ++i) {
        out[static_cast<size_t>(i)] =
                static_cast<uint16_t>(labels[static_cast<size_t>(i)]);
    }
    return out;
}

std::vector<uint16_t> assign_residual_tail_topk(
        const RowMatrixXf& residuals,
        const RowMatrixXf& codebook,
        int k) {
    k = std::clamp(k, 1, static_cast<int>(codebook.rows()));
    faiss::IndexFlatL2 assign_index(codebook.cols());
    assign_index.add(codebook.rows(), codebook.data());
    std::vector<float> distances(
            static_cast<size_t>(residuals.rows()) * static_cast<size_t>(k));
    std::vector<faiss::idx_t> labels(
            static_cast<size_t>(residuals.rows()) * static_cast<size_t>(k));
    assign_index.search(
            residuals.rows(),
            residuals.data(),
            k,
            distances.data(),
            labels.data());
    std::vector<uint16_t> out(
            static_cast<size_t>(residuals.rows()) * static_cast<size_t>(k));
    for (Eigen::Index i = 0; i < residuals.rows(); ++i) {
        for (int j = 0; j < k; ++j) {
            out[static_cast<size_t>(i) * static_cast<size_t>(k) +
                static_cast<size_t>(j)] =
                    static_cast<uint16_t>(
                            labels[static_cast<size_t>(i) *
                                           static_cast<size_t>(k) +
                                   static_cast<size_t>(j)]);
        }
    }
    return out;
}

template <bool Add>
inline void accumulate_u16_lut_row(
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

inline void write_bits(uint8_t* bytes, int& bit_offset, uint32_t value, int nbits) {
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

inline uint32_t read_bits(const uint8_t* bytes, int& bit_offset, int nbits) {
    uint32_t value = 0;
    int read = 0;
    while (read < nbits) {
        const int byte_index = bit_offset / 8;
        const int bit_in_byte = bit_offset % 8;
        const int take = std::min(nbits - read, 8 - bit_in_byte);
        const uint32_t mask = (uint32_t{1} << take) - 1U;
        value |=
                ((static_cast<uint32_t>(bytes[byte_index]) >> bit_in_byte) & mask)
                << read;
        read += take;
        bit_offset += take;
    }
    return value;
}

RowMatrixXf transform_rows_with_epq(
        const IndexEPQ& index,
        const RowMatrixXf& x) {
    RowMatrixXf out(x.rows(), x.cols());
    #pragma omp parallel for schedule(static)
    for (Eigen::Index row = 0; row < x.rows(); ++row) {
        index.transform_vector(
                x.data() + static_cast<size_t>(row) * static_cast<size_t>(x.cols()),
                out.data() + static_cast<size_t>(row) * static_cast<size_t>(x.cols()));
    }
    return out;
}

RowMatrixXf build_epq_transform_matrix(const IndexEPQ& index, int d) {
    RowMatrixXf matrix(d, d);
    std::vector<float> basis(static_cast<size_t>(d), 0.0f);
    std::vector<float> transformed(static_cast<size_t>(d), 0.0f);
    for (int dim = 0; dim < d; ++dim) {
        std::fill(basis.begin(), basis.end(), 0.0f);
        basis[static_cast<size_t>(dim)] = 1.0f;
        index.transform_vector(basis.data(), transformed.data());
        for (int j = 0; j < d; ++j) {
            matrix(dim, j) = transformed[static_cast<size_t>(j)];
        }
    }
    return matrix;
}

void decode_main_assignments_transformed(
        const IndexEPQ& index,
        faiss::idx_t n,
        const uint16_t* assignments,
        RowMatrixXf& out) {
    const size_t m = index.structure().group_count();
    const auto& groups = index.active_groups();
    const auto& codebooks = index.codebooks();
    out.resize(n, index.d);
    out.setZero();
    for (faiss::idx_t row = 0; row < n; ++row) {
        const size_t row_offset = static_cast<size_t>(row) * m;
        for (size_t gi = 0; gi < m; ++gi) {
            const uint16_t code = assignments[row_offset + gi];
            const auto& dims = groups[gi];
            const auto& codebook = codebooks[gi];
            for (size_t j = 0; j < dims.size(); ++j) {
                out(row, dims[j]) = codebook(
                        static_cast<Eigen::Index>(code),
                        static_cast<Eigen::Index>(j));
            }
        }
    }
}

RowMatrixXf gather_columns_local(
        const RowMatrixXf& x,
        const std::vector<int>& dims) {
    RowMatrixXf out(x.rows(), static_cast<Eigen::Index>(dims.size()));
    for (size_t j = 0; j < dims.size(); ++j) {
        out.col(static_cast<Eigen::Index>(j)) = x.col(dims[j]);
    }
    return out;
}

void assign_main_codes_for_transformed_targets(
        const IndexEPQ& index,
        const RowMatrixXf& target_y,
        std::vector<uint16_t>& assignments) {
    const size_t m = index.structure().group_count();
    const auto& groups = index.active_groups();
    const auto& codebooks = index.codebooks();
    assignments.assign(static_cast<size_t>(target_y.rows()) * m, uint16_t{0});
    for (size_t gi = 0; gi < m; ++gi) {
        const auto& codebook = codebooks[gi];
        RowMatrixXf sub = gather_columns_local(target_y, groups[gi]);
        faiss::IndexFlatL2 assign_index(codebook.cols());
        assign_index.add(codebook.rows(), codebook.data());
        std::vector<float> distances(static_cast<size_t>(target_y.rows()));
        std::vector<faiss::idx_t> labels(static_cast<size_t>(target_y.rows()));
        assign_index.search(
                target_y.rows(),
                sub.data(),
                1,
                distances.data(),
                labels.data());
        for (Eigen::Index row = 0; row < target_y.rows(); ++row) {
            assignments[static_cast<size_t>(row) * m + gi] =
                    static_cast<uint16_t>(labels[static_cast<size_t>(row)]);
        }
    }
}

}  // namespace

size_t TailMemoryStats::serialized_tail_bytes() const noexcept {
    return payload_code_bytes + serialized_codebook_bytes;
}

size_t TailMemoryStats::resident_search_model_bytes() const noexcept {
    return serialized_codebook_bytes + norm_table_bytes +
            product_tail_table_bytes + tail_pair_table_bytes;
}

size_t TailMemoryStats::resident_auxiliary_table_bytes() const noexcept {
    return norm_table_bytes + product_tail_table_bytes + tail_pair_table_bytes;
}

size_t TailMemoryStats::resident_model_bytes() const noexcept {
    return resident_search_model_bytes() + reconstruction_codebook_bytes +
            transform_copy_bytes;
}

IndexAREPQ::IndexAREPQ(
        int d_in,
        int total_bits_in,
        int tail_bits_in,
        int tail_stages_in,
        std::shared_ptr<StructureBuilder> structure_builder)
        : faiss::Index(d_in, faiss::METRIC_L2),
          total_bits(total_bits_in),
          tail_bits(tail_bits_in),
          tail_stages(tail_stages_in),
          main_bits(total_bits_in - tail_bits_in * tail_stages_in),
          tail_ksub(
                  tail_bits_in > 0 && tail_bits_in < 31
                          ? (1 << tail_bits_in)
                          : 0),
          main_(d_in, total_bits_in - tail_bits_in * tail_stages_in, std::move(structure_builder)) {
    is_trained = false;
    main_.use_uneven_transform = true;
}

IndexEPQ& IndexAREPQ::main_index() noexcept {
    return main_;
}

const IndexEPQ& IndexAREPQ::main_index() const noexcept {
    return main_;
}

int IndexAREPQ::component_count() const {
    return static_cast<int>(main_.structure().group_count()) + tail_stages;
}

int IndexAREPQ::effective_budget_bits() const noexcept {
    return total_bits;
}

const TrainingStats& IndexAREPQ::training_stats() const noexcept {
    return training_stats_;
}

double IndexAREPQ::tail_train_time() const noexcept {
    return tail_train_time_;
}

double IndexAREPQ::tail_alt_initial_mse() const noexcept {
    return tail_alt_initial_mse_;
}

double IndexAREPQ::tail_alt_best_mse() const noexcept {
    return tail_alt_best_mse_;
}

double IndexAREPQ::tail_alt_final_mse() const noexcept {
    return tail_alt_final_mse_;
}

void IndexAREPQ::validate_config() const {
    if (d <= 0) {
        throw std::invalid_argument("IndexAREPQ: d must be positive");
    }
    if (metric_type != faiss::METRIC_L2) {
        throw std::invalid_argument("IndexAREPQ: only METRIC_L2 is supported");
    }
    if (tail_bits <= 0 || tail_bits > 12 || tail_stages <= 0 || main_bits <= 0) {
        throw std::invalid_argument(
                "IndexAREPQ: requires positive main bits, tail stages, and 1..12 tail bits");
    }
    if (tail_stages > 64) {
        throw std::invalid_argument(
                "IndexAREPQ: tail_stages > 64 is not supported by the IVF codec");
    }
}

void IndexAREPQ::train(faiss::idx_t n, const float* x) {
    validate_config();
    FAISS_THROW_IF_NOT_MSG(n > 0, "IndexAREPQ::train requires non-empty training data");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "IndexAREPQ::train requires non-null input");

    const auto total_t0 = std::chrono::steady_clock::now();
    main_.train(n, x);
    const auto main_stats = main_.training_stats();

    const auto tail_t0 = std::chrono::steady_clock::now();
    transform_matrix_ = build_epq_transform_matrix(main_, d);
    const Eigen::Map<const RowMatrixXf> xt_map(x, n, d);
    const RowMatrixXf xt = xt_map;
    const RowMatrixXf train_y = transform_rows_with_epq(main_, xt);
    const auto initial_main_codes = main_.compute_assignments(n, x);
    RowMatrixXf main_y;
    decode_main_assignments_transformed(
            main_,
            n,
            initial_main_codes.data(),
            main_y);

    RowMatrixXf residual = train_y - main_y;
    tail_codebooks_y_.clear();
    tail_codebooks_y_.reserve(static_cast<size_t>(tail_stages));
    tail_alt_initial_mse_ = std::numeric_limits<double>::quiet_NaN();
    tail_alt_best_mse_ = std::numeric_limits<double>::quiet_NaN();
    tail_alt_final_mse_ = std::numeric_limits<double>::quiet_NaN();
    for (int stage = 0; stage < tail_stages; ++stage) {
        RowMatrixXf codebook = train_residual_tail_codebook(
                residual,
                tail_ksub,
                tail_kmeans_niter,
                tail_kmeans_nredo);
        const auto stage_codes = assign_residual_tail(residual, codebook);
        for (Eigen::Index row = 0; row < residual.rows(); ++row) {
            residual.row(row) -= codebook.row(stage_codes[static_cast<size_t>(row)]);
        }
        tail_codebooks_y_.push_back(std::move(codebook));
    }
    build_tail_auxiliary_tables();
    run_tail_alternating_optimization(train_y, initial_main_codes);
    tail_train_time_ = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - tail_t0)
                               .count();

    training_stats_ = main_stats;
    training_stats_.codebook_time += tail_train_time_;
    training_stats_.total_time =
            std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - total_t0)
                    .count();
    is_trained = true;
    ntotal = 0;
}

void IndexAREPQ::add(faiss::idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "IndexAREPQ must be trained before add");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "IndexAREPQ::add requires non-null input");
    if (n <= 0) {
        return;
    }

    const size_t m = main_.structure().group_count();
    const faiss::idx_t old_total = ntotal;
    const faiss::idx_t new_total = old_total + n;
    if (old_total == 0) {
        main_codes_by_group_.assign(m * static_cast<size_t>(new_total), uint16_t{0});
        tail_codes_by_stage_.assign(
                static_cast<size_t>(tail_stages),
                std::vector<uint16_t>(static_cast<size_t>(new_total), uint16_t{0}));
    } else {
        std::vector<uint16_t> new_main(
                m * static_cast<size_t>(new_total),
                uint16_t{0});
        for (size_t gi = 0; gi < m; ++gi) {
            std::copy(
                    main_codes_by_group_.begin() +
                            static_cast<std::ptrdiff_t>(gi * static_cast<size_t>(old_total)),
                    main_codes_by_group_.begin() +
                            static_cast<std::ptrdiff_t>((gi + 1) * static_cast<size_t>(old_total)),
                    new_main.begin() +
                            static_cast<std::ptrdiff_t>(gi * static_cast<size_t>(new_total)));
        }
        main_codes_by_group_.swap(new_main);
        for (auto& codes : tail_codes_by_stage_) {
            codes.resize(static_cast<size_t>(new_total), uint16_t{0});
        }
    }

    for (faiss::idx_t row0 = 0; row0 < n; row0 += add_batch_rows) {
        const faiss::idx_t rows =
                std::min<faiss::idx_t>(add_batch_rows, n - row0);
        const Eigen::Map<const RowMatrixXf> batch_map(
                x + static_cast<size_t>(row0) * static_cast<size_t>(d),
                rows,
                d);
        const RowMatrixXf x_batch = batch_map;
        std::vector<uint16_t> main_codes;
        std::vector<std::vector<uint16_t>> tail_codes;
        encode_batch_joint(x_batch, main_codes, tail_codes);

        const faiss::idx_t dst0 = old_total + row0;
        for (size_t gi = 0; gi < m; ++gi) {
            uint16_t* dst = main_codes_by_group_.data() +
                    gi * static_cast<size_t>(new_total) +
                    static_cast<size_t>(dst0);
            for (faiss::idx_t row = 0; row < rows; ++row) {
                dst[static_cast<size_t>(row)] =
                        main_codes[static_cast<size_t>(row) * m + gi];
            }
        }
        for (int stage = 0; stage < tail_stages; ++stage) {
            std::copy(
                    tail_codes[static_cast<size_t>(stage)].begin(),
                    tail_codes[static_cast<size_t>(stage)].end(),
                    tail_codes_by_stage_[static_cast<size_t>(stage)].begin() +
                            dst0);
        }
    }
    ntotal = new_total;
}

void IndexAREPQ::search(
        faiss::idx_t n,
        const float* x,
        faiss::idx_t k,
        float* distances,
        faiss::idx_t* labels,
        const faiss::SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "IndexAREPQ must be trained before search");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "IndexAREPQ::search requires non-null input");
    FAISS_THROW_IF_NOT_MSG(distances != nullptr, "IndexAREPQ::search requires distances");
    FAISS_THROW_IF_NOT_MSG(labels != nullptr, "IndexAREPQ::search requires labels");
    FAISS_THROW_IF_NOT_MSG(k > 0, "IndexAREPQ::search requires positive k");
    if (const auto* epq_params = dynamic_cast<const SearchParametersEPQ*>(params);
        epq_params != nullptr && epq_params->mode != SearchMode::kADC) {
        throw std::runtime_error("IndexAREPQ currently supports ADC mode only");
    }

    const faiss::idx_t k_eff = std::min<faiss::idx_t>(k, ntotal);
    for (faiss::idx_t i = 0; i < n * k; ++i) {
        distances[i] = std::numeric_limits<float>::infinity();
        labels[i] = faiss::idx_t{-1};
    }
    if (n <= 0 || k_eff <= 0) {
        return;
    }

    const size_t m = main_.structure().group_count();
    const auto& codebooks = main_.codebooks();
    std::vector<size_t> lut_offsets(m + 1, 0);
    for (size_t gi = 0; gi < m; ++gi) {
        lut_offsets[gi + 1] =
                lut_offsets[gi] + static_cast<size_t>(codebooks[gi].rows());
    }
    const size_t flat_lut_size = lut_offsets.back();

    #pragma omp parallel for schedule(dynamic)
    for (faiss::idx_t q0 = 0; q0 < n; q0 += search_query_batch) {
        const faiss::idx_t qb =
                std::min<faiss::idx_t>(search_query_batch, n - q0);
        std::vector<float> q_trans(static_cast<size_t>(qb) * static_cast<size_t>(d));
        for (faiss::idx_t qi = 0; qi < qb; ++qi) {
            main_.transform_vector(
                    x + static_cast<size_t>(q0 + qi) * static_cast<size_t>(d),
                    q_trans.data() + static_cast<size_t>(qi) * static_cast<size_t>(d));
        }

        std::vector<std::vector<float>> main_luts(m);
        for (size_t gi = 0; gi < m; ++gi) {
            main_luts[gi].resize(
                    static_cast<size_t>(qb) *
                    static_cast<size_t>(codebooks[gi].rows()));
        }
        std::vector<float> flat_lut(flat_lut_size);
        for (faiss::idx_t qi = 0; qi < qb; ++qi) {
            main_.compute_adc_lut_from_transformed(
                    q_trans.data() + static_cast<size_t>(qi) * static_cast<size_t>(d),
                    flat_lut.data());
            for (size_t gi = 0; gi < m; ++gi) {
                const size_t ksub = static_cast<size_t>(codebooks[gi].rows());
                std::copy(
                        flat_lut.begin() +
                                static_cast<std::ptrdiff_t>(lut_offsets[gi]),
                        flat_lut.begin() +
                                static_cast<std::ptrdiff_t>(lut_offsets[gi] + ksub),
                        main_luts[gi].begin() +
                                static_cast<std::ptrdiff_t>(
                                        static_cast<size_t>(qi) * ksub));
            }
        }

        std::vector<std::vector<float>> tail_luts(
                static_cast<size_t>(tail_stages));
        for (int stage = 0; stage < tail_stages; ++stage) {
            auto& lut = tail_luts[static_cast<size_t>(stage)];
            lut.resize(static_cast<size_t>(qb) * static_cast<size_t>(tail_ksub));
            const auto& codebook = tail_codebooks_y_[static_cast<size_t>(stage)];
            const auto& norms = tail_norms_[static_cast<size_t>(stage)];
            for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                const float* qrow =
                        q_trans.data() + static_cast<size_t>(qi) * static_cast<size_t>(d);
                float* lut_row =
                        lut.data() + static_cast<size_t>(qi) * static_cast<size_t>(tail_ksub);
                for (int code = 0; code < tail_ksub; ++code) {
                    float dot = 0.0f;
                    for (int dim = 0; dim < d; ++dim) {
                        dot += qrow[dim] * codebook(code, dim);
                    }
                    lut_row[code] =
                            norms[static_cast<size_t>(code)] - 2.0f * dot;
                }
            }
        }

        std::vector<float> heap_dist(
                static_cast<size_t>(qb) * static_cast<size_t>(k_eff));
        std::vector<faiss::idx_t> heap_ids(
                static_cast<size_t>(qb) * static_cast<size_t>(k_eff));
        for (faiss::idx_t qi = 0; qi < qb; ++qi) {
            float* hdist =
                    heap_dist.data() +
                    static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
            faiss::idx_t* hids =
                    heap_ids.data() +
                    static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
            std::fill(
                    hdist,
                    hdist + k_eff,
                    std::numeric_limits<float>::infinity());
            std::fill(hids, hids + k_eff, faiss::idx_t{-1});
            faiss::maxheap_heapify(k_eff, hdist, hids);
        }

        std::vector<float> dist_chunk(
                static_cast<size_t>(qb) * static_cast<size_t>(search_db_chunk));
        for (faiss::idx_t b0 = 0; b0 < ntotal; b0 += search_db_chunk) {
            const faiss::idx_t csz =
                    std::min<faiss::idx_t>(search_db_chunk, ntotal - b0);
            dist_chunk.resize(static_cast<size_t>(qb) * static_cast<size_t>(csz));
            for (size_t gi = 0; gi < m; ++gi) {
                const uint16_t* codes =
                        main_codes_by_group_.data() +
                        gi * static_cast<size_t>(ntotal) +
                        static_cast<size_t>(b0);
                const size_t ksub =
                        static_cast<size_t>(codebooks[gi].rows());
                for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                    float* row =
                            dist_chunk.data() +
                            static_cast<size_t>(qi) * static_cast<size_t>(csz);
                    const float* lut =
                            main_luts[gi].data() +
                            static_cast<size_t>(qi) * ksub;
                    if (gi == 0) {
                        accumulate_u16_lut_row<false>(row, codes, lut, csz);
                    } else {
                        accumulate_u16_lut_row<true>(row, codes, lut, csz);
                    }
                }
            }

            add_tail_terms_to_chunk(qb, b0, csz, tail_luts, dist_chunk);

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
                    heap_dist.data() +
                    static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
            faiss::idx_t* hids =
                    heap_ids.data() +
                    static_cast<size_t>(qi) * static_cast<size_t>(k_eff);
            faiss::maxheap_reorder(k_eff, hdist, hids);
            std::copy(
                    hdist,
                    hdist + k_eff,
                    distances + static_cast<size_t>(q0 + qi) * static_cast<size_t>(k));
            std::copy(
                    hids,
                    hids + k_eff,
                    labels + static_cast<size_t>(q0 + qi) * static_cast<size_t>(k));
        }
    }
}

void IndexAREPQ::reset() {
    main_codes_by_group_.clear();
    tail_codes_by_stage_.clear();
    ntotal = 0;
}

void IndexAREPQ::reconstruct(faiss::idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(
            key >= 0 && key < ntotal,
            "IndexAREPQ: reconstruct key out of range");
    std::vector<faiss::idx_t> ids{key};
    RowMatrixXf out;
    reconstruct_rows(ids, out);
    std::memcpy(recons, out.data(), static_cast<size_t>(d) * sizeof(float));
}

void IndexAREPQ::reconstruct_rows(
        const std::vector<faiss::idx_t>& ids,
        RowMatrixXf& out) const {
    const size_t m = main_.structure().group_count();
    std::vector<uint16_t> assignments(ids.size() * m);
    for (size_t i = 0; i < ids.size(); ++i) {
        const auto uid = static_cast<size_t>(ids[i]);
        for (size_t gi = 0; gi < m; ++gi) {
            assignments[i * m + gi] =
                    main_codes_by_group_[gi * static_cast<size_t>(ntotal) + uid];
        }
    }
    out.resize(static_cast<Eigen::Index>(ids.size()), d);
    main_.decode_assignments(ids.size(), assignments.data(), out.data());
    for (size_t i = 0; i < ids.size(); ++i) {
        const auto uid = static_cast<size_t>(ids[i]);
        for (int stage = 0; stage < tail_stages; ++stage) {
            const uint16_t code =
                    tail_codes_by_stage_[static_cast<size_t>(stage)][uid];
            out.row(static_cast<Eigen::Index>(i)) +=
                    tail_codebooks_original_[static_cast<size_t>(stage)].row(
                            static_cast<Eigen::Index>(code));
        }
    }
}

size_t IndexAREPQ::sa_code_size() const {
    return (static_cast<size_t>(std::max(0, total_bits)) + 7U) / 8U;
}

void IndexAREPQ::sa_encode(
        faiss::idx_t n,
        const float* x,
        uint8_t* bytes) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "IndexAREPQ must be trained before encode");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "IndexAREPQ::sa_encode requires input");
    FAISS_THROW_IF_NOT_MSG(bytes != nullptr, "IndexAREPQ::sa_encode requires output");
    if (n <= 0) {
        return;
    }

    const Eigen::Map<const RowMatrixXf> x_map(x, n, d);
    const RowMatrixXf x_batch = x_map;
    std::vector<uint16_t> main_codes;
    std::vector<std::vector<uint16_t>> tail_codes;
    encode_batch_joint(x_batch, main_codes, tail_codes);

    const auto& structure = main_.structure();
    const size_t m = structure.group_count();
    const size_t code_size = sa_code_size();
    for (faiss::idx_t row = 0; row < n; ++row) {
        uint8_t* row_bytes = bytes + static_cast<size_t>(row) * code_size;
        std::memset(row_bytes, 0, code_size);
        int bit_offset = 0;
        const size_t row_offset = static_cast<size_t>(row) * m;
        for (size_t gi = 0; gi < m; ++gi) {
            write_bits(
                    row_bytes,
                    bit_offset,
                    main_codes[row_offset + gi],
                    structure.groups[gi].nbits);
        }
        for (int stage = 0; stage < tail_stages; ++stage) {
            write_bits(
                    row_bytes,
                    bit_offset,
                    tail_codes[static_cast<size_t>(stage)][static_cast<size_t>(row)],
                    tail_bits);
        }
    }
}

void IndexAREPQ::sa_decode(
        faiss::idx_t n,
        const uint8_t* bytes,
        float* x) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "IndexAREPQ must be trained before decode");
    FAISS_THROW_IF_NOT_MSG(bytes != nullptr, "IndexAREPQ::sa_decode requires input");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "IndexAREPQ::sa_decode requires output");
    if (n <= 0) {
        return;
    }

    const auto& structure = main_.structure();
    const size_t m = structure.group_count();
    const size_t code_size = sa_code_size();
    std::vector<uint16_t> main_codes(static_cast<size_t>(n) * m, uint16_t{0});
    std::vector<std::vector<uint16_t>> tail_codes(
            static_cast<size_t>(tail_stages),
            std::vector<uint16_t>(static_cast<size_t>(n), uint16_t{0}));
    for (faiss::idx_t row = 0; row < n; ++row) {
        const uint8_t* row_bytes = bytes + static_cast<size_t>(row) * code_size;
        int bit_offset = 0;
        const size_t row_offset = static_cast<size_t>(row) * m;
        for (size_t gi = 0; gi < m; ++gi) {
            main_codes[row_offset + gi] = static_cast<uint16_t>(
                    read_bits(row_bytes, bit_offset, structure.groups[gi].nbits));
        }
        for (int stage = 0; stage < tail_stages; ++stage) {
            tail_codes[static_cast<size_t>(stage)][static_cast<size_t>(row)] =
                    static_cast<uint16_t>(
                            read_bits(row_bytes, bit_offset, tail_bits));
        }
    }

    main_.decode_assignments(n, main_codes.data(), x);
    Eigen::Map<RowMatrixXf> out(x, n, d);
    for (faiss::idx_t row = 0; row < n; ++row) {
        for (int stage = 0; stage < tail_stages; ++stage) {
            const uint16_t code =
                    tail_codes[static_cast<size_t>(stage)][static_cast<size_t>(row)];
            out.row(row) += tail_codebooks_original_[static_cast<size_t>(stage)].row(
                    static_cast<Eigen::Index>(code));
        }
    }
}

size_t IndexAREPQ::adc_lut_size() const noexcept {
    return main_.adc_lut_size() +
            static_cast<size_t>(tail_stages) * static_cast<size_t>(tail_ksub);
}

void IndexAREPQ::transform_vector(const float* x, float* out) const {
    main_.transform_vector(x, out);
}

void IndexAREPQ::compute_adc_lut_from_transformed(
        const float* transformed_x,
        float* lut) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "IndexAREPQ must be trained before LUT build");
    FAISS_THROW_IF_NOT_MSG(
            transformed_x != nullptr,
            "IndexAREPQ::compute_adc_lut_from_transformed requires query");
    FAISS_THROW_IF_NOT_MSG(
            lut != nullptr,
            "IndexAREPQ::compute_adc_lut_from_transformed requires output");

    main_.compute_adc_lut_from_transformed(transformed_x, lut);
    size_t offset = main_.adc_lut_size();
    for (int stage = 0; stage < tail_stages; ++stage) {
        const auto& codebook = tail_codebooks_y_[static_cast<size_t>(stage)];
        const auto& norms = tail_norms_[static_cast<size_t>(stage)];
        for (int code = 0; code < tail_ksub; ++code) {
            float dot = 0.0f;
            for (int dim = 0; dim < d; ++dim) {
                dot += transformed_x[static_cast<size_t>(dim)] * codebook(code, dim);
            }
            lut[offset + static_cast<size_t>(code)] =
                    norms[static_cast<size_t>(code)] - 2.0f * dot;
        }
        offset += static_cast<size_t>(tail_ksub);
    }
}

float IndexAREPQ::adc_distance_from_packed_code(
        const uint8_t* code,
        const float* lut) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "IndexAREPQ must be trained before ADC");
    FAISS_THROW_IF_NOT_MSG(code != nullptr, "IndexAREPQ requires packed code");
    FAISS_THROW_IF_NOT_MSG(lut != nullptr, "IndexAREPQ requires LUT");

    static constexpr size_t kMaxMainGroupsOnStack = 1024;
    static constexpr size_t kMaxTailStagesOnStack = 64;
    const auto& structure = main_.structure();
    const size_t m = structure.group_count();
    FAISS_THROW_IF_NOT_MSG(
            m <= kMaxMainGroupsOnStack,
            "IndexAREPQ IVF ADC supports at most 1024 main groups");
    FAISS_THROW_IF_NOT_MSG(
            static_cast<size_t>(tail_stages) <= kMaxTailStagesOnStack,
            "IndexAREPQ IVF ADC supports at most 64 tail stages");

    std::array<uint16_t, kMaxMainGroupsOnStack> main_codes{};
    std::array<uint16_t, kMaxTailStagesOnStack> tail_codes{};

    int bit_offset = 0;
    float distance = 0.0f;
    size_t lut_offset = 0;
    const auto& codebooks = main_.codebooks();
    for (size_t gi = 0; gi < m; ++gi) {
        const uint16_t main_code = static_cast<uint16_t>(
                read_bits(code, bit_offset, structure.groups[gi].nbits));
        main_codes[gi] = main_code;
        distance += lut[lut_offset + static_cast<size_t>(main_code)];
        lut_offset += static_cast<size_t>(codebooks[gi].rows());
    }

    const size_t tail_lut_base = main_.adc_lut_size();
    for (int stage = 0; stage < tail_stages; ++stage) {
        const uint16_t tail_code =
                static_cast<uint16_t>(read_bits(code, bit_offset, tail_bits));
        tail_codes[static_cast<size_t>(stage)] = tail_code;
        distance += lut[tail_lut_base +
                static_cast<size_t>(stage) * static_cast<size_t>(tail_ksub) +
                static_cast<size_t>(tail_code)];
    }

    for (int stage = 0; stage < tail_stages; ++stage) {
        const uint16_t tail_code = tail_codes[static_cast<size_t>(stage)];
        for (size_t gi = 0; gi < m; ++gi) {
            const auto& cross = cross_tables_[static_cast<size_t>(stage)][gi];
            distance += cross[static_cast<size_t>(main_codes[gi]) *
                                      static_cast<size_t>(tail_ksub) +
                              static_cast<size_t>(tail_code)];
        }
    }

    for (int s0 = 0; s0 < tail_stages; ++s0) {
        for (int s1 = s0 + 1; s1 < tail_stages; ++s1) {
            const auto& table =
                    tail_pair_tables_[static_cast<size_t>(s0)][static_cast<size_t>(s1)];
            distance += table[static_cast<size_t>(tail_codes[static_cast<size_t>(s0)]) *
                                      static_cast<size_t>(tail_ksub) +
                              static_cast<size_t>(tail_codes[static_cast<size_t>(s1)])];
        }
    }
    return distance;
}

size_t IndexAREPQ::serialized_payload_bytes() const {
    const size_t all_code_bytes =
            (static_cast<size_t>(std::max(0, total_bits)) *
                     static_cast<size_t>(ntotal) +
             7) /
            8;
    return main_.serialized_payload_bytes() + all_code_bytes +
            tail_memory_stats().serialized_codebook_bytes;
}

TailMemoryStats IndexAREPQ::tail_memory_stats() const noexcept {
    TailMemoryStats stats;
    const size_t tail_bits_per_vec =
            static_cast<size_t>(std::max(0, tail_bits)) *
            static_cast<size_t>(std::max(0, tail_stages));
    stats.payload_code_bytes =
            (tail_bits_per_vec * static_cast<size_t>(ntotal) + 7) / 8;
    for (const auto& codes : tail_codes_by_stage_) {
        stats.resident_flat_code_bytes += codes.size() * sizeof(uint16_t);
    }

    for (const auto& codebook : tail_codebooks_y_) {
        stats.serialized_codebook_bytes +=
                static_cast<size_t>(codebook.size()) * sizeof(float);
    }
    for (const auto& codebook : tail_codebooks_original_) {
        stats.reconstruction_codebook_bytes +=
                static_cast<size_t>(codebook.size()) * sizeof(float);
    }
    stats.transform_copy_bytes =
            static_cast<size_t>(transform_matrix_.size()) * sizeof(float);
    for (const auto& norms : tail_norms_) {
        stats.norm_table_entries += norms.size();
    }
    stats.norm_table_bytes = stats.norm_table_entries * sizeof(float);

    for (const auto& stage_tables : cross_tables_) {
        for (const auto& table : stage_tables) {
            stats.product_tail_table_entries += table.size();
        }
    }
    stats.product_tail_table_bytes =
            stats.product_tail_table_entries * sizeof(float);

    for (const auto& row : tail_pair_tables_) {
        for (const auto& table : row) {
            stats.tail_pair_table_entries += table.size();
        }
    }
    stats.tail_pair_table_bytes =
            stats.tail_pair_table_entries * sizeof(float);

    stats.query_lut_entries_per_query =
            static_cast<size_t>(std::max(0, tail_stages)) *
            static_cast<size_t>(std::max(0, tail_ksub));
    stats.query_lut_bytes_per_query =
            stats.query_lut_entries_per_query * sizeof(float);
    return stats;
}

void IndexAREPQ::decode_tail_sum(
        const std::vector<std::vector<uint16_t>>& tail_codes,
        RowMatrixXf& out) const {
    const Eigen::Index rows =
            tail_codes.empty()
            ? 0
            : static_cast<Eigen::Index>(tail_codes.front().size());
    out.resize(rows, d);
    out.setZero();
    for (int stage = 0; stage < tail_stages; ++stage) {
        const auto& codebook = tail_codebooks_y_[static_cast<size_t>(stage)];
        const auto& codes = tail_codes[static_cast<size_t>(stage)];
        for (Eigen::Index row = 0; row < rows; ++row) {
            out.row(row) += codebook.row(codes[static_cast<size_t>(row)]);
        }
    }
}

void IndexAREPQ::decode_tail_sum_except(
        const std::vector<std::vector<uint16_t>>& tail_codes,
        int excluded_stage,
        RowMatrixXf& out) const {
    const Eigen::Index rows =
            tail_codes.empty()
            ? 0
            : static_cast<Eigen::Index>(tail_codes.front().size());
    out.resize(rows, d);
    out.setZero();
    for (int stage = 0; stage < tail_stages; ++stage) {
        if (stage == excluded_stage) {
            continue;
        }
        const auto& codebook = tail_codebooks_y_[static_cast<size_t>(stage)];
        const auto& codes = tail_codes[static_cast<size_t>(stage)];
        for (Eigen::Index row = 0; row < rows; ++row) {
            out.row(row) += codebook.row(codes[static_cast<size_t>(row)]);
        }
    }
}

void IndexAREPQ::encode_batch_joint(
        const RowMatrixXf& x_batch,
        std::vector<uint16_t>& main_codes,
        std::vector<std::vector<uint16_t>>& tail_codes) const {
    const RowMatrixXf y = transform_rows_with_epq(main_, x_batch);
    const auto initial_main_codes =
            main_.compute_assignments(x_batch.rows(), x_batch.data());
    encode_transformed_batch_joint(
            y,
            initial_main_codes,
            main_codes,
            tail_codes);
}

void IndexAREPQ::encode_transformed_batch_joint(
        const RowMatrixXf& y,
        const std::vector<uint16_t>& initial_main_codes,
        std::vector<uint16_t>& main_codes,
        std::vector<std::vector<uint16_t>>& tail_codes) const {
    const size_t m = main_.structure().group_count();
    main_codes = initial_main_codes;
    RowMatrixXf main_y;
    decode_main_assignments_transformed(
            main_,
            y.rows(),
            main_codes.data(),
            main_y);
    RowMatrixXf residual = y - main_y;
    tail_codes.assign(
            static_cast<size_t>(tail_stages),
            std::vector<uint16_t>(static_cast<size_t>(y.rows()), uint16_t{0}));
    for (int stage = 0; stage < tail_stages; ++stage) {
        tail_codes[static_cast<size_t>(stage)] = assign_residual_tail(
                residual,
                tail_codebooks_y_[static_cast<size_t>(stage)]);
        const auto& codebook = tail_codebooks_y_[static_cast<size_t>(stage)];
        const auto& codes = tail_codes[static_cast<size_t>(stage)];
        for (Eigen::Index row = 0; row < residual.rows(); ++row) {
            residual.row(row) -= codebook.row(codes[static_cast<size_t>(row)]);
        }
    }

    for (int iter = 0; iter < icm_iters; ++iter) {
        const std::vector<uint16_t> prev_main_codes =
                skip_stable_tail_reassign && tail_stages == 1
                ? main_codes
                : std::vector<uint16_t>();
        RowMatrixXf tail_sum;
        decode_tail_sum(tail_codes, tail_sum);
        assign_main_codes_for_transformed_targets(main_, y - tail_sum, main_codes);
        decode_main_assignments_transformed(
                main_,
                y.rows(),
                main_codes.data(),
                main_y);
        if (skip_stable_tail_reassign && tail_stages == 1) {
            std::vector<Eigen::Index> changed_rows;
            changed_rows.reserve(static_cast<size_t>(y.rows()));
            for (Eigen::Index row = 0; row < y.rows(); ++row) {
                const size_t offset = static_cast<size_t>(row) * m;
                bool changed = false;
                for (size_t gi = 0; gi < m; ++gi) {
                    if (prev_main_codes[offset + gi] != main_codes[offset + gi]) {
                        changed = true;
                        break;
                    }
                }
                if (changed) {
                    changed_rows.push_back(row);
                }
            }
            if (changed_rows.size() == static_cast<size_t>(y.rows())) {
                tail_codes.front() = assign_residual_tail(
                        y - main_y,
                        tail_codebooks_y_.front());
            } else if (!changed_rows.empty()) {
                RowMatrixXf changed_target(
                        static_cast<Eigen::Index>(changed_rows.size()),
                        d);
                for (Eigen::Index i = 0;
                     i < static_cast<Eigen::Index>(changed_rows.size());
                     ++i) {
                    const Eigen::Index row =
                            changed_rows[static_cast<size_t>(i)];
                    changed_target.row(i) = y.row(row) - main_y.row(row);
                }
                const auto changed_codes = assign_residual_tail(
                        changed_target,
                        tail_codebooks_y_.front());
                auto& codes = tail_codes.front();
                for (size_t i = 0; i < changed_rows.size(); ++i) {
                    codes[static_cast<size_t>(changed_rows[i])] =
                            changed_codes[i];
                }
            }
        } else {
            for (int stage = 0; stage < tail_stages; ++stage) {
                RowMatrixXf other_tail;
                decode_tail_sum_except(tail_codes, stage, other_tail);
                tail_codes[static_cast<size_t>(stage)] = assign_residual_tail(
                        y - main_y - other_tail,
                        tail_codebooks_y_[static_cast<size_t>(stage)]);
            }
        }
    }

    if (icm_iters > 0 && final_main_reassign) {
        RowMatrixXf tail_sum;
        decode_tail_sum(tail_codes, tail_sum);
        assign_main_codes_for_transformed_targets(main_, y - tail_sum, main_codes);
    }

    refine_single_tail_beam(y, main_codes, tail_codes);
}

void IndexAREPQ::refine_single_tail_beam(
        const RowMatrixXf& y,
        std::vector<uint16_t>& main_codes,
        std::vector<std::vector<uint16_t>>& tail_codes) const {
    if (tail_stages != 1 || tail_beam_candidates <= 1 || y.rows() <= 0) {
        return;
    }

    const int beam = std::clamp(tail_beam_candidates, 1, tail_ksub);
    const size_t m = main_.structure().group_count();
    const auto& tail_codebook = tail_codebooks_y_.front();
    auto& tail = tail_codes.front();

    RowMatrixXf main_y;
    decode_main_assignments_transformed(
            main_,
            y.rows(),
            main_codes.data(),
            main_y);
    const RowMatrixXf residual = y - main_y;
    const auto candidate_tail_codes =
            assign_residual_tail_topk(residual, tail_codebook, beam);

    RowMatrixXf candidate_targets(y.rows() * beam, d);
    for (Eigen::Index row = 0; row < y.rows(); ++row) {
        for (int b = 0; b < beam; ++b) {
            const uint16_t code =
                    candidate_tail_codes[static_cast<size_t>(row) *
                                                 static_cast<size_t>(beam) +
                                         static_cast<size_t>(b)];
            candidate_targets.row(row * beam + b) =
                    y.row(row) - tail_codebook.row(code);
        }
    }

    std::vector<uint16_t> candidate_main_codes;
    assign_main_codes_for_transformed_targets(
            main_,
            candidate_targets,
            candidate_main_codes);

    RowMatrixXf candidate_main_y;
    decode_main_assignments_transformed(
            main_,
            candidate_targets.rows(),
            candidate_main_codes.data(),
            candidate_main_y);

    for (Eigen::Index row = 0; row < y.rows(); ++row) {
        const uint16_t current_tail = tail[static_cast<size_t>(row)];
        double best_error = static_cast<double>(
                (y.row(row) - main_y.row(row) -
                 tail_codebook.row(current_tail))
                        .squaredNorm());
        int best_b = -1;
        for (int b = 0; b < beam; ++b) {
            const Eigen::Index candidate_row = row * beam + b;
            const uint16_t tail_code =
                    candidate_tail_codes[static_cast<size_t>(row) *
                                                 static_cast<size_t>(beam) +
                                         static_cast<size_t>(b)];
            const double error = static_cast<double>(
                    (y.row(row) - candidate_main_y.row(candidate_row) -
                     tail_codebook.row(tail_code))
                            .squaredNorm());
            if (error + 1e-7 < best_error) {
                best_error = error;
                best_b = b;
            }
        }
        if (best_b >= 0) {
            const size_t src =
                    (static_cast<size_t>(row) * static_cast<size_t>(beam) +
                     static_cast<size_t>(best_b)) *
                    m;
            const size_t dst = static_cast<size_t>(row) * m;
            std::copy(
                    candidate_main_codes.begin() +
                            static_cast<std::ptrdiff_t>(src),
                    candidate_main_codes.begin() +
                            static_cast<std::ptrdiff_t>(src + m),
                    main_codes.begin() + static_cast<std::ptrdiff_t>(dst));
            tail[static_cast<size_t>(row)] =
                    candidate_tail_codes[static_cast<size_t>(row) *
                                                 static_cast<size_t>(beam) +
                                         static_cast<size_t>(best_b)];
        }
    }
}

void IndexAREPQ::build_tail_auxiliary_tables() {
    tail_codebooks_original_.clear();
    tail_codebooks_original_.reserve(tail_codebooks_y_.size());
    tail_norms_.assign(tail_codebooks_y_.size(), {});
    for (size_t stage = 0; stage < tail_codebooks_y_.size(); ++stage) {
        const auto& codebook_y = tail_codebooks_y_[stage];
        tail_codebooks_original_.push_back(codebook_y * transform_matrix_.transpose());
        auto& norms = tail_norms_[stage];
        norms.assign(static_cast<size_t>(codebook_y.rows()), 0.0f);
        for (Eigen::Index code = 0; code < codebook_y.rows(); ++code) {
            norms[static_cast<size_t>(code)] = codebook_y.row(code).squaredNorm();
        }
    }

    const size_t m = main_.structure().group_count();
    const auto& groups = main_.active_groups();
    const auto& codebooks = main_.codebooks();
    cross_tables_.assign(
            static_cast<size_t>(tail_stages),
            std::vector<std::vector<float>>(m));
    for (int stage = 0; stage < tail_stages; ++stage) {
        const auto& tail_codebook = tail_codebooks_y_[static_cast<size_t>(stage)];
        for (size_t gi = 0; gi < m; ++gi) {
            const auto& dims = groups[gi];
            const auto& main_codebook = codebooks[gi];
            auto& table = cross_tables_[static_cast<size_t>(stage)][gi];
            table.resize(
                    static_cast<size_t>(main_codebook.rows()) *
                    static_cast<size_t>(tail_ksub));
            for (Eigen::Index mc = 0; mc < main_codebook.rows(); ++mc) {
                for (int tc = 0; tc < tail_ksub; ++tc) {
                    float dot = 0.0f;
                    for (size_t j = 0; j < dims.size(); ++j) {
                        dot += main_codebook(mc, static_cast<Eigen::Index>(j)) *
                                tail_codebook(tc, dims[j]);
                    }
                    table[static_cast<size_t>(mc) *
                                  static_cast<size_t>(tail_ksub) +
                          static_cast<size_t>(tc)] = 2.0f * dot;
                }
            }
        }
    }

    tail_pair_tables_.assign(
            static_cast<size_t>(tail_stages),
            std::vector<std::vector<float>>(static_cast<size_t>(tail_stages)));
    for (int s0 = 0; s0 < tail_stages; ++s0) {
        for (int s1 = s0 + 1; s1 < tail_stages; ++s1) {
            auto& table =
                    tail_pair_tables_[static_cast<size_t>(s0)][static_cast<size_t>(s1)];
            table.resize(
                    static_cast<size_t>(tail_ksub) *
                    static_cast<size_t>(tail_ksub));
            const auto& c0 = tail_codebooks_y_[static_cast<size_t>(s0)];
            const auto& c1 = tail_codebooks_y_[static_cast<size_t>(s1)];
            for (int a = 0; a < tail_ksub; ++a) {
                for (int b = 0; b < tail_ksub; ++b) {
                    table[static_cast<size_t>(a) *
                                  static_cast<size_t>(tail_ksub) +
                          static_cast<size_t>(b)] =
                            2.0f * c0.row(a).dot(c1.row(b));
                }
            }
        }
    }
}

void IndexAREPQ::update_tail_codebooks_from_assignments(
        const RowMatrixXf& y,
        const std::vector<uint16_t>& main_codes,
        const std::vector<std::vector<uint16_t>>& tail_codes,
        float update_weight) {
    update_weight = std::clamp(update_weight, 0.0f, 1.0f);
    RowMatrixXf main_y;
    decode_main_assignments_transformed(
            main_,
            y.rows(),
            main_codes.data(),
            main_y);

    for (int stage = 0; stage < tail_stages; ++stage) {
        RowMatrixXf other_tail;
        decode_tail_sum_except(tail_codes, stage, other_tail);
        const RowMatrixXf target = y - main_y - other_tail;

        RowMatrixXf updated = RowMatrixXf::Zero(tail_ksub, d);
        std::vector<int> counts(static_cast<size_t>(tail_ksub), 0);
        const auto& codes = tail_codes[static_cast<size_t>(stage)];
        for (Eigen::Index row = 0; row < target.rows(); ++row) {
            const uint16_t code = codes[static_cast<size_t>(row)];
            updated.row(static_cast<Eigen::Index>(code)) += target.row(row);
            counts[static_cast<size_t>(code)] += 1;
        }

        const auto& old_codebook = tail_codebooks_y_[static_cast<size_t>(stage)];
        for (int code = 0; code < tail_ksub; ++code) {
            const int count = counts[static_cast<size_t>(code)];
            if (count > 0) {
                updated.row(code) /= static_cast<float>(count);
                if (update_weight < 1.0f) {
                    updated.row(code) =
                            (1.0f - update_weight) * old_codebook.row(code) +
                            update_weight * updated.row(code);
                }
            } else {
                updated.row(code) = old_codebook.row(code);
            }
        }
        tail_codebooks_y_[static_cast<size_t>(stage)] = std::move(updated);
    }
}

double IndexAREPQ::additive_mse(
        const RowMatrixXf& y,
        const std::vector<uint16_t>& main_codes,
        const std::vector<std::vector<uint16_t>>& tail_codes) const {
    RowMatrixXf main_y;
    decode_main_assignments_transformed(
            main_,
            y.rows(),
            main_codes.data(),
            main_y);
    RowMatrixXf tail_y;
    decode_tail_sum(tail_codes, tail_y);
    double sse = 0.0;
    for (Eigen::Index row = 0; row < y.rows(); ++row) {
        sse += static_cast<double>(
                (y.row(row) - main_y.row(row) - tail_y.row(row)).squaredNorm());
    }
    return sse / static_cast<double>(std::max<Eigen::Index>(1, y.rows()));
}

void IndexAREPQ::run_tail_alternating_optimization(
        const RowMatrixXf& train_y,
        const std::vector<uint16_t>& initial_main_codes) {
    if (tail_alt_iters <= 0) {
        return;
    }

    std::vector<RowMatrixXf> best_codebooks = tail_codebooks_y_;
    double best_mse = std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < tail_alt_iters; ++iter) {
        std::vector<uint16_t> main_codes;
        std::vector<std::vector<uint16_t>> tail_codes;
        encode_transformed_batch_joint(
                train_y,
                initial_main_codes,
                main_codes,
                tail_codes);
        const double mse = additive_mse(train_y, main_codes, tail_codes);
        if (iter == 0) {
            tail_alt_initial_mse_ = mse;
        }
        if (mse < best_mse) {
            best_mse = mse;
            best_codebooks = tail_codebooks_y_;
        }

        update_tail_codebooks_from_assignments(
                train_y,
                main_codes,
                tail_codes,
                tail_alt_update_weight);
        build_tail_auxiliary_tables();
    }

    std::vector<uint16_t> final_main_codes;
    std::vector<std::vector<uint16_t>> final_tail_codes;
    encode_transformed_batch_joint(
            train_y,
            initial_main_codes,
            final_main_codes,
            final_tail_codes);
    tail_alt_final_mse_ =
            additive_mse(train_y, final_main_codes, final_tail_codes);
    if (tail_alt_final_mse_ < best_mse) {
        best_mse = tail_alt_final_mse_;
        best_codebooks = tail_codebooks_y_;
    }
    tail_alt_best_mse_ = best_mse;

    if (!best_codebooks.empty() && best_mse < tail_alt_final_mse_) {
        tail_codebooks_y_ = std::move(best_codebooks);
        build_tail_auxiliary_tables();
        tail_alt_final_mse_ = tail_alt_best_mse_;
    }
}

void IndexAREPQ::add_tail_terms_to_chunk(
        faiss::idx_t qb,
        faiss::idx_t b0,
        faiss::idx_t csz,
        const std::vector<std::vector<float>>& tail_luts,
        std::vector<float>& dist_chunk) const {
    const size_t m = main_.structure().group_count();
    for (int stage = 0; stage < tail_stages; ++stage) {
        const uint16_t* tail_codes =
                tail_codes_by_stage_[static_cast<size_t>(stage)].data() +
                static_cast<size_t>(b0);
        for (faiss::idx_t qi = 0; qi < qb; ++qi) {
            float* row =
                    dist_chunk.data() +
                    static_cast<size_t>(qi) * static_cast<size_t>(csz);
            const float* lut =
                    tail_luts[static_cast<size_t>(stage)].data() +
                    static_cast<size_t>(qi) * static_cast<size_t>(tail_ksub);
            accumulate_u16_lut_row<true>(row, tail_codes, lut, csz);
        }

        for (size_t gi = 0; gi < m; ++gi) {
            const uint16_t* main_codes =
                    main_codes_by_group_.data() +
                    gi * static_cast<size_t>(ntotal) +
                    static_cast<size_t>(b0);
            const auto& cross = cross_tables_[static_cast<size_t>(stage)][gi];
            for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                float* row =
                        dist_chunk.data() +
                        static_cast<size_t>(qi) * static_cast<size_t>(csz);
                for (faiss::idx_t j = 0; j < csz; ++j) {
                    const size_t jj = static_cast<size_t>(j);
                    row[jj] += cross[static_cast<size_t>(main_codes[jj]) *
                                             static_cast<size_t>(tail_ksub) +
                                     static_cast<size_t>(tail_codes[jj])];
                }
            }
        }
    }

    for (int s0 = 0; s0 < tail_stages; ++s0) {
        const uint16_t* codes0 =
                tail_codes_by_stage_[static_cast<size_t>(s0)].data() +
                static_cast<size_t>(b0);
        for (int s1 = s0 + 1; s1 < tail_stages; ++s1) {
            const uint16_t* codes1 =
                    tail_codes_by_stage_[static_cast<size_t>(s1)].data() +
                    static_cast<size_t>(b0);
            const auto& table =
                    tail_pair_tables_[static_cast<size_t>(s0)][static_cast<size_t>(s1)];
            for (faiss::idx_t qi = 0; qi < qb; ++qi) {
                float* row =
                        dist_chunk.data() +
                        static_cast<size_t>(qi) * static_cast<size_t>(csz);
                for (faiss::idx_t j = 0; j < csz; ++j) {
                    const size_t jj = static_cast<size_t>(j);
                    row[jj] += table[static_cast<size_t>(codes0[jj]) *
                                             static_cast<size_t>(tail_ksub) +
                                     static_cast<size_t>(codes1[jj])];
                }
            }
        }
    }
}

}  // namespace epq
