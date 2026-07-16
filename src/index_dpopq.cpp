#include "epq/index_dpopq.h"

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissException.h>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>

namespace epq {
namespace {

int getenv_int_or(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0') {
        return fallback;
    }
    return static_cast<int>(value);
}

bool test_bit(const std::vector<uint64_t>& bits, int pos) {
    return (bits[static_cast<size_t>(pos / 64)] >> (pos % 64)) & uint64_t{1};
}

void or_shift_left(
        std::vector<uint64_t>& dst,
        const std::vector<uint64_t>& src,
        int shift,
        int max_bit,
        std::vector<uint64_t>& new_bits) {
    std::fill(new_bits.begin(), new_bits.end(), uint64_t{0});
    const int word_shift = shift / 64;
    const int bit_shift = shift % 64;
    for (int i = 0; i < static_cast<int>(src.size()); ++i) {
        const uint64_t word = src[static_cast<size_t>(i)];
        if (word == 0) {
            continue;
        }
        const int out = i + word_shift;
        if (out >= static_cast<int>(dst.size())) {
            continue;
        }
        new_bits[static_cast<size_t>(out)] |= word << bit_shift;
        if (bit_shift != 0 && out + 1 < static_cast<int>(dst.size())) {
            new_bits[static_cast<size_t>(out + 1)] |= word >> (64 - bit_shift);
        }
    }
    const int valid_bits_last_word = (max_bit + 1) % 64;
    if (valid_bits_last_word != 0 && !new_bits.empty()) {
        const uint64_t mask = (uint64_t{1} << valid_bits_last_word) - uint64_t{1};
        new_bits.back() &= mask;
    }
    for (size_t i = 0; i < dst.size(); ++i) {
        new_bits[i] &= ~dst[i];
        dst[i] |= new_bits[i];
    }
}

}  // namespace

IndexDPOPQ::IndexDPOPQ(int d, int total_bits)
        : faiss::Index(d, faiss::METRIC_L2),
          total_bits(total_bits),
          M_(total_bits / 8) {
    validate_config();
}

void IndexDPOPQ::validate_config() const {
    FAISS_THROW_IF_NOT_MSG(total_bits % 8 == 0, "DPOPQ requires bits divisible by 8");
    FAISS_THROW_IF_NOT_MSG(M_ > 0, "invalid DPOPQ component count");
    FAISS_THROW_IF_NOT_MSG(d > 0, "invalid DPOPQ dimension");
    FAISS_THROW_IF_NOT_MSG(d % M_ == 0, "DPOPQ requires d divisible by M");
}

RowMatrixXf IndexDPOPQ::as_matrix(faiss::idx_t n, const float* x) const {
    Eigen::Map<const RowMatrixXf> mapped(x, n, d);
    return RowMatrixXf(mapped);
}

void IndexDPOPQ::train(faiss::idx_t n, const float* x) {
    validate_config();
    FAISS_THROW_IF_NOT_MSG(n > 0 && x != nullptr, "DPOPQ train requires data");
    const auto t0 = std::chrono::steady_clock::now();
    stats_ = {};
    RowMatrixXf xt = as_matrix(n, x);

    const auto prep0 = std::chrono::steady_clock::now();
    train_pca_rotation(xt);
    solve_dp_partition();
    RowMatrixXf x_rot = apply_transform(xt);
    const auto prep1 = std::chrono::steady_clock::now();
    stats_.preparation_time = std::chrono::duration<double>(prep1 - prep0).count();
    stats_.structure_time = stats_.preparation_time;

    const auto cb0 = std::chrono::steady_clock::now();
    codebooks_.clear();
    codebooks_.reserve(static_cast<size_t>(M_));
    for (int g = 0; g < M_; ++g) {
        const int begin = group_offsets_[static_cast<size_t>(g)];
        const int end = group_offsets_[static_cast<size_t>(g + 1)];
        RowMatrixXf sub = x_rot.middleCols(begin, end - begin).eval();
        codebooks_.push_back(train_kmeans(sub, 256, kmeans_niter, kmeans_nredo));
    }
    const auto cb1 = std::chrono::steady_clock::now();
    stats_.codebook_time = std::chrono::duration<double>(cb1 - cb0).count();
    stats_.total_time = std::chrono::duration<double>(cb1 - t0).count();
    is_trained = true;
}

void IndexDPOPQ::add(faiss::idx_t, const float*) {
    FAISS_THROW_MSG("IndexDPOPQ standalone add is not implemented; use as IVF codec");
}

void IndexDPOPQ::search(
        faiss::idx_t,
        const float*,
        faiss::idx_t,
        float*,
        faiss::idx_t*,
        const faiss::SearchParameters*) const {
    FAISS_THROW_MSG("IndexDPOPQ standalone search is not implemented; use as IVF codec");
}

void IndexDPOPQ::reset() {
    ntotal = 0;
}

void IndexDPOPQ::reconstruct(faiss::idx_t, float*) const {
    FAISS_THROW_MSG("IndexDPOPQ reconstruct by id is not implemented");
}

size_t IndexDPOPQ::sa_code_size() const {
    return static_cast<size_t>(M_);
}

void IndexDPOPQ::sa_encode(faiss::idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "DPOPQ codec is not trained");
    RowMatrixXf x_rot = apply_transform(as_matrix(n, x));
#pragma omp parallel for schedule(static)
    for (faiss::idx_t i = 0; i < n; ++i) {
        uint8_t* out = bytes + static_cast<size_t>(i) * static_cast<size_t>(M_);
        for (int g = 0; g < M_; ++g) {
            const int begin = group_offsets_[static_cast<size_t>(g)];
            const int end = group_offsets_[static_cast<size_t>(g + 1)];
            const float* xptr = x_rot.row(i).data() + begin;
            const RowMatrixXf& cb = codebooks_[static_cast<size_t>(g)];
            int best = 0;
            float best_dist = std::numeric_limits<float>::infinity();
            for (int c = 0; c < 256; ++c) {
                const float dist = l2_distance(xptr, cb.row(c).data(), end - begin);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = c;
                }
            }
            out[g] = static_cast<uint8_t>(best);
        }
    }
}

void IndexDPOPQ::sa_decode(faiss::idx_t n, const uint8_t* bytes, float* x) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "DPOPQ codec is not trained");
    RowMatrixXf y(n, d);
    y.setZero();
    for (faiss::idx_t i = 0; i < n; ++i) {
        const uint8_t* code = bytes + static_cast<size_t>(i) * static_cast<size_t>(M_);
        for (int g = 0; g < M_; ++g) {
            const int begin = group_offsets_[static_cast<size_t>(g)];
            const int end = group_offsets_[static_cast<size_t>(g + 1)];
            y.block(i, begin, 1, end - begin) =
                    codebooks_[static_cast<size_t>(g)].row(code[g]);
        }
    }
    Eigen::Map<RowMatrixXf> out(x, n, d);
    out.noalias() = y * inverse_transform_;
    out.rowwise() += mean_;
}

int IndexDPOPQ::component_count() const noexcept {
    return M_;
}

const DPOPQTrainingStats& IndexDPOPQ::training_stats() const noexcept {
    return stats_;
}

size_t IndexDPOPQ::serialized_payload_bytes() const {
    size_t bytes = static_cast<size_t>(d) * sizeof(float);
    bytes += static_cast<size_t>(d) * static_cast<size_t>(d) * sizeof(float) * 2;
    bytes += transform_scales_.size() * sizeof(float);
    bytes += pc_order_.size() * sizeof(int);
    bytes += eigenvalues_.size() * sizeof(float);
    bytes += partition_values_.size() * sizeof(double);
    bytes += group_offsets_.size() * sizeof(int);
    for (const auto& cb : codebooks_) {
        bytes += static_cast<size_t>(cb.rows()) * static_cast<size_t>(cb.cols()) *
                sizeof(float);
    }
    return bytes;
}

size_t IndexDPOPQ::adc_lut_size() const noexcept {
    return static_cast<size_t>(M_) * 256;
}

void IndexDPOPQ::transform_vector(const float* x, float* out) const {
    Eigen::Map<const Eigen::RowVectorXf> row(x, d);
    Eigen::Map<Eigen::RowVectorXf> dst(out, d);
    dst.noalias() = (row - mean_) * rotation_;
}

void IndexDPOPQ::compute_adc_lut_from_transformed(
        const float* query_transformed,
        float* lut) const {
    for (int g = 0; g < M_; ++g) {
        const int begin = group_offsets_[static_cast<size_t>(g)];
        const int end = group_offsets_[static_cast<size_t>(g + 1)];
        const RowMatrixXf& cb = codebooks_[static_cast<size_t>(g)];
        float* dst = lut + static_cast<size_t>(g) * 256;
        for (int c = 0; c < 256; ++c) {
            dst[c] = l2_distance(
                    query_transformed + begin,
                    cb.row(c).data(),
                    end - begin);
        }
    }
}

float IndexDPOPQ::adc_distance_from_packed_code(
        const uint8_t* code,
        const float* lut) const {
    float dist = 0.0f;
    for (int g = 0; g < M_; ++g) {
        dist += lut[static_cast<size_t>(g) * 256 + code[g]];
    }
    return dist;
}

nlohmann::json IndexDPOPQ::metadata() const {
    nlohmann::json group_dims = nlohmann::json::array();
    nlohmann::json group_log_lambda_sums = nlohmann::json::array();
    for (int g = 0; g < M_; ++g) {
        const int begin = group_offsets_[static_cast<size_t>(g)];
        const int end = group_offsets_[static_cast<size_t>(g + 1)];
        group_dims.push_back(end - begin);
        double sum = 0.0;
        for (int i = begin; i < end; ++i) {
            sum += partition_values_[static_cast<size_t>(i)];
        }
        group_log_lambda_sums.push_back(sum);
    }
    return {
            {"family", "dpopq"},
            {"impl", "paper_based_implementation"},
            {"native_index", block_alignment
                     ? "PCA+DPLogEigenvalueAllocation+BlockAlignment+per-group-kmeans"
                     : "PCA+DPLogEigenvalueAllocation+per-group-kmeans"},
            {"d", d},
            {"M", M_},
            {"nbits", 8},
            {"total_bits", total_bits},
            {"block_alignment", block_alignment},
            {"partition_cost", partition_cost_},
            {"partition_units_exact", partition_units_exact_},
            {"partition_units_scale", partition_units_scale_},
            {"partition_units_sum", partition_units_sum_},
            {"group_dims", std::move(group_dims)},
            {"group_log_lambda_sums", std::move(group_log_lambda_sums)},
    };
}

void IndexDPOPQ::train_pca_rotation(const RowMatrixXf& xt) {
    mean_ = xt.colwise().mean();
    RowMatrixXf centered = xt;
    centered.rowwise() -= mean_;
    Eigen::MatrixXf cov =
            (centered.transpose() * centered) /
            std::max<float>(1.0f, static_cast<float>(xt.rows() - 1));
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(cov);
    FAISS_THROW_IF_NOT_MSG(
            solver.info() == Eigen::Success,
            "DPOPQ PCA eigendecomposition failed");
    pca_rotation_.resize(d, d);
    pca_eigenvalues_.assign(static_cast<size_t>(d), 0.0f);
    for (int out = 0; out < d; ++out) {
        const int src = d - 1 - out;
        pca_rotation_.col(out) = solver.eigenvectors().col(src);
        pca_eigenvalues_[static_cast<size_t>(out)] =
                std::max(0.0f, solver.eigenvalues()[src]);
    }
    prepare_partition_weights();
}

void IndexDPOPQ::prepare_partition_weights() {
    pca_partition_values_.assign(static_cast<size_t>(d), 0.0);
    partition_units_.assign(static_cast<size_t>(d), 0);
    float min_positive = std::numeric_limits<float>::infinity();
    float max_value = 0.0f;
    for (float value : pca_eigenvalues_) {
        if (value > 0.0f) {
            min_positive = std::min(min_positive, value);
            max_value = std::max(max_value, value);
        }
    }
    if (!std::isfinite(min_positive)) {
        partition_units_exact_ = true;
        partition_units_scale_ = 1.0;
        partition_units_sum_ = 0;
        return;
    }
    const float floor_value = std::max(
            min_positive,
            std::max(max_value, 1.0f) *
                    static_cast<float>(std::numeric_limits<double>::epsilon()));
    double min_log = std::numeric_limits<double>::infinity();
    std::vector<double> logs(static_cast<size_t>(d), 0.0);
    for (int i = 0; i < d; ++i) {
        const double log_value = std::log(std::max(
                static_cast<double>(pca_eigenvalues_[static_cast<size_t>(i)]),
                static_cast<double>(floor_value)));
        logs[static_cast<size_t>(i)] = log_value;
        min_log = std::min(min_log, log_value);
    }
    std::vector<int64_t> raw_units(static_cast<size_t>(d), 0);
    int64_t raw_sum = 0;
    for (int i = 0; i < d; ++i) {
        const double shifted = std::max(0.0, logs[static_cast<size_t>(i)] - min_log);
        pca_partition_values_[static_cast<size_t>(i)] = shifted;
        const int64_t units = static_cast<int64_t>(std::llround(shifted * 1000.0));
        raw_units[static_cast<size_t>(i)] = units;
        raw_sum += units;
    }
    const int default_max_units = d <= 200 ? 500000 : 20000;
    const int max_units =
            std::max(1024, dp_max_units > 0 ? dp_max_units
                                            : getenv_int_or("EPQ_DPOPQ_DP_MAX_UNITS", default_max_units));
    partition_units_exact_ = raw_sum <= max_units;
    partition_units_scale_ = 1.0;
    if (!partition_units_exact_ && raw_sum > 0) {
        partition_units_scale_ =
                static_cast<double>(max_units) / static_cast<double>(raw_sum);
    }
    partition_units_sum_ = 0;
    for (int i = 0; i < d; ++i) {
        int units = 0;
        if (partition_units_exact_) {
            units = static_cast<int>(raw_units[static_cast<size_t>(i)]);
        } else {
            units = static_cast<int>(std::llround(
                    static_cast<double>(raw_units[static_cast<size_t>(i)]) *
                    partition_units_scale_));
            if (raw_units[static_cast<size_t>(i)] > 0 && units == 0) {
                units = 1;
            }
        }
        partition_units_[static_cast<size_t>(i)] = units;
        partition_units_sum_ += units;
    }
}

std::vector<int> IndexDPOPQ::choose_balanced_subset_dp(
        const std::vector<int>& items,
        int take,
        int target_units) const {
    if (take <= 0) {
        return {};
    }
    if (take >= static_cast<int>(items.size())) {
        return items;
    }
    int sum_units = 0;
    for (int item : items) {
        sum_units += partition_units_[static_cast<size_t>(item)];
    }
    if (sum_units <= 0) {
        return std::vector<int>(items.begin(), items.begin() + take);
    }
    target_units = std::clamp(target_units, 0, sum_units);
    const int words = (sum_units + 64) / 64;
    std::vector<std::vector<uint64_t>> reachable(
            static_cast<size_t>(take + 1),
            std::vector<uint64_t>(static_cast<size_t>(words), uint64_t{0}));
    reachable[0][0] = uint64_t{1};
    const size_t state_count =
            static_cast<size_t>(take + 1) * static_cast<size_t>(sum_units + 1);
    std::vector<int> parent_item(state_count, -1);
    std::vector<int> parent_prev_sum(state_count, -1);
    auto state_index = [sum_units](int count, int sum) {
        return static_cast<size_t>(count) * static_cast<size_t>(sum_units + 1) +
                static_cast<size_t>(sum);
    };
    std::vector<uint64_t> newly(static_cast<size_t>(words), uint64_t{0});
    for (int pos = 0; pos < static_cast<int>(items.size()); ++pos) {
        const int w = partition_units_[static_cast<size_t>(items[pos])];
        const int max_count = std::min(take, pos + 1);
        for (int count = max_count; count >= 1; --count) {
            or_shift_left(
                    reachable[static_cast<size_t>(count)],
                    reachable[static_cast<size_t>(count - 1)],
                    w,
                    sum_units,
                    newly);
            for (int word_i = 0; word_i < words; ++word_i) {
                uint64_t word = newly[static_cast<size_t>(word_i)];
                while (word != 0) {
                    const int bit = __builtin_ctzll(word);
                    const int sum = word_i * 64 + bit;
                    if (sum <= sum_units) {
                        parent_item[state_index(count, sum)] = pos;
                        parent_prev_sum[state_index(count, sum)] = sum - w;
                    }
                    word &= word - 1;
                }
            }
        }
    }
    int best_sum = -1;
    int best_delta = std::numeric_limits<int>::max();
    for (int sum = 0; sum <= sum_units; ++sum) {
        if (!test_bit(reachable[static_cast<size_t>(take)], sum)) {
            continue;
        }
        const int delta = std::abs(sum - target_units);
        if (delta < best_delta) {
            best_delta = delta;
            best_sum = sum;
        }
    }
    FAISS_THROW_IF_NOT_MSG(best_sum >= 0, "DPOPQ subset DP failed");
    std::vector<int> selected_positions;
    selected_positions.reserve(static_cast<size_t>(take));
    int count = take;
    int sum = best_sum;
    while (count > 0) {
        const int pos = parent_item[state_index(count, sum)];
        const int prev = parent_prev_sum[state_index(count, sum)];
        FAISS_THROW_IF_NOT_MSG(pos >= 0 && prev >= 0, "DPOPQ subset DP invalid backpointer");
        selected_positions.push_back(pos);
        sum = prev;
        --count;
    }
    std::vector<char> chosen(items.size(), 0);
    for (int pos : selected_positions) {
        chosen[static_cast<size_t>(pos)] = 1;
    }
    std::vector<int> subset;
    subset.reserve(static_cast<size_t>(take));
    for (size_t i = 0; i < items.size(); ++i) {
        if (chosen[i]) {
            subset.push_back(items[i]);
        }
    }
    return subset;
}

std::vector<std::vector<int>> IndexDPOPQ::partition_recursive(
        const std::vector<int>& items,
        int groups) const {
    if (groups == 1) {
        std::vector<int> group = items;
        std::sort(group.begin(), group.end());
        return {std::move(group)};
    }
    const int left_groups = groups / 2;
    const int right_groups = groups - left_groups;
    const int left_take = static_cast<int>(
            (static_cast<int64_t>(items.size()) * left_groups) / groups);
    int total_units = 0;
    for (int item : items) {
        total_units += partition_units_[static_cast<size_t>(item)];
    }
    const int left_target = static_cast<int>(std::llround(
            static_cast<double>(total_units) * static_cast<double>(left_groups) /
            static_cast<double>(groups)));
    const std::vector<int> left =
            choose_balanced_subset_dp(items, left_take, left_target);
    std::vector<char> in_left(static_cast<size_t>(d), 0);
    for (int item : left) {
        in_left[static_cast<size_t>(item)] = 1;
    }
    std::vector<int> right;
    right.reserve(items.size() - left.size());
    for (int item : items) {
        if (!in_left[static_cast<size_t>(item)]) {
            right.push_back(item);
        }
    }
    auto out = partition_recursive(left, left_groups);
    auto right_out = partition_recursive(right, right_groups);
    out.insert(
            out.end(),
            std::make_move_iterator(right_out.begin()),
            std::make_move_iterator(right_out.end()));
    return out;
}

void IndexDPOPQ::solve_dp_partition() {
    std::vector<int> items(static_cast<size_t>(d));
    std::iota(items.begin(), items.end(), 0);
    const auto groups = partition_recursive(items, M_);
    group_offsets_.assign(static_cast<size_t>(M_ + 1), 0);
    pc_order_.clear();
    eigenvalues_.clear();
    partition_values_.clear();
    partition_cost_ = 0.0;
    double total = 0.0;
    for (double value : pca_partition_values_) {
        total += value;
    }
    const double target = total / static_cast<double>(M_);
    for (int g = 0; g < M_; ++g) {
        group_offsets_[static_cast<size_t>(g)] = static_cast<int>(pc_order_.size());
        double sum = 0.0;
        for (int pc : groups[static_cast<size_t>(g)]) {
            pc_order_.push_back(pc);
            eigenvalues_.push_back(pca_eigenvalues_[static_cast<size_t>(pc)]);
            const double partition_value = pca_partition_values_[static_cast<size_t>(pc)];
            partition_values_.push_back(partition_value);
            sum += partition_value;
        }
        const double diff = sum - target;
        partition_cost_ += diff * diff;
    }
    group_offsets_[static_cast<size_t>(M_)] = static_cast<int>(pc_order_.size());
    base_rotation_.resize(d, d);
    for (int out = 0; out < d; ++out) {
        base_rotation_.col(out) = pca_rotation_.col(pc_order_[static_cast<size_t>(out)]);
    }
    configure_block_alignment();
}

void IndexDPOPQ::configure_block_alignment() {
    transform_scales_.assign(static_cast<size_t>(d), 1.0f);
    if (block_alignment && M_ > 1) {
        const int group_dim = d / M_;
        for (int rank = 0; rank < group_dim; ++rank) {
            float min_eigenvalue = std::numeric_limits<float>::infinity();
            for (int g = 0; g < M_; ++g) {
                const int offset = g * group_dim + rank;
                const int pc = pc_order_[static_cast<size_t>(offset)];
                const float value = pca_eigenvalues_[static_cast<size_t>(pc)];
                if (value > 0.0f) {
                    min_eigenvalue = std::min(min_eigenvalue, value);
                }
            }
            if (!std::isfinite(min_eigenvalue) || min_eigenvalue <= 0.0f) {
                continue;
            }
            for (int g = 0; g < M_; ++g) {
                const int offset = g * group_dim + rank;
                const int pc = pc_order_[static_cast<size_t>(offset)];
                const float value = pca_eigenvalues_[static_cast<size_t>(pc)];
                if (value > min_eigenvalue) {
                    transform_scales_[static_cast<size_t>(offset)] =
                            std::sqrt(min_eigenvalue / value);
                }
            }
        }
    }
    rotation_ = base_rotation_;
    Eigen::VectorXf inverse_scales(d);
    for (int i = 0; i < d; ++i) {
        const float scale = std::max(
                transform_scales_[static_cast<size_t>(i)],
                std::numeric_limits<float>::min());
        rotation_.col(i) *= scale;
        inverse_scales[i] = 1.0f / scale;
    }
    inverse_transform_ = inverse_scales.asDiagonal() * base_rotation_.transpose();
}

RowMatrixXf IndexDPOPQ::apply_transform(const RowMatrixXf& x) const {
    RowMatrixXf centered = x;
    centered.rowwise() -= mean_;
    return centered * rotation_;
}

RowMatrixXf IndexDPOPQ::train_kmeans(
        const RowMatrixXf& x,
        int k,
        int niter,
        int nredo) {
    FAISS_THROW_IF_NOT_MSG(x.rows() > 0 && x.cols() > 0, "DPOPQ kmeans got empty matrix");
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

float IndexDPOPQ::l2_distance(const float* a, const float* b, int dim) {
    float acc = 0.0f;
    for (int j = 0; j < dim; ++j) {
        const float diff = a[j] - b[j];
        acc += diff * diff;
    }
    return acc;
}

}  // namespace epq
