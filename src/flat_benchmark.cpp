#include "epq/index_epq.h"
#include "epq/index_arepq.h"
#include "epq/index_bapq.h"
#include "epq/index_avq.h"
#include "epq/index_vaq.h"
#include "epq/benchmark_metadata.h"
#include "epq/serialization_size.h"
#include "epq/structure.h"
#include "epq/structure_builder.h"
#include "epq/training_config.h"
#include "structure_builder_internal.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/VectorTransform.h>
#include <faiss/Clustering.h>
#include <faiss/utils/distances.h>
#include <omp.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using RowMatrixXf =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
namespace sbi = epq::structure_builder_internal;

extern "C" void openblas_set_num_threads(int threads) __attribute__((weak));
extern "C" void goto_set_num_threads(int threads) __attribute__((weak));

void set_thread_env_var(const char* name, int value) {
    const std::string text = std::to_string(value);
    setenv(name, text.c_str(), 1);
}

int getenv_int_or(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return fallback;
    }
    try {
        return std::stoi(raw);
    } catch (const std::exception&) {
        return fallback;
    }
}

float getenv_float_or(const char* name, float fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return fallback;
    }
    try {
        return std::stof(raw);
    } catch (const std::exception&) {
        return fallback;
    }
}

bool env_flag_enabled(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return false;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value != "0" && value != "false" && value != "no" &&
            value != "off";
}

bool effective_group_stats_env_enabled() {
    return env_flag_enabled("EPQ_PRINT_EFFECTIVE_GROUP_STATS");
}

int get_config_int_or_env(
        const std::optional<nlohmann::json>& config,
        const char* section,
        const char* key,
        const char* env_name,
        int fallback) {
    if (config.has_value() && config->contains(section) &&
        config->at(section).is_object()) {
        const auto& payload = config->at(section);
        if (payload.contains(key) && !payload.at(key).is_null()) {
            return payload.at(key).get<int>();
        }
    }
    return getenv_int_or(env_name, fallback);
}

void cap_downstream_thread_pools(int threads) {
    if (threads <= 0) {
        return;
    }
    omp_set_num_threads(threads);
    set_thread_env_var("OMP_NUM_THREADS", threads);
    set_thread_env_var("OPENBLAS_NUM_THREADS", 1);
    set_thread_env_var("GOTO_NUM_THREADS", 1);
    set_thread_env_var("MKL_NUM_THREADS", 1);
    set_thread_env_var("BLIS_NUM_THREADS", 1);
    set_thread_env_var("VECLIB_MAXIMUM_THREADS", 1);
    if (openblas_set_num_threads != nullptr) {
        openblas_set_num_threads(1);
    }
    if (goto_set_num_threads != nullptr) {
        goto_set_num_threads(1);
    }
}

enum class QueryMode {
    kADC,
    kSDC,
};

struct Dataset {
    std::string name;
    int d = 0;
    RowMatrixXf xt;
    RowMatrixXf xb;
    RowMatrixXf xq;
    std::vector<int> gt;
    int gt_k = 0;
};

struct BenchmarkSummary {
    int budget_bits = 0;
    double train_total = 0.0;
    double structure_time = 0.0;
    double preparation_time = 0.0;
    double codebook_time = 0.0;
    double add_time = 0.0;
    double search_time = 0.0;
    double encode_per_vector = 0.0;
    double search_per_query = 0.0;
    double qps = 0.0;
    double recall1 = 0.0;
    double recall10 = 0.0;
    double recall100 = 0.0;
    double recall1000 = 0.0;
    double overlap1000 = 0.0;
    double reconstruction_error = std::numeric_limits<double>::quiet_NaN();
    double index_size_mib = std::numeric_limits<double>::quiet_NaN();
    bool train_only = false;
    bool skip_search = false;
};

struct Args {
    std::filesystem::path data_root = "data";
    std::string dataset;
    int bits = 0;
    std::vector<std::string> targets;
    std::optional<std::filesystem::path> epq_structure;
    std::optional<std::filesystem::path> config_path;
    std::optional<nlohmann::json> epq_config;
    QueryMode mode = QueryMode::kADC;
    int topk = 1000;
    int threads = 0;
    int recon_sample = 200000;
    std::optional<int> epq_transform_niter;
    std::optional<int> epq_kmeans_niter;
    std::optional<int> epq_transform_kmeans_niter;
    std::optional<int> vaq_subspaces;
    std::optional<int> vaq_min_bits;
    std::optional<int> vaq_max_bits;
    int vaq_validation_base = 0;
    int vaq_validation_queries = 0;
    int maxtrain = 0;
    bool train_only = false;
    bool skip_search = false;
};

sbi::Groups contiguous_balanced_groups(int d, int groups) {
    sbi::Groups out;
    out.reserve(static_cast<size_t>(groups));
    const int base = d / groups;
    const int rem = d % groups;
    int offset = 0;
    for (int i = 0; i < groups; ++i) {
        const int size = base + (i < rem ? 1 : 0);
        std::vector<int> dims;
        dims.reserve(static_cast<size_t>(size));
        for (int j = 0; j < size; ++j) {
            dims.push_back(offset++);
        }
        out.push_back(std::move(dims));
    }
    return out;
}

void maybe_print_uniform_pq_group_stats(
        const std::string& quantizer_name,
        const std::string& space_label,
        const RowMatrixXf& xt,
        int d,
        int total_bits,
        int groups,
        int nbits) {
    if (!sbi::group_stats_env_enabled()) {
        return;
    }
    epq::BuildContext ctx{
            .d = d,
            .total_bits = total_bits,
            .min_bits = 0,
            .max_bits = 12,
    };
    sbi::print_group_proxy_stats_from_matrix(
            std::cout,
            quantizer_name,
            space_label,
            contiguous_balanced_groups(d, groups),
            sbi::Bits(static_cast<size_t>(groups), nbits),
            xt,
            ctx);
}

void print_effective_group_residual_stats(
        std::ostream& os,
        const std::string& quantizer_name,
        const std::string& space_label,
        const sbi::Groups& groups,
        const sbi::Bits& bits,
        const RowMatrixXf& x_ref,
        const RowMatrixXf& x_recons) {
    if (groups.size() != bits.size()) {
        throw std::runtime_error("effective group stats groups/bits size mismatch");
    }
    if (x_ref.rows() != x_recons.rows() || x_ref.cols() != x_recons.cols()) {
        throw std::runtime_error(
                "effective group stats reference/reconstruction shape mismatch");
    }
    if (x_ref.rows() <= 0) {
        throw std::runtime_error(
                "effective group stats requires at least one sampled row");
    }

    int total_dims = 0;
    int total_bits = 0;
    for (size_t i = 0; i < groups.size(); ++i) {
        total_dims += static_cast<int>(groups[i].size());
        total_bits += bits[i];
        for (const int dim : groups[i]) {
            if (dim < 0 || dim >= x_ref.cols()) {
                throw std::runtime_error(
                        "effective group stats dimension out of range");
            }
        }
    }

    const auto old_flags = os.flags();
    const auto old_precision = os.precision();
    os << "\t[group-stats] quantizer=" << quantizer_name
       << " space=" << space_label
       << " metric=effective_residual"
       << " entries=" << groups.size()
       << " total_dims=" << total_dims
       << " total_bits=" << total_bits << '\n';

    double j_eff = 0.0;
    for (size_t gi = 0; gi < groups.size(); ++gi) {
        double d_eff = 0.0;
        const auto& dims = groups[gi];
        for (Eigen::Index row = 0; row < x_ref.rows(); ++row) {
            for (const int dim : dims) {
                const double diff = static_cast<double>(x_ref(row, dim)) -
                        static_cast<double>(x_recons(row, dim));
                d_eff += diff * diff;
            }
        }
        d_eff /= static_cast<double>(x_ref.rows());
        j_eff += d_eff;
        os << "\t[group-stats] group[" << std::setw(3) << std::setfill('0')
           << gi << std::setfill(' ')
           << "] ndims=" << dims.size()
           << " bits=" << bits[gi]
           << " D_proxy=" << std::fixed << std::setprecision(6)
           << d_eff
           << " D_eff=" << d_eff << '\n';
    }
    os << "\t[group-stats] J_proxy=" << std::fixed << std::setprecision(6)
       << j_eff << " J_eff=" << j_eff << '\n';
    os.flags(old_flags);
    os.precision(old_precision);
}

RowMatrixXf transform_rows_epq_space(
        const epq::IndexEPQ& index,
        const RowMatrixXf& x) {
    RowMatrixXf out(x.rows(), index.d);
    for (Eigen::Index i = 0; i < x.rows(); ++i) {
        index.transform_vector(x.row(i).data(), out.row(i).data());
    }
    return out;
}

RowMatrixXf transform_rows_bapq_space(
        const epq::IndexBAPQ& index,
        const RowMatrixXf& x) {
    RowMatrixXf out(x.rows(), index.d);
    for (Eigen::Index i = 0; i < x.rows(); ++i) {
        index.transform_vector(x.row(i).data(), out.row(i).data());
    }
    return out;
}

std::tuple<sbi::Groups, sbi::Bits> epq_effective_group_layout(
        const epq::IndexEPQ& index) {
    sbi::Groups groups = index.active_groups();
    sbi::Bits bits;
    bits.reserve(index.structure().groups.size());
    for (const auto& group : index.structure().groups) {
        bits.push_back(group.nbits);
    }
    return {std::move(groups), std::move(bits)};
}

struct RecordHeader {
    int dim = 0;
    size_t count = 0;
};

class BenchIndex {
   public:
    virtual ~BenchIndex() = default;
    virtual std::string name() const = 0;
    virtual int component_count() const = 0;
    virtual void train(const RowMatrixXf& xt) = 0;
    virtual void add(const RowMatrixXf& xb) = 0;
    virtual void search(
            const RowMatrixXf& xq,
            int k,
            QueryMode mode,
            std::vector<float>& distances,
            std::vector<faiss::idx_t>& labels) const = 0;
    virtual epq::TrainingStats training_stats() const = 0;
    virtual int effective_budget_bits() const {
        return 0;
    }
    virtual size_t serialized_payload_bytes() const {
        return 0;
    }
    virtual bool can_save_epq_structure() const {
        return false;
    }
    virtual void save_epq_structure(const std::filesystem::path& path) const {
        (void)path;
        throw std::runtime_error("EPQ structure save not supported");
    }
    virtual nlohmann::json method_metadata() const {
        return nlohmann::json::object();
    }
    virtual void print_diagnostics(std::ostream& os) const {
        (void)os;
    }
    virtual bool can_add_search() const {
        return true;
    }
    virtual bool supports_reconstruction() const {
        return false;
    }
    virtual void reconstruct_rows(
            const std::vector<faiss::idx_t>& ids,
            RowMatrixXf& out) const {
        (void)ids;
        (void)out;
        throw std::runtime_error("reconstruction not supported");
    }
    virtual void print_effective_group_stats(
            std::ostream& os,
            const RowMatrixXf& xb_sample,
            const RowMatrixXf& xb_recons) const {
        (void)os;
        (void)xb_sample;
        (void)xb_recons;
    }
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void log_stage_done(
        std::string_view owner,
        std::string_view stage,
        double seconds) {
    std::cout << "----- " << owner << ": " << stage
              << " done in " << std::fixed << std::setprecision(3)
              << seconds << " s\n";
}

template <typename T>
RecordHeader inspect_xvecs_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail("failed to open " + path.string());
    }
    int dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(int));
    if (!in || dim <= 0) {
        fail("invalid xvecs header in " + path.string());
    }
    const auto bytes = std::filesystem::file_size(path);
    const size_t record_size = static_cast<size_t>(dim + 1) * sizeof(T);
    if (bytes % record_size != 0) {
        fail("unexpected file size for " + path.string());
    }
    return RecordHeader{
            .dim = dim,
            .count = static_cast<size_t>(bytes / record_size),
    };
}

template <typename T>
std::vector<T> load_xvecs_flat(
        const std::filesystem::path& path,
        int* out_dim,
        size_t* out_count) {
    const auto header = inspect_xvecs_file<T>(path);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail("failed to open " + path.string());
    }

    std::vector<T> values(header.count * static_cast<size_t>(header.dim));
    std::vector<T> row(static_cast<size_t>(header.dim));
    for (size_t i = 0; i < header.count; ++i) {
        int dim = 0;
        in.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (!in || dim != header.dim) {
            fail("invalid record header while reading " + path.string());
        }
        in.read(reinterpret_cast<char*>(row.data()), sizeof(T) * row.size());
        if (!in) {
            fail("short read while reading " + path.string());
        }
        std::copy(
                row.begin(),
                row.end(),
                values.begin() + static_cast<std::ptrdiff_t>(i * row.size()));
    }

    if (out_dim != nullptr) {
        *out_dim = header.dim;
    }
    if (out_count != nullptr) {
        *out_count = header.count;
    }
    return values;
}

RowMatrixXf load_fvecs_matrix(const std::filesystem::path& path) {
    int dim = 0;
    size_t count = 0;
    auto values = load_xvecs_flat<float>(path, &dim, &count);
    Eigen::Map<RowMatrixXf> map(values.data(), static_cast<Eigen::Index>(count), dim);
    return map;
}

std::vector<int> load_ivecs_flat(
        const std::filesystem::path& path,
        int* out_dim,
        size_t* out_count) {
    int dim = 0;
    size_t count = 0;
    auto values = load_xvecs_flat<int>(path, &dim, &count);
    if (out_dim != nullptr) {
        *out_dim = dim;
    }
    if (out_count != nullptr) {
        *out_count = count;
    }
    return values;
}

Dataset load_dataset(
        std::string_view dataset_name,
        const std::filesystem::path& data_root) {
    Dataset ds;
    ds.name = std::string(dataset_name);
    if (dataset_name == "sift1M") {
        const auto root = data_root / "sift1M";
        ds.xt = load_fvecs_matrix(root / "sift_learn.fvecs");
        ds.xb = load_fvecs_matrix(root / "sift_base.fvecs");
        ds.xq = load_fvecs_matrix(root / "sift_query.fvecs");
        size_t gt_count = 0;
        ds.gt = load_ivecs_flat(root / "sift_groundtruth.ivecs", &ds.gt_k, &gt_count);
        ds.d = static_cast<int>(ds.xt.cols());
        if (gt_count != static_cast<size_t>(ds.xq.rows())) {
            fail("groundtruth/query size mismatch for sift1M");
        }
        return ds;
    }
    if (dataset_name == "gist1M") {
        const auto root = data_root / "gist1M";
        ds.xt = load_fvecs_matrix(root / "gist_learn.fvecs");
        ds.xb = load_fvecs_matrix(root / "gist_base.fvecs");
        ds.xq = load_fvecs_matrix(root / "gist_query.fvecs");
        size_t gt_count = 0;
        ds.gt = load_ivecs_flat(root / "gist_groundtruth.ivecs", &ds.gt_k, &gt_count);
        ds.d = static_cast<int>(ds.xt.cols());
        if (gt_count != static_cast<size_t>(ds.xq.rows())) {
            fail("groundtruth/query size mismatch for gist1M");
        }
        return ds;
    }
    if (dataset_name == "deep10M") {
        const auto root = data_root / "deep1b";
        ds.xt = load_fvecs_matrix(root / "learn.fvecs");
        ds.xb = load_fvecs_matrix(root / "base.fvecs");
        ds.xq = load_fvecs_matrix(root / "deep1B_queries.fvecs");
        size_t gt_count = 0;
        ds.gt = load_ivecs_flat(root / "deep10M_groundtruth.ivecs", &ds.gt_k, &gt_count);
        ds.d = static_cast<int>(ds.xt.cols());
        if (gt_count != static_cast<size_t>(ds.xq.rows())) {
            fail("groundtruth/query size mismatch for deep10M");
        }
        return ds;
    }
    fail("unsupported dataset: " + std::string(dataset_name));
}

void cap_training_set(Dataset& ds, int maxtrain) {
    if (maxtrain <= 0 || ds.xt.rows() <= maxtrain) {
        return;
    }
    ds.xt = ds.xt.topRows(maxtrain).eval();
}

double recall_at_k(
        const std::vector<faiss::idx_t>& labels,
        int nq,
        int topk,
        const std::vector<int>& gt,
        int gt_k,
        int k) {
    const int kk = std::min(k, topk);
    int hit = 0;
    for (int i = 0; i < nq; ++i) {
        const int truth = gt[static_cast<size_t>(i) * gt_k];
        for (int j = 0; j < kk; ++j) {
            if (labels[static_cast<size_t>(i) * topk + j] == truth) {
                ++hit;
                break;
            }
        }
    }
    return static_cast<double>(hit) / nq;
}

double overlap_at_k(
        const std::vector<faiss::idx_t>& labels,
        int nq,
        int topk,
        const std::vector<int>& gt,
        int gt_k,
        int k,
        int gt_use) {
    const int kk = std::min(k, topk);
    const int gg = std::min(gt_use, gt_k);
    double acc = 0.0;
    for (int i = 0; i < nq; ++i) {
        const int* gt_row = gt.data() + static_cast<size_t>(i) * gt_k;
        std::vector<int> truth(gt_row, gt_row + gg);
        std::sort(truth.begin(), truth.end());
        int count = 0;
        for (int j = 0; j < kk; ++j) {
            const auto id = static_cast<int>(labels[static_cast<size_t>(i) * topk + j]);
            count += std::binary_search(truth.begin(), truth.end(), id) ? 1 : 0;
        }
        acc += static_cast<double>(count) / gg;
    }
    return acc / nq;
}

std::vector<faiss::idx_t> sample_ids(faiss::idx_t n, int sample, uint32_t seed = 123) {
    if (sample <= 0 || sample >= n) {
        std::vector<faiss::idx_t> ids(static_cast<size_t>(n));
        std::iota(ids.begin(), ids.end(), 0);
        return ids;
    }
    std::vector<faiss::idx_t> ids(static_cast<size_t>(n));
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(ids.begin(), ids.end(), rng);
    ids.resize(sample);
    return ids;
}

struct HeapEntry {
    float dist;
    faiss::idx_t id;
    bool operator<(const HeapEntry& other) const {
        return dist < other.dist;
    }
};

class FaissPQIndex final : public BenchIndex {
   public:
    FaissPQIndex(int d, int total_bits, bool use_opq, std::string name)
            : d_(d),
              total_bits_(total_bits),
              M_(total_bits / 8),
              d2_(use_opq ? ((d + M_ - 1) / M_) * M_ : d),
              use_opq_(use_opq),
              name_(std::move(name)) {
        if (total_bits_ % 8 != 0) {
            fail("Faiss PQ/OPQ benchmark currently requires bits divisible by 8");
        }
        if (M_ <= 0) {
            fail("invalid M for Faiss PQ/OPQ");
        }
        if (d2_ % M_ != 0) {
            fail("invalid d2/M for Faiss PQ/OPQ");
        }
    }

    std::string name() const override {
        return name_;
    }

    int component_count() const override {
        return M_;
    }

    void train(const RowMatrixXf& xt) override {
        const auto t0 = std::chrono::steady_clock::now();
        stats_ = {};
        if (use_opq_) {
            RowMatrixXf x_train = xt;
            const auto prep0 = std::chrono::steady_clock::now();
            auto* opq = new faiss::OPQMatrix(d_, M_, d2_);
            opq_ = opq;
            opq->train(xt.rows(), xt.data());
            std::vector<float> out(
                    static_cast<size_t>(xt.rows()) * static_cast<size_t>(d2_));
            opq->apply_noalloc(xt.rows(), xt.data(), out.data());
            Eigen::Map<const RowMatrixXf> mapped(out.data(), xt.rows(), d2_);
            x_train = mapped;
            const auto prep1 = std::chrono::steady_clock::now();
            stats_.preparation_time =
                    std::chrono::duration<double>(prep1 - prep0).count();

            const auto cb0 = std::chrono::steady_clock::now();
            auto* pq = new faiss::IndexPQ(d2_, M_, 8, faiss::METRIC_L2);
            pq->train(x_train.rows(), x_train.data());
            const auto cb1 = std::chrono::steady_clock::now();
            stats_.codebook_time =
                    std::chrono::duration<double>(cb1 - cb0).count();
            maybe_print_uniform_pq_group_stats(
                    "OPQ",
                    "opq-space(d2=" + std::to_string(d2_) + ")",
                    x_train,
                    d2_,
                    total_bits_,
                    M_,
                    8);

            auto pre = std::make_unique<faiss::IndexPreTransform>(opq, pq);
            pre->own_fields = true;
            index_ = std::move(pre);
        } else {
            opq_ = nullptr;
            index_ = std::make_unique<faiss::IndexPQ>(d_, M_, 8, faiss::METRIC_L2);
            const auto cb0 = std::chrono::steady_clock::now();
            index_->train(xt.rows(), xt.data());
            const auto cb1 = std::chrono::steady_clock::now();
            stats_.codebook_time = std::chrono::duration<double>(cb1 - cb0).count();
            maybe_print_uniform_pq_group_stats(
                    "PQ",
                    "original",
                    xt,
                    d_,
                    total_bits_,
                    M_,
                    8);
        }
        stats_.structure_time = 0.0;
        const auto t1 = std::chrono::steady_clock::now();
        stats_.total_time = std::chrono::duration<double>(t1 - t0).count();
    }

    void add(const RowMatrixXf& xb) override {
        index_->add(xb.rows(), xb.data());
    }

    void search(
            const RowMatrixXf& xq,
            int k,
            QueryMode mode,
            std::vector<float>& distances,
            std::vector<faiss::idx_t>& labels) const override {
        if (mode != QueryMode::kADC) {
            fail(name_ + " benchmark currently uses Faiss ADC search only");
        }
        distances.resize(static_cast<size_t>(xq.rows()) * k);
        labels.resize(static_cast<size_t>(xq.rows()) * k);
        index_->search(xq.rows(), xq.data(), k, distances.data(), labels.data());
    }

    epq::TrainingStats training_stats() const override {
        return stats_;
    }

    int effective_budget_bits() const override {
        return total_bits_;
    }

    size_t serialized_payload_bytes() const override {
        if (!index_) {
            return 0;
        }
        return epq::serialized_faiss_index_bytes(*index_);
    }

    void print_diagnostics(std::ostream& os) const override {
        os << "\t[profile] backend=faiss native_index="
           << (use_opq_ ? "IndexPreTransform(OPQ+PQ)" : "IndexPQ")
           << " M=" << M_ << " nbits=8\n";
    }

    nlohmann::json method_metadata() const override {
        nlohmann::json meta = {
                {"family", use_opq_ ? "opq" : "pq"},
                {"impl", "faiss"},
                {"native_index",
                 use_opq_ ? "IndexPreTransform(OPQ+PQ)" : "IndexPQ"},
                {"d", d_},
                {"d2", d2_},
                {"M", M_},
                {"nbits", 8},
                {"metric", "L2"},
                {"total_bits", total_bits_},
                {"train_only", false},
        };
        if (use_opq_) {
            meta["opq"] = epq::benchmark_metadata::default_opq_metadata(d_, M_, d2_);
        }
        return meta;
    }

    bool supports_reconstruction() const override {
        return true;
    }

    void reconstruct_rows(
            const std::vector<faiss::idx_t>& ids,
            RowMatrixXf& out) const override {
        out.resize(static_cast<Eigen::Index>(ids.size()), d_);
        for (size_t i = 0; i < ids.size(); ++i) {
            index_->reconstruct(
                    ids[i],
                    out.row(static_cast<Eigen::Index>(i)).data());
        }
    }

    void print_effective_group_stats(
            std::ostream& os,
            const RowMatrixXf& xb_sample,
            const RowMatrixXf& xb_recons) const override {
        RowMatrixXf x_ref;
        RowMatrixXf x_rec;
        int stat_dim = d_;
        std::string quantizer = "PQ";
        std::string space = "original";
        if (use_opq_) {
            if (opq_ == nullptr) {
                fail("OPQ effective group stats requested before OPQ transform is available");
            }
            stat_dim = d2_;
            quantizer = "OPQ";
            space = "opq-space(d2=" + std::to_string(d2_) + ")";
            x_ref.resize(xb_sample.rows(), d2_);
            x_rec.resize(xb_recons.rows(), d2_);
            opq_->apply_noalloc(xb_sample.rows(), xb_sample.data(), x_ref.data());
            opq_->apply_noalloc(xb_recons.rows(), xb_recons.data(), x_rec.data());
        } else {
            x_ref = xb_sample;
            x_rec = xb_recons;
        }
        print_effective_group_residual_stats(
                os,
                quantizer,
                space,
                contiguous_balanced_groups(stat_dim, M_),
                sbi::Bits(static_cast<size_t>(M_), 8),
                x_ref,
                x_rec);
    }

   private:
    int d_;
    int total_bits_;
    int M_;
    int d2_;
    bool use_opq_;
    std::string name_;
    std::unique_ptr<faiss::Index> index_;
    faiss::OPQMatrix* opq_ = nullptr;
    epq::TrainingStats stats_;
};

class VAQBenchIndex final : public BenchIndex {
   public:
    VAQBenchIndex(
            int d,
            int total_bits,
            std::optional<int> subspaces,
            std::optional<int> min_bits,
            std::optional<int> max_bits,
            int validation_base_rows,
            int validation_query_rows,
            std::string name)
            : index_(
                      d,
                      total_bits,
                      subspaces.value_or(0),
                      min_bits.value_or(1),
                      max_bits.value_or(8)),
              validation_base_rows_(validation_base_rows),
              validation_query_rows_(validation_query_rows),
              name_(std::move(name)) {}

    std::string name() const override {
        return name_;
    }

    int component_count() const override {
        return index_.component_count();
    }

    void train(const RowMatrixXf& xt) override {
        if (validation_base_rows_ == 0 && validation_query_rows_ == 0) {
            index_.train(xt.rows(), xt.data());
            return;
        }
        if (validation_base_rows_ <= 0 || validation_query_rows_ <= 0 ||
            validation_base_rows_ + validation_query_rows_ >= xt.rows()) {
            fail("VAQ validation requires positive base/query rows and non-empty fitting rows");
        }

        const faiss::idx_t fit_rows =
                xt.rows() - validation_base_rows_ - validation_query_rows_;
        const float* validation_base =
                xt.data() + static_cast<size_t>(fit_rows) * index_.d;
        const float* validation_queries = validation_base +
                static_cast<size_t>(validation_base_rows_) * index_.d;
        index_.train(fit_rows, xt.data());
        index_.add(validation_base_rows_, validation_base);

        constexpr int k = 10;
        std::vector<float> approx_distances(
                static_cast<size_t>(validation_query_rows_) * k);
        std::vector<faiss::idx_t> approx_labels(
                static_cast<size_t>(validation_query_rows_) * k);
        index_.search(
                validation_query_rows_,
                validation_queries,
                k,
                approx_distances.data(),
                approx_labels.data());

        faiss::IndexFlatL2 exact(index_.d);
        exact.add(validation_base_rows_, validation_base);
        std::vector<float> exact_distances(
                static_cast<size_t>(validation_query_rows_) * k);
        std::vector<faiss::idx_t> exact_labels(
                static_cast<size_t>(validation_query_rows_) * k);
        exact.search(
                validation_query_rows_,
                validation_queries,
                k,
                exact_distances.data(),
                exact_labels.data());

        double overlap = 0.0;
        size_t top1_hits = 0;
        for (int query = 0; query < validation_query_rows_; ++query) {
            const auto base = static_cast<size_t>(query) * k;
            std::array<faiss::idx_t, k> truth{};
            std::copy_n(exact_labels.data() + base, k, truth.begin());
            std::sort(truth.begin(), truth.end());
            size_t matches = 0;
            for (int rank = 0; rank < k; ++rank) {
                const auto label = approx_labels[base + static_cast<size_t>(rank)];
                matches += std::binary_search(truth.begin(), truth.end(), label) ? 1 : 0;
                top1_hits += label == exact_labels[base] ? 1 : 0;
            }
            overlap += static_cast<double>(matches) / k;
        }
        validation_overlap_at_10_ = overlap / validation_query_rows_;
        validation_recall_at_10_ =
                static_cast<double>(top1_hits) / validation_query_rows_;
        validation_fit_rows_ = fit_rows;
    }

    void add(const RowMatrixXf& xb) override {
        index_.add(xb.rows(), xb.data());
    }

    void search(
            const RowMatrixXf& xq,
            int k,
            QueryMode mode,
            std::vector<float>& distances,
            std::vector<faiss::idx_t>& labels) const override {
        if (mode != QueryMode::kADC) {
            fail(name_ + " benchmark currently supports ADC search only");
        }
        distances.resize(static_cast<size_t>(xq.rows()) * k);
        labels.resize(static_cast<size_t>(xq.rows()) * k);
        index_.search(
                xq.rows(), xq.data(), k, distances.data(), labels.data());
    }

    epq::TrainingStats training_stats() const override {
        const auto& vaq_stats = index_.training_stats();
        return {
                .structure_time = vaq_stats.structure_time,
                .preparation_time = vaq_stats.preparation_time,
                .codebook_time = vaq_stats.codebook_time,
                .total_time = vaq_stats.total_time,
        };
    }

    int effective_budget_bits() const override {
        return index_.total_bits;
    }

    size_t serialized_payload_bytes() const override {
        return index_.serialized_payload_bytes();
    }

    nlohmann::json method_metadata() const override {
        auto meta = index_.metadata();
        if (validation_fit_rows_ > 0) {
            meta["validation"] = {
                    {"fit_rows", validation_fit_rows_},
                    {"base_rows", validation_base_rows_},
                    {"query_rows", validation_query_rows_},
                    {"overlap_at_10", validation_overlap_at_10_},
                    {"recall_at_10", validation_recall_at_10_},
                    {"selection_metric", "overlap_at_10_then_recall_at_10"},
            };
        }
        return meta;
    }

    void print_diagnostics(std::ostream& os) const override {
        os << "\t[profile] backend=TheDatumOrg/VAQ"
           << " subspaces=" << index_.component_count()
           << " code_size_bytes=" << index_.sa_code_size()
           << " bit_allocation=";
        const auto& bits = index_.bit_allocation();
        for (size_t i = 0; i < bits.size(); ++i) {
            if (i != 0) {
                os << ',';
            }
            os << bits[i];
        }
        os << '\n';
        if (validation_fit_rows_ > 0) {
            os << "\t[validation] fit_rows=" << validation_fit_rows_
               << " base_rows=" << validation_base_rows_
               << " query_rows=" << validation_query_rows_
               << " overlap@10=" << validation_overlap_at_10_
               << " recall@10=" << validation_recall_at_10_ << '\n';
        }
    }

    bool supports_reconstruction() const override {
        return true;
    }

    void reconstruct_rows(
            const std::vector<faiss::idx_t>& ids,
            RowMatrixXf& out) const override {
        out.resize(static_cast<Eigen::Index>(ids.size()), index_.d);
        for (size_t i = 0; i < ids.size(); ++i) {
            index_.reconstruct(
                    ids[i], out.row(static_cast<Eigen::Index>(i)).data());
        }
    }

   private:
    epq::IndexVAQ index_;
    int validation_base_rows_ = 0;
    int validation_query_rows_ = 0;
    faiss::idx_t validation_fit_rows_ = 0;
    double validation_overlap_at_10_ = std::numeric_limits<double>::quiet_NaN();
    double validation_recall_at_10_ = std::numeric_limits<double>::quiet_NaN();
    std::string name_;
};

// DP-OPQ adapter for Cai/Ji/Li, "Dynamic programming based
// optimized product quantization for approximate nearest neighbor search",
// Neurocomputing 217:110-118, 2016, DOI 10.1016/j.neucom.2016.01.112.
//
// This adapter follows the published algorithm because no public reference
// implementation is available. It exposes the PCA eigenvalue partition in
// the benchmark metadata for reproducibility.
class DPOPQBenchIndex final : public BenchIndex {
   public:
    DPOPQBenchIndex(int d, int total_bits, std::string name)
            : d_(d),
              total_bits_(total_bits),
              M_(total_bits / 8),
              name_(std::move(name)) {
        if (total_bits_ % 8 != 0) {
            fail("DP-OPQ benchmark currently requires bits divisible by 8");
        }
        if (M_ <= 0 || M_ > d_) {
            fail("invalid M for DP-OPQ");
        }
    }

    std::string name() const override {
        return name_;
    }

    int component_count() const override {
        return M_;
    }

    void train(const RowMatrixXf& xt) override {
        const auto t0 = std::chrono::steady_clock::now();
        stats_ = {};
        const auto prep0 = std::chrono::steady_clock::now();
        train_pca_rotation(xt);
        solve_dp_partition();
        RowMatrixXf x_rot = apply_transform(xt);
        const auto prep1 = std::chrono::steady_clock::now();
        stats_.preparation_time =
                std::chrono::duration<double>(prep1 - prep0).count();
        stats_.structure_time = stats_.preparation_time;

        const auto cb0 = std::chrono::steady_clock::now();
        const int niter = getenv_int_or("EPQ_DPOPQ_KMEANS_NITER", 25);
        const int nredo = getenv_int_or("EPQ_DPOPQ_KMEANS_NREDO", 1);
        codebooks_.clear();
        codebooks_.reserve(static_cast<size_t>(M_));
        for (int g = 0; g < M_; ++g) {
            const int begin = group_offsets_[static_cast<size_t>(g)];
            const int end = group_offsets_[static_cast<size_t>(g + 1)];
            RowMatrixXf sub = x_rot.middleCols(begin, end - begin).eval();
            codebooks_.push_back(train_kmeans(sub, 256, niter, nredo));
        }
        const auto cb1 = std::chrono::steady_clock::now();
        stats_.codebook_time =
                std::chrono::duration<double>(cb1 - cb0).count();
        stats_.total_time = std::chrono::duration<double>(cb1 - t0).count();
    }

    void add(const RowMatrixXf& xb) override {
        nb_ = static_cast<faiss::idx_t>(xb.rows());
        codes_.assign(
                static_cast<size_t>(nb_) * static_cast<size_t>(M_),
                uint8_t{0});
        const int chunk_rows = std::max(
                1, getenv_int_or("EPQ_DPOPQ_ADD_CHUNK_ROWS", 8192));
        for (Eigen::Index start = 0; start < xb.rows(); start += chunk_rows) {
            const Eigen::Index rows = std::min<Eigen::Index>(
                    chunk_rows, xb.rows() - start);
            RowMatrixXf x_rot = apply_transform(xb.middleRows(start, rows));
            encode_transformed_chunk(
                    x_rot,
                    static_cast<size_t>(start) * static_cast<size_t>(M_));
        }
    }

    void search(
            const RowMatrixXf& xq,
            int k,
            QueryMode mode,
            std::vector<float>& distances,
            std::vector<faiss::idx_t>& labels) const override {
        if (mode != QueryMode::kADC) {
            fail(name_ + " benchmark currently supports ADC search only");
        }
        if (nb_ <= 0 || codes_.empty()) {
            fail(name_ + " search requested before add");
        }
        const int nq = static_cast<int>(xq.rows());
        distances.assign(static_cast<size_t>(nq) * static_cast<size_t>(k), 0.0f);
        labels.assign(
                static_cast<size_t>(nq) * static_cast<size_t>(k),
                faiss::idx_t{-1});
        RowMatrixXf q_rot = apply_transform(xq);

#pragma omp parallel for schedule(static)
        for (int qi = 0; qi < nq; ++qi) {
            std::vector<float> lut(static_cast<size_t>(M_) * 256);
            for (int g = 0; g < M_; ++g) {
                const int begin = group_offsets_[static_cast<size_t>(g)];
                const int end = group_offsets_[static_cast<size_t>(g + 1)];
                const float* qptr = q_rot.row(qi).data() + begin;
                const RowMatrixXf& cb = codebooks_[static_cast<size_t>(g)];
                float* dst = lut.data() + static_cast<size_t>(g) * 256;
                for (int c = 0; c < 256; ++c) {
                    dst[c] = l2_distance(qptr, cb.row(c).data(), end - begin);
                }
            }

            std::priority_queue<HeapEntry> heap;
            for (faiss::idx_t id = 0; id < nb_; ++id) {
                const uint8_t* code = codes_.data() +
                        static_cast<size_t>(id) * static_cast<size_t>(M_);
                float dist = 0.0f;
                for (int g = 0; g < M_; ++g) {
                    dist += lut[static_cast<size_t>(g) * 256 + code[g]];
                }
                if (static_cast<int>(heap.size()) < k) {
                    heap.push(HeapEntry{dist, id});
                } else if (dist < heap.top().dist) {
                    heap.pop();
                    heap.push(HeapEntry{dist, id});
                }
            }

            const size_t row_off = static_cast<size_t>(qi) * static_cast<size_t>(k);
            for (int out = static_cast<int>(heap.size()) - 1; out >= 0; --out) {
                distances[row_off + static_cast<size_t>(out)] = heap.top().dist;
                labels[row_off + static_cast<size_t>(out)] = heap.top().id;
                heap.pop();
            }
        }
    }

    epq::TrainingStats training_stats() const override {
        return stats_;
    }

    int effective_budget_bits() const override {
        return total_bits_;
    }

    size_t serialized_payload_bytes() const override {
        size_t bytes = codes_.size();
        bytes += static_cast<size_t>(d_) * sizeof(float);
        bytes += static_cast<size_t>(d_) * static_cast<size_t>(d_) *
                sizeof(float);
        bytes += static_cast<size_t>(d_) * static_cast<size_t>(d_) *
                sizeof(float);
        bytes += transform_scales_.size() * sizeof(float);
        bytes += pc_order_.size() * sizeof(int);
        bytes += eigenvalues_.size() * sizeof(float);
        bytes += partition_values_.size() * sizeof(double);
        bytes += group_offsets_.size() * sizeof(int);
        for (const auto& cb : codebooks_) {
            bytes += static_cast<size_t>(cb.rows()) *
                    static_cast<size_t>(cb.cols()) * sizeof(float);
        }
        return bytes;
    }

    void print_diagnostics(std::ostream& os) const override {
        os << "\t[profile] backend=local native_index=DP-OPQ"
           << " M=" << M_ << " nbits=8"
           << " partition_cost=" << std::setprecision(6) << partition_cost_
           << " group_dims=";
        for (int g = 0; g < M_; ++g) {
            if (g != 0) {
                os << ',';
            }
            os << group_offsets_[static_cast<size_t>(g + 1)] -
                            group_offsets_[static_cast<size_t>(g)];
        }
        os << '\n';
    }

    nlohmann::json method_metadata() const override {
        nlohmann::json group_dims = nlohmann::json::array();
        nlohmann::json group_lambda_sums = nlohmann::json::array();
        nlohmann::json group_log_lambda_sums = nlohmann::json::array();
        for (int g = 0; g < M_; ++g) {
            const int begin = group_offsets_[static_cast<size_t>(g)];
            const int end = group_offsets_[static_cast<size_t>(g + 1)];
            group_dims.push_back(end - begin);
            double sum = 0.0;
            double log_sum = 0.0;
            for (int i = begin; i < end; ++i) {
                sum += eigenvalues_[static_cast<size_t>(i)];
                log_sum += partition_values_[static_cast<size_t>(i)];
            }
            group_lambda_sums.push_back(sum);
            group_log_lambda_sums.push_back(log_sum);
        }
        const std::string native_index = block_alignment_enabled_
                ? "PCA+DPLogEigenvalueAllocation+BlockAlignment+per-group-kmeans+ADC"
                : "PCA+DPLogEigenvalueAllocation+per-group-kmeans+ADC";
        return {
                {"family", "dpopq"},
                {"impl", "paper_based_implementation"},
                {"native_index", native_index},
                {"source_status", "no public reference implementation available"},
                {"paper", "Dynamic programming based optimized product quantization"},
                {"d", d_},
                {"M", M_},
                {"nbits", 8},
                {"total_bits", total_bits_},
                {"metric", "L2"},
                {"dpopq_variant",
                 "paper DP-embedding/DP-OPQa adapted recursively for flat M-way product codes"},
                {"partition_objective",
                 "equal-size PCA log-eigenvalue sums balanced by recursive DP allocation"},
                {"partition_discretization",
                 "log(lambda)-min(log(lambda)), rounded to 3 decimal digits before DP"},
                {"partition_units_exact", partition_units_exact_},
                {"partition_units_scale", partition_units_scale_},
                {"partition_units_sum", partition_units_sum_},
                {"block_alignment", block_alignment_enabled_},
                {"block_alignment_rule", "sqrt(min group-rank eigenvalue / eigenvalue)"},
                {"partition_cost", partition_cost_},
                {"group_dims", std::move(group_dims)},
                {"group_lambda_sums", std::move(group_lambda_sums)},
                {"group_log_lambda_sums", std::move(group_log_lambda_sums)},
                {"pc_order", pc_order_},
        };
    }

    bool supports_reconstruction() const override {
        return true;
    }

    void reconstruct_rows(
            const std::vector<faiss::idx_t>& ids,
            RowMatrixXf& out) const override {
        out.resize(static_cast<Eigen::Index>(ids.size()), d_);
        RowMatrixXf y(static_cast<Eigen::Index>(ids.size()), d_);
        y.setZero();
        for (size_t i = 0; i < ids.size(); ++i) {
            const uint8_t* code = codes_.data() +
                    static_cast<size_t>(ids[i]) * static_cast<size_t>(M_);
            for (int g = 0; g < M_; ++g) {
                const int begin = group_offsets_[static_cast<size_t>(g)];
                const int end = group_offsets_[static_cast<size_t>(g + 1)];
                y.block(
                         static_cast<Eigen::Index>(i),
                         begin,
                         1,
                         end - begin) =
                        codebooks_[static_cast<size_t>(g)].row(code[g]);
            }
        }
        out.noalias() = y * inverse_transform_;
        out.rowwise() += mean_;
    }

   private:
    static float l2_distance(const float* a, const float* b, int dim) {
        float acc = 0.0f;
        for (int j = 0; j < dim; ++j) {
            const float diff = a[j] - b[j];
            acc += diff * diff;
        }
        return acc;
    }

    static RowMatrixXf train_kmeans(
            const RowMatrixXf& x,
            int k,
            int niter,
            int nredo) {
        if (x.rows() <= 0 || x.cols() <= 0) {
            fail("DP-OPQ cannot train k-means on empty matrix");
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

    void train_pca_rotation(const RowMatrixXf& xt) {
        mean_ = xt.colwise().mean();
        RowMatrixXf centered = xt;
        centered.rowwise() -= mean_;
        Eigen::MatrixXf cov =
                (centered.transpose() * centered) /
                std::max<float>(1.0f, static_cast<float>(xt.rows() - 1));
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(cov);
        if (solver.info() != Eigen::Success) {
            fail("DP-OPQ PCA eigendecomposition failed");
        }
        pca_rotation_.resize(d_, d_);
        pca_eigenvalues_.assign(static_cast<size_t>(d_), 0.0f);
        for (int out = 0; out < d_; ++out) {
            const int src = d_ - 1 - out;
            pca_rotation_.col(out) = solver.eigenvectors().col(src);
            pca_eigenvalues_[static_cast<size_t>(out)] =
                    std::max(0.0f, solver.eigenvalues()[src]);
        }
        prepare_partition_weights();
    }

    RowMatrixXf apply_transform(const Eigen::Ref<const RowMatrixXf>& x) const {
        RowMatrixXf centered = x;
        centered.rowwise() -= mean_;
        return centered * rotation_;
    }

    static bool test_bit(const std::vector<uint64_t>& bits, int pos) {
        return (bits[static_cast<size_t>(pos / 64)] >> (pos % 64)) & uint64_t{1};
    }

    static void or_shift_left(
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

    std::vector<int> choose_balanced_subset_dp(
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

        std::vector<int> weights(items.size(), 0);
        for (size_t i = 0; i < items.size(); ++i) {
            weights[i] = partition_units_[static_cast<size_t>(items[i])];
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
            const int w = weights[static_cast<size_t>(pos)];
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
        if (best_sum < 0) {
            fail("DP-OPQ subset DP failed");
        }

        std::vector<int> selected_positions;
        selected_positions.reserve(static_cast<size_t>(take));
        int count = take;
        int sum = best_sum;
        while (count > 0) {
            const int pos = parent_item[state_index(count, sum)];
            const int prev = parent_prev_sum[state_index(count, sum)];
            if (pos < 0 || prev < 0) {
                fail("DP-OPQ subset DP invalid backpointer");
            }
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

    std::vector<std::vector<int>> partition_recursive(
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
                static_cast<double>(total_units) *
                static_cast<double>(left_groups) / static_cast<double>(groups)));
        const std::vector<int> left =
                choose_balanced_subset_dp(items, left_take, left_target);
        std::vector<char> in_left(static_cast<size_t>(d_), 0);
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

    void solve_dp_partition() {
        if (d_ % M_ != 0) {
            fail("DP-OPQ currently requires d divisible by M for equal-size subspaces");
        }
        std::vector<int> items(static_cast<size_t>(d_));
        std::iota(items.begin(), items.end(), 0);
        const auto groups = partition_recursive(items, M_);
        if (static_cast<int>(groups.size()) != M_) {
            fail("DP-OPQ partition produced wrong group count");
        }

        group_offsets_.assign(static_cast<size_t>(M_ + 1), 0);
        pc_order_.clear();
        pc_order_.reserve(static_cast<size_t>(d_));
        eigenvalues_.clear();
        eigenvalues_.reserve(static_cast<size_t>(d_));
        partition_values_.clear();
        partition_values_.reserve(static_cast<size_t>(d_));
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
                const float eigenvalue = pca_eigenvalues_[static_cast<size_t>(pc)];
                eigenvalues_.push_back(eigenvalue);
                const double partition_value =
                        pca_partition_values_[static_cast<size_t>(pc)];
                partition_values_.push_back(partition_value);
                sum += partition_value;
            }
            const double diff = sum - target;
            partition_cost_ += diff * diff;
        }
        group_offsets_[static_cast<size_t>(M_)] = static_cast<int>(pc_order_.size());
        if (static_cast<int>(pc_order_.size()) != d_) {
            fail("DP-OPQ partition did not cover all PCA components");
        }
        base_rotation_.resize(d_, d_);
        for (int out = 0; out < d_; ++out) {
            base_rotation_.col(out) =
                    pca_rotation_.col(pc_order_[static_cast<size_t>(out)]);
        }
        configure_block_alignment();
    }

    void prepare_partition_weights() {
        pca_partition_values_.assign(static_cast<size_t>(d_), 0.0);
        partition_units_.assign(static_cast<size_t>(d_), 0);

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
        std::vector<double> logs(static_cast<size_t>(d_), 0.0);
        for (int i = 0; i < d_; ++i) {
            const double log_value = std::log(std::max(
                    static_cast<double>(pca_eigenvalues_[static_cast<size_t>(i)]),
                    static_cast<double>(floor_value)));
            logs[static_cast<size_t>(i)] = log_value;
            min_log = std::min(min_log, log_value);
        }

        std::vector<int64_t> raw_units(static_cast<size_t>(d_), 0);
        int64_t raw_sum = 0;
        for (int i = 0; i < d_; ++i) {
            const double shifted = std::max(0.0, logs[static_cast<size_t>(i)] - min_log);
            pca_partition_values_[static_cast<size_t>(i)] = shifted;
            const int64_t units = static_cast<int64_t>(std::llround(shifted * 1000.0));
            raw_units[static_cast<size_t>(i)] = units;
            raw_sum += units;
        }

        const int default_max_units = d_ <= 200 ? 500000 : 20000;
        const int max_units = std::max(
                1024,
                getenv_int_or("EPQ_DPOPQ_DP_MAX_UNITS", default_max_units));
        partition_units_exact_ = raw_sum <= max_units;
        partition_units_scale_ = 1.0;
        if (!partition_units_exact_ && raw_sum > 0) {
            partition_units_scale_ =
                    static_cast<double>(max_units) / static_cast<double>(raw_sum);
        }

        partition_units_sum_ = 0;
        for (int i = 0; i < d_; ++i) {
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

    void configure_block_alignment() {
        block_alignment_enabled_ =
                getenv_int_or("EPQ_DPOPQ_BLOCK_ALIGN", 0) != 0;
        transform_scales_.assign(static_cast<size_t>(d_), 1.0f);
        if (block_alignment_enabled_ && M_ > 1) {
            const int group_dim = d_ / M_;
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
        Eigen::VectorXf inverse_scales(d_);
        for (int i = 0; i < d_; ++i) {
            const float scale = std::max(
                    transform_scales_[static_cast<size_t>(i)],
                    std::numeric_limits<float>::min());
            rotation_.col(i) *= scale;
            inverse_scales[i] = 1.0f / scale;
        }
        inverse_transform_ = inverse_scales.asDiagonal() * base_rotation_.transpose();
    }

    void encode_transformed_chunk(const RowMatrixXf& x_rot, size_t code_offset) {
        const int rows = static_cast<int>(x_rot.rows());
#pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; ++i) {
            uint8_t* out = codes_.data() + code_offset +
                    static_cast<size_t>(i) * static_cast<size_t>(M_);
            for (int g = 0; g < M_; ++g) {
                const int begin = group_offsets_[static_cast<size_t>(g)];
                const int end = group_offsets_[static_cast<size_t>(g + 1)];
                const float* xptr = x_rot.row(i).data() + begin;
                const RowMatrixXf& cb = codebooks_[static_cast<size_t>(g)];
                int best = 0;
                float best_dist = std::numeric_limits<float>::infinity();
                for (int c = 0; c < 256; ++c) {
                    const float dist =
                            l2_distance(xptr, cb.row(c).data(), end - begin);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best = c;
                    }
                }
                out[g] = static_cast<uint8_t>(best);
            }
        }
    }

    int d_;
    int total_bits_;
    int M_;
    std::string name_;
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
    std::vector<uint8_t> codes_;
    faiss::idx_t nb_ = 0;
    double partition_cost_ = 0.0;
    double partition_units_scale_ = 1.0;
    int partition_units_sum_ = 0;
    bool partition_units_exact_ = true;
    bool block_alignment_enabled_ = false;
    epq::TrainingStats stats_;
};

class FaissProductQuantizerTrainOnly final : public BenchIndex {
   public:
    FaissProductQuantizerTrainOnly(int d, int total_bits, bool use_opq, std::string name)
            : d_(d),
              total_bits_(total_bits),
              M_(total_bits / 8),
              use_opq_(use_opq),
              name_(std::move(name)) {
        if (total_bits_ % 8 != 0) {
            fail("Faiss ProductQuantizer benchmark requires bits divisible by 8");
        }
        if (M_ <= 0) {
            fail("invalid M for Faiss ProductQuantizer benchmark");
        }
        d2_ = use_opq_ ? ((d_ + M_ - 1) / M_) * M_ : d_;
        if (d2_ % M_ != 0) {
            fail("invalid d2/M for Faiss ProductQuantizer benchmark");
        }
    }

    std::string name() const override {
        return name_;
    }

    int component_count() const override {
        return M_;
    }

    void train(const RowMatrixXf& xt) override {
        const auto t0 = std::chrono::steady_clock::now();
        RowMatrixXf x_train = xt;
        stats_ = {};
        if (use_opq_) {
            const auto prep0 = std::chrono::steady_clock::now();
            opq_ = std::make_unique<faiss::OPQMatrix>(d_, M_, d2_);
            opq_->train(xt.rows(), xt.data());
            std::vector<float> out(
                    static_cast<size_t>(xt.rows()) * static_cast<size_t>(d2_));
            opq_->apply_noalloc(xt.rows(), xt.data(), out.data());
            Eigen::Map<const RowMatrixXf> mapped(out.data(), xt.rows(), d2_);
            x_train = mapped;
            const auto prep1 = std::chrono::steady_clock::now();
            stats_.preparation_time =
                    std::chrono::duration<double>(prep1 - prep0).count();
        } else {
            opq_.reset();
        }

        const auto cb0 = std::chrono::steady_clock::now();
        pq_ = std::make_unique<faiss::ProductQuantizer>(d2_, M_, 8);
        pq_->train(x_train.rows(), x_train.data());
        const auto cb1 = std::chrono::steady_clock::now();
        stats_.structure_time = 0.0;
        stats_.codebook_time =
                std::chrono::duration<double>(cb1 - cb0).count();
        stats_.total_time =
                std::chrono::duration<double>(cb1 - t0).count();
        maybe_print_uniform_pq_group_stats(
                use_opq_ ? "OPQ" : "PQ",
                use_opq_ ? "opq-space(d2=" + std::to_string(d2_) + ")"
                         : "original",
                x_train,
                d2_,
                total_bits_,
                M_,
                8);
    }

    void add(const RowMatrixXf&) override {
        fail(name_ + " is train-only");
    }

    void search(
            const RowMatrixXf&,
            int,
            QueryMode,
            std::vector<float>&,
            std::vector<faiss::idx_t>&) const override {
        fail(name_ + " is train-only");
    }

    epq::TrainingStats training_stats() const override {
        return stats_;
    }

    int effective_budget_bits() const override {
        return total_bits_;
    }

    void print_diagnostics(std::ostream& os) const override {
        os << "\t[profile] backend=faiss train_path="
           << (use_opq_ ? "OPQMatrix + ProductQuantizer.train"
                        : "ProductQuantizer.train")
           << " M=" << M_ << " nbits=8 d2=" << d2_ << '\n';
    }

    nlohmann::json method_metadata() const override {
        nlohmann::json meta = {
                {"family", use_opq_ ? "opq" : "pq"},
                {"impl", "faiss"},
                {"native_index",
                 use_opq_ ? "OPQMatrix+ProductQuantizer(train_only)"
                          : "ProductQuantizer(train_only)"},
                {"d", d_},
                {"d2", d2_},
                {"M", M_},
                {"nbits", 8},
                {"metric", "L2"},
                {"total_bits", total_bits_},
                {"train_only", true},
        };
        if (use_opq_) {
            meta["opq"] = epq::benchmark_metadata::default_opq_metadata(d_, M_, d2_);
        }
        return meta;
    }

    bool can_add_search() const override {
        return false;
    }

   private:
    int d_;
    int total_bits_;
    int M_;
    int d2_;
    bool use_opq_;
    std::string name_;
    std::unique_ptr<faiss::OPQMatrix> opq_;
    std::unique_ptr<faiss::ProductQuantizer> pq_;
    epq::TrainingStats stats_;
};

class FaissRaBitQIndex final : public BenchIndex {
   public:
    FaissRaBitQIndex(int d, int total_bits, std::string name)
            : d_(d),
              total_bits_(total_bits),
              nb_bits_(resolve_nb_bits(d, total_bits)),
              name_(std::move(name)) {}

    std::string name() const override {
        return name_;
    }

    int component_count() const override {
        return d_;
    }

    void train(const RowMatrixXf& xt) override {
        const auto t0 = std::chrono::steady_clock::now();
        stats_ = {};
        index_ = std::make_unique<faiss::IndexRaBitQ>(
                d_, faiss::METRIC_L2, nb_bits_);
        const auto cb0 = std::chrono::steady_clock::now();
        index_->train(xt.rows(), xt.data());
        const auto cb1 = std::chrono::steady_clock::now();
        stats_.codebook_time = std::chrono::duration<double>(cb1 - cb0).count();
        const auto t1 = std::chrono::steady_clock::now();
        stats_.total_time = std::chrono::duration<double>(t1 - t0).count();
    }

    void add(const RowMatrixXf& xb) override {
        index_->add(xb.rows(), xb.data());
    }

    void search(
            const RowMatrixXf& xq,
            int k,
            QueryMode mode,
            std::vector<float>& distances,
            std::vector<faiss::idx_t>& labels) const override {
        if (mode != QueryMode::kADC) {
            fail(name_ + " benchmark currently uses Faiss ADC search only");
        }
        distances.resize(static_cast<size_t>(xq.rows()) * k);
        labels.resize(static_cast<size_t>(xq.rows()) * k);
        index_->search(xq.rows(), xq.data(), k, distances.data(), labels.data());
    }

    epq::TrainingStats training_stats() const override {
        return stats_;
    }

    int effective_budget_bits() const override {
        return static_cast<int>(
                faiss::RaBitQuantizer(static_cast<size_t>(d_), faiss::METRIC_L2, nb_bits_)
                        .compute_code_size(static_cast<size_t>(d_), nb_bits_) *
                8);
    }

    size_t serialized_payload_bytes() const override {
        if (!index_) {
            return 0;
        }
        return epq::serialized_faiss_index_bytes(*index_);
    }

    void print_diagnostics(std::ostream& os) const override {
        os << "\t[profile] backend=faiss native_index=IndexRaBitQ"
           << " nb_bits=" << static_cast<int>(nb_bits_)
           << " nominal_budget_bits=" << total_bits_ << "\n";
    }

    nlohmann::json method_metadata() const override {
        return {
                {"family", "rabitq"},
                {"impl", "faiss"},
                {"native_index", "IndexRaBitQ"},
                {"d", d_},
                {"nb_bits", static_cast<int>(nb_bits_)},
                {"nominal_budget_bits", total_bits_},
                {"effective_budget_bits", effective_budget_bits()},
                {"metric", "L2"},
        };
    }

    bool supports_reconstruction() const override {
        return true;
    }

    void reconstruct_rows(
            const std::vector<faiss::idx_t>& ids,
            RowMatrixXf& out) const override {
        out.resize(static_cast<Eigen::Index>(ids.size()), d_);
        for (size_t i = 0; i < ids.size(); ++i) {
            index_->reconstruct(
                    ids[i],
                    out.row(static_cast<Eigen::Index>(i)).data());
        }
    }

   private:
    static uint8_t resolve_nb_bits(int d, int total_bits) {
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

    int d_;
    int total_bits_;
    uint8_t nb_bits_;
    std::string name_;
    std::unique_ptr<faiss::IndexRaBitQ> index_;
    epq::TrainingStats stats_;
};

class FaissAdditiveQuantizerIndex final : public BenchIndex {
   public:
    enum class Kind {
        kRQ,
        kLSQ,
    };

    enum class Storage {
        kNormQint8,
        kNormFromLUT,
        kLUTNonorm,
    };

    FaissAdditiveQuantizerIndex(
            int d,
            int total_bits,
            Kind kind,
            Storage storage,
            std::string name)
            : d_(d),
              total_bits_(total_bits),
              kind_(kind),
              storage_(storage),
              code_size_bytes_(total_bits / 8),
              M_(storage == Storage::kNormQint8 ? code_size_bytes_ - 1
                                                : code_size_bytes_),
              name_(std::move(name)) {
        if (total_bits_ % 8 != 0) {
            fail(name_ + " benchmark currently requires bits divisible by 8");
        }
        if (code_size_bytes_ < (storage_ == Storage::kNormQint8 ? 2 : 1)) {
            fail(name_ + " benchmark requires at least 16 total bits");
        }
        if (M_ <= 0) {
            fail("invalid stage count for " + name_);
        }
    }

    std::string name() const override {
        return name_;
    }

    int component_count() const override {
        return M_;
    }

    void train(const RowMatrixXf& xt) override {
        const auto t0 = std::chrono::steady_clock::now();
        stats_ = {};
        const auto search_type =
                storage_ == Storage::kNormQint8
                ? faiss::AdditiveQuantizer::ST_norm_qint8
                : storage_ == Storage::kNormFromLUT
                ? faiss::AdditiveQuantizer::ST_norm_from_LUT
                : faiss::AdditiveQuantizer::ST_LUT_nonorm;
        if (kind_ == Kind::kRQ) {
            auto index = std::make_unique<faiss::IndexResidualQuantizer>(
                    d_,
                    static_cast<size_t>(M_),
                    8,
                    faiss::METRIC_L2,
                    search_type);
            index->rq.max_beam_size = getenv_int_or("EPQ_RQ_MAX_BEAM_SIZE", 8);
            index_ = std::move(index);
        } else {
            index_ = std::make_unique<faiss::IndexLocalSearchQuantizer>(
                    d_,
                    static_cast<size_t>(M_),
                    8,
                    faiss::METRIC_L2,
                    search_type);
        }
        const auto cb0 = std::chrono::steady_clock::now();
        index_->train(xt.rows(), xt.data());
        const auto cb1 = std::chrono::steady_clock::now();
        stats_.codebook_time = std::chrono::duration<double>(cb1 - cb0).count();
        stats_.structure_time = 0.0;
        stats_.preparation_time = 0.0;
        stats_.total_time = std::chrono::duration<double>(cb1 - t0).count();
    }

    void add(const RowMatrixXf& xb) override {
        index_->add(xb.rows(), xb.data());
    }

    void search(
            const RowMatrixXf& xq,
            int k,
            QueryMode mode,
            std::vector<float>& distances,
            std::vector<faiss::idx_t>& labels) const override {
        if (mode != QueryMode::kADC) {
            fail(name_ + " benchmark currently uses Faiss ADC search only");
        }
        distances.resize(static_cast<size_t>(xq.rows()) * k);
        labels.resize(static_cast<size_t>(xq.rows()) * k);
        index_->search(xq.rows(), xq.data(), k, distances.data(), labels.data());
    }

    epq::TrainingStats training_stats() const override {
        return stats_;
    }

    int effective_budget_bits() const override {
        return index_ ? static_cast<int>(index_->code_size * 8) : total_bits_;
    }

    size_t serialized_payload_bytes() const override {
        if (!index_) {
            return 0;
        }
        return epq::serialized_faiss_index_bytes(*index_);
    }

    void print_diagnostics(std::ostream& os) const override {
        os << "\t[profile] backend=faiss native_index="
           << native_index_name() << " M=" << M_
           << " nbits=8 search_type=" << search_type_name();
        if (kind_ == Kind::kRQ) {
            const auto* rq =
                    dynamic_cast<const faiss::IndexResidualQuantizer*>(index_.get());
            if (rq != nullptr) {
                os << " max_beam_size=" << rq->rq.max_beam_size;
            }
        } else {
            const auto* lsq =
                    dynamic_cast<const faiss::IndexLocalSearchQuantizer*>(index_.get());
            if (lsq != nullptr) {
                os << " encode_ils_iters=" << lsq->lsq.encode_ils_iters;
            }
        }
        os << '\n';
    }

    nlohmann::json method_metadata() const override {
        nlohmann::json meta = {
                {"family", family_name()},
                {"impl", "faiss"},
                {"native_index", native_index_name()},
                {"d", d_},
                {"M", M_},
                {"nbits", 8},
                {"metric", "L2"},
                {"search_type", search_type_name()},
                {"code_size_bytes", code_size_bytes_},
                {"stage_bits", M_ * 8},
                {"total_bits", total_bits_},
                {"effective_budget_bits", effective_budget_bits()},
        };
        if (storage_ == Storage::kNormQint8) {
            meta["quantized_norm_bits"] = 8;
        } else if (storage_ == Storage::kNormFromLUT) {
            meta["norm_storage_bits"] = 0;
            meta["norm_from_lut"] = true;
        } else {
            meta["norm_storage_bits"] = 0;
            meta["norm_from_lut"] = false;
            meta["nonorm"] = true;
        }
        if (kind_ == Kind::kRQ) {
            if (const auto* rq =
                        dynamic_cast<const faiss::IndexResidualQuantizer*>(
                                index_.get());
                rq != nullptr) {
                meta["max_beam_size"] = rq->rq.max_beam_size;
                meta["train_type"] = rq->rq.train_type;
                meta["use_beam_LUT"] = rq->rq.use_beam_LUT;
            }
        } else {
            if (const auto* lsq =
                        dynamic_cast<const faiss::IndexLocalSearchQuantizer*>(
                                index_.get());
                lsq != nullptr) {
                meta["train_iters"] = lsq->lsq.train_iters;
                meta["encode_ils_iters"] = lsq->lsq.encode_ils_iters;
                meta["train_ils_iters"] = lsq->lsq.train_ils_iters;
                meta["icm_iters"] = lsq->lsq.icm_iters;
            }
        }
        return meta;
    }

    bool supports_reconstruction() const override {
        return true;
    }

    void reconstruct_rows(
            const std::vector<faiss::idx_t>& ids,
            RowMatrixXf& out) const override {
        out.resize(static_cast<Eigen::Index>(ids.size()), d_);
        for (size_t i = 0; i < ids.size(); ++i) {
            index_->reconstruct(
                    ids[i],
                    out.row(static_cast<Eigen::Index>(i)).data());
        }
    }

   private:
    const char* family_name() const {
        if (kind_ == Kind::kRQ) {
            return storage_ == Storage::kNormQint8
                    ? "rq"
                    : storage_ == Storage::kNormFromLUT ? "rq_lut"
                                                        : "rq_nonorm";
        }
        return storage_ == Storage::kNormQint8
                ? "lsq"
                : storage_ == Storage::kNormFromLUT ? "lsq_lut"
                                                    : "lsq_nonorm";
    }

    const char* native_index_name() const {
        return kind_ == Kind::kRQ ? "IndexResidualQuantizer"
                                  : "IndexLocalSearchQuantizer";
    }

    const char* search_type_name() const {
        return storage_ == Storage::kNormQint8
                ? "ST_norm_qint8"
                : storage_ == Storage::kNormFromLUT ? "ST_norm_from_LUT"
                                                    : "ST_LUT_nonorm";
    }

    int d_;
    int total_bits_;
    Kind kind_;
    Storage storage_;
    int code_size_bytes_;
    int M_;
    std::string name_;
    std::unique_ptr<faiss::IndexAdditiveQuantizer> index_;
    epq::TrainingStats stats_;
};

class EPQBenchIndex final : public BenchIndex {
   public:
    EPQBenchIndex(
            int d,
            int total_bits,
            bool use_transform,
            std::shared_ptr<epq::StructureBuilder> builder,
            const std::optional<nlohmann::json>& config,
            std::optional<int> transform_niter,
            std::optional<int> kmeans_niter,
            std::optional<int> transform_kmeans_niter,
            std::string name)
            : index_(d, total_bits, std::move(builder)),
              name_(std::move(name)) {
        if (config.has_value()) {
            epq::apply_index_training_config(index_, *config);
        }
        index_.use_uneven_transform = use_transform;
        if (transform_niter.has_value()) {
            index_.transform_niter = *transform_niter;
        }
        if (kmeans_niter.has_value()) {
            index_.kmeans_niter = *kmeans_niter;
        }
        if (transform_kmeans_niter.has_value()) {
            index_.transform_kmeans_niter = *transform_kmeans_niter;
        }
    }

    std::string name() const override {
        return name_;
    }

    int component_count() const override {
        return static_cast<int>(index_.structure().group_count());
    }

    void train(const RowMatrixXf& xt) override {
        index_.train(xt.rows(), xt.data());
    }

    void add(const RowMatrixXf& xb) override {
        index_.add(xb.rows(), xb.data());
    }

    void search(
            const RowMatrixXf& xq,
            int k,
            QueryMode mode,
            std::vector<float>& distances,
            std::vector<faiss::idx_t>& labels) const override {
        distances.resize(static_cast<size_t>(xq.rows()) * k);
        labels.resize(static_cast<size_t>(xq.rows()) * k);
        epq::SearchParametersEPQ params;
        params.mode = (mode == QueryMode::kADC) ? epq::SearchMode::kADC
                                                : epq::SearchMode::kSDC;
        index_.search(
                xq.rows(),
                xq.data(),
                k,
                distances.data(),
                labels.data(),
                &params);
    }

    epq::TrainingStats training_stats() const override {
        return index_.training_stats();
    }

    size_t serialized_payload_bytes() const override {
        return index_.serialized_payload_bytes();
    }

    void print_diagnostics(std::ostream& os) const override {
        const auto& tf = index_.transform_profile();
        const auto& groups = index_.codebook_profiles();
        const auto& rt = index_.runtime_profile();
        const auto& structure = index_.structure();

        os << "\t[profile] backend=epq groups=" << structure.group_count()
           << " uneven_transform=" << (tf.used ? "on" : "off") << '\n';
        os << "\t[profile] structure sizes=";
        const auto sizes = structure.group_sizes();
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (i > 0) {
                os << ',';
            }
            os << sizes[i];
        }
        os << " bits=";
        for (size_t i = 0; i < structure.groups.size(); ++i) {
            if (i > 0) {
                os << ',';
            }
            os << structure.groups[i].nbits;
        }
        os << '\n';
        if (structure.meta.is_object()) {
            const auto& meta = structure.meta;
            if (meta.contains("proxy_train_rows") || meta.contains("proxy_eval_rows")) {
                os << "\t[profile] structure proxy_train_rows="
                   << meta.value("proxy_train_rows", 0)
                   << " proxy_eval_rows="
                   << meta.value("proxy_eval_rows", 0) << '\n';
            }
            if (meta.contains("stage_times")) {
                const auto& stage_times = meta.at("stage_times");
                os << "\t[profile] structure stages";
                if (stage_times.contains("grow_time")) {
                    os << " grow=" << stage_times.value("grow_time", 0.0);
                }
                if (stage_times.contains("crystallize_time")) {
                    os << " crystallize="
                       << stage_times.value("crystallize_time", 0.0);
                }
                if (stage_times.contains("mbeam_time")) {
                    os << " mbeam=" << stage_times.value("mbeam_time", 0.0);
                }
                if (stage_times.contains("greedy_tail_time")) {
                    os << " greedy_tail="
                       << stage_times.value("greedy_tail_time", 0.0);
                }
                if (stage_times.contains("chain_tail_time")) {
                    os << " chain_tail="
                       << stage_times.value("chain_tail_time", 0.0);
                }
                os << " total=" << meta.value("pipeline_time", 0.0) << " s\n";
            }
            if (meta.contains("proxy_cache") && meta.at("proxy_cache").is_object()) {
                const auto& cache = meta.at("proxy_cache");
                os << "\t[profile] structure cache"
                   << " slices=" << (cache.value("cache_slices", true) ? "on" : "off")
                   << " D(hit=" << cache.value("d_hits", uint64_t{0})
                   << " miss=" << cache.value("d_misses", uint64_t{0})
                   << " size=" << cache.value("d_size", size_t{0})
                   << "/" << cache.value("d_capacity", size_t{0}) << ')'
                   << " Dfast(hit=" << cache.value("d_fast_hits", uint64_t{0})
                   << " miss=" << cache.value("d_fast_misses", uint64_t{0})
                   << " size=" << cache.value("d_fast_size", size_t{0})
                   << "/" << cache.value("d_fast_capacity", size_t{0}) << ')'
                   << " Xtr(hit=" << cache.value("xtr_hits", uint64_t{0})
                   << " miss=" << cache.value("xtr_misses", uint64_t{0})
                   << " entries=" << cache.value("xtr_size", size_t{0})
                   << " bytes=" << cache.value("xtr_bytes", size_t{0})
                   << "/" << cache.value("xtr_capacity_bytes", size_t{0}) << ')'
                   << " Xev(hit=" << cache.value("xev_hits", uint64_t{0})
                   << " miss=" << cache.value("xev_misses", uint64_t{0})
                   << " entries=" << cache.value("xev_size", size_t{0})
                   << " bytes=" << cache.value("xev_bytes", size_t{0})
                   << "/" << cache.value("xev_capacity_bytes", size_t{0}) << ')'
                   << " PCA(top=" << cache.value("pca_top_dims", 0)
                   << " hit=" << cache.value("pca_hits", uint64_t{0})
                   << " miss=" << cache.value("pca_misses", uint64_t{0})
                   << " size=" << cache.value("pca_size", size_t{0})
                   << "/" << cache.value("pca_capacity", size_t{0}) << ')'
                   << " PCAfast(top=" << cache.value("pca_fast_top_dims", 0)
                   << " hit=" << cache.value("pca_fast_hits", uint64_t{0})
                   << " miss=" << cache.value("pca_fast_misses", uint64_t{0})
                   << " size=" << cache.value("pca_fast_size", size_t{0})
                   << "/" << cache.value("pca_fast_capacity", size_t{0}) << ")\n";
            }
            if (meta.contains("proxy_work") && meta.at("proxy_work").is_object()) {
                const auto& work = meta.at("proxy_work");
                os << "\t[profile] structure work"
                   << " D(calls=" << work.value("d_calls", uint64_t{0})
                   << " empty=" << work.value("d_empty_calls", uint64_t{0})
                   << ") Dfast(calls=" << work.value("d_fast_calls", uint64_t{0})
                   << " empty=" << work.value("d_fast_empty_calls", uint64_t{0})
                   << ") kmeans(calls=" << work.value("kmeans_calls", uint64_t{0})
                   << " ksum=" << work.value("kmeans_k_total", uint64_t{0})
                   << " dimsum=" << work.value("kmeans_dims_total", uint64_t{0}) << ')'
                   << " kmeans_fast(calls=" << work.value("kmeans_fast_calls", uint64_t{0})
                   << " ksum=" << work.value("kmeans_fast_k_total", uint64_t{0})
                   << " dimsum=" << work.value("kmeans_fast_dims_total", uint64_t{0}) << ')'
                   << " pca(calls=" << work.value("pca_approx_calls", uint64_t{0})
                   << " fits=" << work.value("pca_fits", uint64_t{0})
                   << " full_dims=" << work.value("pca_full_dims_total", uint64_t{0})
                   << " proj_dims=" << work.value("pca_proj_dims_total", uint64_t{0})
                   << " tail_dims=" << work.value("pca_tail_dims_total", uint64_t{0})
                   << ") pca_fast(calls="
                   << work.value("pca_fast_approx_calls", uint64_t{0})
                   << " fits=" << work.value("pca_fast_fits", uint64_t{0})
                   << " full_dims="
                   << work.value("pca_fast_full_dims_total", uint64_t{0})
                   << " proj_dims="
                   << work.value("pca_fast_proj_dims_total", uint64_t{0})
                   << " tail_dims="
                   << work.value("pca_fast_tail_dims_total", uint64_t{0})
                   << ')'
                   << " solve_bits(calls=" << work.value("solve_bits_calls", uint64_t{0})
                   << " groups=" << work.value("solve_bits_groups_total", uint64_t{0})
                   << " cost_evals=" << work.value("solve_bits_cost_evals", uint64_t{0})
                   << " states=" << work.value("solve_bits_dp_states", uint64_t{0})
                   << " transitions="
                   << work.value("solve_bits_dp_transitions", uint64_t{0}) << ")\n";
            }
            if (meta.contains("chain_tail_profile") &&
                meta.at("chain_tail_profile").is_object()) {
                const auto& tail = meta.at("chain_tail_profile");
                os << "\t[profile] chain_tail"
                   << " iters=" << tail.value("iterations", uint64_t{0})
                   << " active_iters="
                   << tail.value("iters_with_candidates", uint64_t{0})
                   << " seeds_raw=" << tail.value("seeds_raw_total", uint64_t{0})
                   << " seeds_kept="
                   << tail.value("seeds_kept_total", uint64_t{0})
                   << " cand=" << tail.value("candidates_total", uint64_t{0})
                   << " reranked="
                   << tail.value("exact_local_reranked_total", uint64_t{0})
                   << " rerank_kept="
                   << tail.value("exact_local_kept_total", uint64_t{0})
                   << " exact_attempted="
                   << tail.value("exact_attempted", uint64_t{0})
                   << " exact_children="
                   << tail.value("exact_children", uint64_t{0})
                   << " dup=" << tail.value("exact_dup_pruned", uint64_t{0})
                   << " seen=" << tail.value("exact_seen_pruned", uint64_t{0})
                   << " prefix_cut="
                   << tail.value("prefix_cut_stops_total", uint64_t{0})
                   << " local_gate="
                   << tail.value("local_gate_pruned_total", uint64_t{0})
                   << " max_steps=" << tail.value("max_steps", uint64_t{0})
                   << " improved_iters="
                   << tail.value("improved_iters", uint64_t{0}) << '\n';
            }
        }

        if (tf.used) {
            os << "\t[profile] transform init_mode=" << tf.init_mode
               << " init_seed=" << tf.init_seed
               << " init_orthogonality_error="
               << std::setprecision(9) << tf.init_orthogonality_error
               << " train_rows=" << tf.train_rows
               << " eval_rows=" << tf.eval_rows
               << " proxy_max_bits=" << tf.proxy_max_bits
               << " proxy_iters=" << tf.proxy_iterations
               << " exact_polish_iters=" << tf.exact_polish_iters
               << " exact_iters=" << tf.exact_iterations
               << " iterations=" << tf.iterations_run
               << " total=" << std::fixed << std::setprecision(3)
               << tf.total_time << " s\n";
            for (const auto& it : tf.iterations) {
                os << "\t[profile] transform.iter=" << it.iteration
                   << " stage=" << (it.proxy_stage ? "proxy" : "exact")
                   << " total=" << it.total_time
                   << " codebook=" << it.codebook_time
                   << " quantize=" << it.quantize_time
                   << " procrustes=" << it.procrustes_time
                   << " eval=" << it.eval_time
                   << " objective=" << std::setprecision(6) << it.objective
                   << (it.objective_is_eval ? " eval_mse" : " train_mse")
                   << '\n';
            }
            if (tf.has_final_holdout) {
                os << "\t[profile] transform.final_holdout"
                   << " train_rows=" << tf.train_rows
                   << " eval_rows=" << tf.eval_rows
                   << " objective=" << std::setprecision(6)
                   << tf.final_holdout_mse
                   << " exact_mse"
                   << " total=" << tf.final_holdout_seconds << " s\n";
            }
            os << std::setprecision(3);
        }

        for (const auto& group : groups) {
            os << "\t[profile] codebook[" << group.group_index << "]"
               << " ndims=" << group.ndims
               << " bits=" << group.nbits
               << " ksub=" << group.ksub
               << " rows=" << group.train_rows
               << " time=" << std::fixed << std::setprecision(3)
               << group.seconds << " s\n";
        }

        os << "\t[profile] last_add rows=" << rt.last_add_rows
           << " total=" << std::fixed << std::setprecision(3)
           << rt.last_add_total_time
           << " assign=" << rt.last_add_assign_time << " s\n";
        os << "\t[profile] last_search mode="
           << (rt.last_search_mode == epq::SearchMode::kADC ? "adc" : "sdc")
           << " nq=" << rt.last_search_queries
           << " k=" << rt.last_search_k
           << " total=" << rt.last_search_total_time
           << " transform=" << rt.last_search_transform_time
           << " lut=" << rt.last_search_lut_time
           << " scan=" << rt.last_search_scan_time << " s\n";
    }

    nlohmann::json method_metadata() const override {
        return epq::benchmark_metadata::summarize_index_epq(
                index_,
                name_ == "EPQ(raw)" ? "repq" : "epq");
    }

    bool can_save_epq_structure() const override {
        return true;
    }

    void save_epq_structure(const std::filesystem::path& path) const override {
        index_.structure().save_json(path.string());
    }

    bool supports_reconstruction() const override {
        return true;
    }

    void reconstruct_rows(
            const std::vector<faiss::idx_t>& ids,
            RowMatrixXf& out) const override {
        out.resize(static_cast<Eigen::Index>(ids.size()), index_.d);
        for (size_t i = 0; i < ids.size(); ++i) {
            index_.reconstruct(ids[i], out.row(static_cast<Eigen::Index>(i)).data());
        }
    }

    void print_effective_group_stats(
            std::ostream& os,
            const RowMatrixXf& xb_sample,
            const RowMatrixXf& xb_recons) const override {
        const auto [groups, bits] = epq_effective_group_layout(index_);
        print_effective_group_residual_stats(
                os,
                name_ == "EPQ(raw)" ? "REPQ" : "EPQ",
                "epq-space",
                groups,
                bits,
                transform_rows_epq_space(index_, xb_sample),
                transform_rows_epq_space(index_, xb_recons));
    }

   private:
    epq::IndexEPQ index_;
    std::string name_;
};

class AREPQBenchIndex final : public BenchIndex {
   public:
    AREPQBenchIndex(
            int d,
            int total_bits,
            int tail_bits,
            int tail_stages,
            std::shared_ptr<epq::StructureBuilder> builder,
            const std::optional<nlohmann::json>& config,
            std::optional<int> transform_niter,
            std::optional<int> kmeans_niter,
            std::optional<int> transform_kmeans_niter,
            std::string name)
            : index_(d, total_bits, tail_bits, tail_stages, std::move(builder)),
              name_(std::move(name)) {
        if (config.has_value()) {
            epq::apply_index_training_config(index_.main_index(), *config);
        }
        if (transform_niter.has_value()) {
            index_.main_index().transform_niter = *transform_niter;
        }
        if (kmeans_niter.has_value()) {
            index_.main_index().kmeans_niter = *kmeans_niter;
        }
        if (transform_kmeans_niter.has_value()) {
            index_.main_index().transform_kmeans_niter = *transform_kmeans_niter;
        }
        index_.icm_iters = std::max(0, getenv_int_or("EPQ_AREPQ_ICM_ITERS", 2));
        index_.final_main_reassign =
                getenv_int_or("EPQ_AREPQ_FINAL_MAIN_REASSIGN", 0) != 0;
        index_.skip_stable_tail_reassign =
                getenv_int_or("EPQ_AREPQ_SKIP_STABLE_TAIL_REASSIGN", 1) != 0;
        const int legacy_tail_refine_iters =
                getenv_int_or("EPQ_AREPQ_TAIL_REFINE_ITERS", 1);
        index_.tail_alt_iters =
                std::max(
                        0,
                        getenv_int_or(
                                "EPQ_AREPQ_TAIL_ALT_ITERS",
                                legacy_tail_refine_iters));
        index_.tail_alt_update_weight = std::clamp(
                getenv_float_or("EPQ_AREPQ_TAIL_ALT_UPDATE_WEIGHT", 0.5f),
                0.0f,
                1.0f);
        index_.tail_kmeans_niter =
                std::max(1, getenv_int_or("EPQ_AREPQ_TAIL_KMEANS_NITER", 25));
        index_.tail_kmeans_nredo =
                std::max(1, getenv_int_or("EPQ_AREPQ_TAIL_KMEANS_NREDO", 1));
        index_.tail_beam_candidates = std::max(
                1,
                get_config_int_or_env(
                        config,
                        "arepq",
                        "tail_beam_candidates",
                        "EPQ_AREPQ_TAIL_BEAM",
                        1));
        index_.add_batch_rows =
                std::max(1, getenv_int_or("EPQ_AREPQ_ADD_BATCH_ROWS", 100000));
        index_.search_query_batch =
                std::max(1, getenv_int_or("EPQ_AREPQ_SEARCH_QUERY_BATCH", 4));
        index_.search_db_chunk =
                std::max(1024, getenv_int_or("EPQ_AREPQ_SEARCH_DB_CHUNK", 65536));
    }

    std::string name() const override {
        return name_;
    }

    int component_count() const override {
        return index_.component_count();
    }

    int effective_budget_bits() const override {
        return index_.effective_budget_bits();
    }

    void train(const RowMatrixXf& xt) override {
        index_.train(xt.rows(), xt.data());
    }

    void add(const RowMatrixXf& xb) override {
        index_.add(xb.rows(), xb.data());
    }

    void search(
            const RowMatrixXf& xq,
            int k,
            QueryMode mode,
            std::vector<float>& distances,
            std::vector<faiss::idx_t>& labels) const override {
        if (mode != QueryMode::kADC) {
            fail(name_ + " currently supports ADC mode only");
        }
        distances.resize(static_cast<size_t>(xq.rows()) * static_cast<size_t>(k));
        labels.resize(static_cast<size_t>(xq.rows()) * static_cast<size_t>(k));
        epq::SearchParametersEPQ params;
        params.mode = epq::SearchMode::kADC;
        index_.search(
                xq.rows(),
                xq.data(),
                k,
                distances.data(),
                labels.data(),
                &params);
    }

    epq::TrainingStats training_stats() const override {
        return index_.training_stats();
    }

    size_t serialized_payload_bytes() const override {
        return index_.serialized_payload_bytes();
    }

    bool can_save_epq_structure() const override {
        return true;
    }

    void save_epq_structure(const std::filesystem::path& path) const override {
        index_.main_index().structure().save_json(path.string());
    }

    nlohmann::json method_metadata() const override {
        const auto tail_memory = index_.tail_memory_stats();
        nlohmann::json meta = {
                {"family", "arepq"},
                {"impl", "cpp"},
                {"total_bits", index_.total_bits},
                {"main_bits", index_.main_bits},
                {"tail_bits", index_.tail_bits},
                {"tail_stages", index_.tail_stages},
                {"tail_ksub", index_.tail_ksub},
                {"tail_type", "full_dim_transformed_residual"},
                {"assignment", "additive_icm"},
                {"tail_training", "residual_tail_bcd"},
                {"tail_update", "relaxed_centroid_mean"},
                {"tail_update_acceptance", "monotone_train_mse"},
                {"search", "full_additive_adc_scan"},
                {"icm_iters", index_.icm_iters},
                {"final_main_reassign", index_.final_main_reassign},
                {"skip_stable_tail_reassign", index_.skip_stable_tail_reassign},
                {"tail_alt_iters", index_.tail_alt_iters},
                {"tail_alt_update_weight", index_.tail_alt_update_weight},
                {"tail_alt_initial_mse", index_.tail_alt_initial_mse()},
                {"tail_alt_best_mse", index_.tail_alt_best_mse()},
                {"tail_alt_final_mse", index_.tail_alt_final_mse()},
                {"tail_kmeans_niter", index_.tail_kmeans_niter},
                {"tail_kmeans_nredo", index_.tail_kmeans_nredo},
                {"tail_beam_candidates", index_.tail_beam_candidates},
                {"search_query_batch", index_.search_query_batch},
                {"search_db_chunk", index_.search_db_chunk},
                {"tail_memory",
                 {
                         {"payload_code_bytes", tail_memory.payload_code_bytes},
                         {"resident_flat_code_bytes", tail_memory.resident_flat_code_bytes},
                         {"serialized_codebook_bytes", tail_memory.serialized_codebook_bytes},
                         {"serialized_tail_bytes", tail_memory.serialized_tail_bytes()},
                         {"reconstruction_codebook_bytes", tail_memory.reconstruction_codebook_bytes},
                         {"transform_copy_bytes", tail_memory.transform_copy_bytes},
                         {"norm_table_entries", tail_memory.norm_table_entries},
                         {"norm_table_bytes", tail_memory.norm_table_bytes},
                         {"product_tail_table_entries", tail_memory.product_tail_table_entries},
                         {"product_tail_table_bytes", tail_memory.product_tail_table_bytes},
                         {"tail_pair_table_entries", tail_memory.tail_pair_table_entries},
                         {"tail_pair_table_bytes", tail_memory.tail_pair_table_bytes},
                         {"query_lut_entries_per_query", tail_memory.query_lut_entries_per_query},
                         {"query_lut_bytes_per_query", tail_memory.query_lut_bytes_per_query},
                         {"resident_search_model_bytes", tail_memory.resident_search_model_bytes()},
                         {"resident_auxiliary_table_bytes", tail_memory.resident_auxiliary_table_bytes()},
                         {"resident_model_bytes", tail_memory.resident_model_bytes()},
                 }},
        };
        meta["main"] = epq::benchmark_metadata::summarize_index_epq(
                index_.main_index(),
                "arepq_main");
        return meta;
    }

    void print_diagnostics(std::ostream& os) const override {
        const auto tail_memory = index_.tail_memory_stats();
        os << "\t[profile] backend=arepq"
           << " main_bits=" << index_.main_bits
           << " tail_bits=" << index_.tail_bits
           << " tail_stages=" << index_.tail_stages
           << " tail_ksub=" << index_.tail_ksub
           << " icm_iters=" << index_.icm_iters
           << " final_main_reassign=" << index_.final_main_reassign
           << " skip_stable_tail_reassign="
           << index_.skip_stable_tail_reassign
           << " tail_alt_iters=" << index_.tail_alt_iters
           << " tail_alt_weight=" << index_.tail_alt_update_weight
           << " tail_beam=" << index_.tail_beam_candidates
           << " tail_train=" << std::fixed << std::setprecision(3)
           << index_.tail_train_time() << " s\n";
        os << "\t[profile] tail_alt_mse initial="
           << index_.tail_alt_initial_mse()
           << " best=" << index_.tail_alt_best_mse()
           << " final=" << index_.tail_alt_final_mse() << '\n';
        os << "\t[profile] main groups="
           << index_.main_index().structure().group_count()
           << " ntotal=" << index_.ntotal
           << " qbatch=" << index_.search_query_batch
           << " db_chunk=" << index_.search_db_chunk << '\n';
        os << "\t[memory] tail_payload_bytes=" << tail_memory.payload_code_bytes
           << " resident_flat_tail_code_bytes="
           << tail_memory.resident_flat_code_bytes
           << " tail_codebook_bytes=" << tail_memory.serialized_codebook_bytes
           << " transform_copy_bytes=" << tail_memory.transform_copy_bytes
           << " product_tail_entries=" << tail_memory.product_tail_table_entries
           << " product_tail_bytes=" << tail_memory.product_tail_table_bytes
           << " tail_pair_entries=" << tail_memory.tail_pair_table_entries
           << " tail_pair_bytes=" << tail_memory.tail_pair_table_bytes
           << " norm_bytes=" << tail_memory.norm_table_bytes
           << " query_lut_bytes_per_query="
           << tail_memory.query_lut_bytes_per_query
           << " resident_search_model_bytes="
           << tail_memory.resident_search_model_bytes()
           << " resident_auxiliary_table_bytes="
           << tail_memory.resident_auxiliary_table_bytes()
           << " resident_model_bytes=" << tail_memory.resident_model_bytes()
           << '\n';
    }

    bool supports_reconstruction() const override {
        return true;
    }

    void reconstruct_rows(
            const std::vector<faiss::idx_t>& ids,
            RowMatrixXf& out) const override {
        index_.reconstruct_rows(ids, out);
    }

    void print_effective_group_stats(
            std::ostream& os,
            const RowMatrixXf& xb_sample,
            const RowMatrixXf& xb_recons) const override {
        const auto& main = index_.main_index();
        const auto [groups, bits] = epq_effective_group_layout(main);
        print_effective_group_residual_stats(
                os,
                "AREPQ",
                "epq-space(full-reconstruction)",
                groups,
                bits,
                transform_rows_epq_space(main, xb_sample),
                transform_rows_epq_space(main, xb_recons));
    }

   private:
    epq::IndexAREPQ index_;
    std::string name_;
};

class BAPQBenchIndex final : public BenchIndex {
   public:
    BAPQBenchIndex(int d, int total_bits, int max_train_rows, std::string name)
            : index_(d, total_bits, 4), name_(std::move(name)) {
        index_.bmax = 12;
        index_.seed = 123;
        index_.max_train_rows = max_train_rows;
        index_.pca_max_train_rows = max_train_rows;
        index_.kmeans_niter = 50;
        index_.kmeans_nredo = 3;
        index_.query_batch = 8;
        index_.db_chunk = 65536;
    }

    std::string name() const override {
        return name_;
    }

    int component_count() const override {
        return index_.component_count();
    }

    void train(const RowMatrixXf& xt) override {
        index_.train(xt.rows(), xt.data());
    }

    void add(const RowMatrixXf& xb) override {
        index_.add(xb.rows(), xb.data());
    }

    void search(
            const RowMatrixXf& xq,
            int k,
            QueryMode mode,
            std::vector<float>& distances,
            std::vector<faiss::idx_t>& labels) const override {
        if (mode != QueryMode::kADC) {
            fail("BAPQ benchmark currently supports ADC search only");
        }
        distances.resize(static_cast<size_t>(xq.rows()) * k);
        labels.resize(static_cast<size_t>(xq.rows()) * k);
        index_.search(
                xq.rows(),
                xq.data(),
                k,
                distances.data(),
                labels.data());
    }

    epq::TrainingStats training_stats() const override {
        const auto stats = index_.training_stats();
        return epq::TrainingStats{
                .structure_time = stats.structure_time,
                .preparation_time = stats.preparation_time,
                .codebook_time = stats.codebook_time,
                .total_time = stats.total_time,
        };
    }

    size_t serialized_payload_bytes() const override {
        return index_.serialized_payload_bytes();
    }

    void print_diagnostics(std::ostream& os) const override {
        os << "\t[profile] backend=bapq q=4 groups="
           << index_.component_count()
           << " active_groups=" << index_.active_component_count() << '\n';
        os << "\t[profile] group_sizes=";
        const auto& sizes = index_.group_sizes();
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (i > 0) {
                os << ',';
            }
            os << sizes[i];
        }
        os << " bits=";
        const auto& bits = index_.nbits_per_group();
        for (size_t i = 0; i < bits.size(); ++i) {
            if (i > 0) {
                os << ',';
            }
            os << bits[i];
        }
        os << '\n';
        os << "\t[profile] storage theoretical_codes_mb="
           << (static_cast<double>(index_.theoretical_code_bytes()) /
               (1024.0 * 1024.0))
           << " codebooks_mb="
           << (static_cast<double>(index_.codebook_bytes()) /
               (1024.0 * 1024.0))
           << " transform_mb="
           << (static_cast<double>(index_.transform_bytes()) /
               (1024.0 * 1024.0))
           << '\n';
    }

    nlohmann::json method_metadata() const override {
        return epq::benchmark_metadata::summarize_index_bapq(index_);
    }

    bool supports_reconstruction() const override {
        return true;
    }

    void reconstruct_rows(
            const std::vector<faiss::idx_t>& ids,
            RowMatrixXf& out) const override {
        index_.reconstruct_rows(ids, out);
    }

    void print_effective_group_stats(
            std::ostream& os,
            const RowMatrixXf& xb_sample,
            const RowMatrixXf& xb_recons) const override {
        sbi::Groups groups;
        const auto& sizes = index_.group_sizes();
        groups.reserve(sizes.size());
        int begin = 0;
        for (const int size : sizes) {
            std::vector<int> dims;
            dims.reserve(static_cast<size_t>(size));
            for (int j = 0; j < size; ++j) {
                dims.push_back(begin + j);
            }
            begin += size;
            groups.push_back(std::move(dims));
        }
        print_effective_group_residual_stats(
                os,
                "BAPQ",
                "pca-space",
                groups,
                index_.nbits_per_group(),
                transform_rows_bapq_space(index_, xb_sample),
                transform_rows_bapq_space(index_, xb_recons));
    }

   private:
    epq::IndexBAPQ index_;
    std::string name_;
};

class AVQBenchIndex final : public BenchIndex {
   public:
    AVQBenchIndex(int d, int total_bits, int topk, int threads, std::string name)
            : index_(d, total_bits), name_(std::move(name)) {
        index_.default_num_neighbors = std::max(topk, 1);
        index_.training_threads = threads;
        index_.search_threads = threads;
    }

    std::string name() const override {
        return name_;
    }

    int component_count() const override {
        const int dpb = index_.dimensions_per_block > 0
                ? index_.dimensions_per_block
                : std::max(1, static_cast<int>(std::round(
                                         static_cast<double>(index_.d) * 4.0 /
                                         std::max(index_.total_bits, 1))));
        return (index_.d + dpb - 1) / dpb;
    }

    void train(const RowMatrixXf& xt) override {
        index_.train(xt.rows(), xt.data());
    }

    void add(const RowMatrixXf& xb) override {
        index_.add(xb.rows(), xb.data());
    }

    void search(
            const RowMatrixXf& xq,
            int k,
            QueryMode mode,
            std::vector<float>& distances,
            std::vector<faiss::idx_t>& labels) const override {
        if (mode != QueryMode::kADC) {
            fail("AVQ benchmark currently supports ADC-style search only");
        }
        distances.resize(static_cast<size_t>(xq.rows()) * k);
        labels.resize(static_cast<size_t>(xq.rows()) * k);
        index_.search(
                xq.rows(),
                xq.data(),
                k,
                distances.data(),
                labels.data());
    }

    epq::TrainingStats training_stats() const override {
        const auto& stats = index_.training_stats();
        return epq::TrainingStats{
                .structure_time = stats.structure_time,
                .preparation_time = stats.preparation_time,
                .codebook_time = stats.codebook_time,
                .total_time = stats.total_time,
        };
    }

    int effective_budget_bits() const override {
        return index_.effective_budget_bits() > 0
                ? index_.effective_budget_bits()
                : index_.total_bits;
    }

    void print_diagnostics(std::ostream& os) const override {
        os << "\t[profile] backend=scann-avq build_stage=add"
           << " requested_bits=" << index_.total_bits
           << " effective_bits=" << effective_budget_bits()
           << " default_num_neighbors=" << index_.default_num_neighbors
           << " aq_threshold=" << index_.anisotropic_quantization_threshold
           << '\n';
    }

    nlohmann::json method_metadata() const override {
        auto meta = epq::benchmark_metadata::summarize_index_avq(index_);
        meta["impl"] = "scann";
        meta["build_stage"] = "add";
        meta["resolved_component_count"] = component_count();
        return meta;
    }

    bool supports_reconstruction() const override {
        return false;
    }

   private:
    epq::IndexAVQ index_;
    std::string name_;
};

std::filesystem::path default_epq_structure_path(
        const std::filesystem::path& data_root,
        std::string_view dataset,
        int bits,
        std::string_view config_fingerprint = {}) {
    std::string prefix;
    if (dataset == "sift1M") {
        prefix = "sift";
    } else if (dataset == "gist1M") {
        prefix = "gist";
    } else if (dataset == "deep10M") {
        prefix = "deep";
    } else {
        prefix = std::string(dataset);
    }
    std::string suffix;
    if (!config_fingerprint.empty()) {
        suffix = "_cfg" + std::string(config_fingerprint);
    }
    return data_root / "structures" /
            (prefix + "_" + std::to_string(bits) + "B" + suffix +
             "_epq_structure.json");
}

uint64_t fnv1a64_bytes(std::string_view data) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char ch : data) {
        hash ^= static_cast<uint64_t>(ch);
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::optional<std::string> epq_config_fingerprint(const Args& args) {
    if (!args.epq_config.has_value()) {
        return std::nullopt;
    }
    std::string payload;
    if (args.config_path.has_value()) {
        std::ifstream in(*args.config_path, std::ios::binary);
        if (in) {
            payload.assign(
                    std::istreambuf_iterator<char>(in),
                    std::istreambuf_iterator<char>());
        }
    }
    if (payload.empty()) {
        payload = args.epq_config->dump();
    }
    const uint64_t hash = fnv1a64_bytes(payload);
    std::ostringstream oss;
    oss << std::hex << hash;
    return oss.str();
}

bool should_auto_reuse_epq_structure(const Args& args) {
    bool auto_reuse_structure = true;
    if (args.epq_config.has_value()) {
        auto_reuse_structure =
                epq::should_auto_reuse_structure(*args.epq_config, true);
    }
    return auto_reuse_structure;
}

std::optional<std::filesystem::path> resolve_epq_structure_path(
        const Args& args,
        bool require_existing) {
    if (args.epq_structure.has_value()) {
        return args.epq_structure;
    }
    if (!should_auto_reuse_epq_structure(args)) {
        return std::nullopt;
    }
    const auto fingerprint = epq_config_fingerprint(args);
    const auto candidate =
            default_epq_structure_path(
                    args.data_root,
                    args.dataset,
                    args.bits,
                    fingerprint.value_or(""));
    if (!require_existing || std::filesystem::exists(candidate)) {
        return candidate;
    }
    return std::nullopt;
}

struct AREPQTailConfig {
    int tail_bits = 8;
    int tail_stages = 1;
};

AREPQTailConfig resolve_arepq_tail_config(const Args& args) {
    AREPQTailConfig cfg;
    cfg.tail_bits = std::max(
            1,
            get_config_int_or_env(
                    args.epq_config,
                    "arepq",
                    "tail_bits",
                    "EPQ_AREPQ_TAIL_BITS",
                    8));
    cfg.tail_stages = std::max(
            1,
            get_config_int_or_env(
                    args.epq_config,
                    "arepq",
                    "tail_stages",
                    "EPQ_AREPQ_TAIL_STAGES",
                    1));
    return cfg;
}

std::optional<std::filesystem::path> structure_save_path_for_target(
        const Args& args,
        const std::string& target) {
    if (args.maxtrain != 0) {
        return std::nullopt;
    }
    if (target == "epq" || target == "repq") {
        return resolve_epq_structure_path(args, false);
    }
    if (target == "arepq" || target == "arepq_fixed") {
        const auto tail = resolve_arepq_tail_config(args);
        if (args.bits <= tail.tail_bits * tail.tail_stages) {
            return std::nullopt;
        }
        Args main_args = args;
        main_args.bits = args.bits - tail.tail_bits * tail.tail_stages;
        main_args.epq_structure.reset();
        return resolve_epq_structure_path(main_args, false);
    }
    return std::nullopt;
}

BenchmarkSummary run_benchmark(
        BenchIndex& index,
        const Dataset& ds,
        int topk,
        QueryMode mode,
        int recon_sample,
        bool train_only,
        bool skip_search,
        const std::optional<std::filesystem::path>& structure_save_path =
                std::nullopt) {
    std::cout << "----- " << index.name() << ": train\n";
    const auto t0 = std::chrono::steady_clock::now();
    index.train(ds.xt);
    const auto t1 = std::chrono::steady_clock::now();
    BenchmarkSummary summary;
    summary.train_only = train_only;
    summary.skip_search = skip_search;
    summary.budget_bits = index.effective_budget_bits();
    const auto stats = index.training_stats();
    summary.train_total = stats.total_time > 0.0
            ? stats.total_time
            : std::chrono::duration<double>(t1 - t0).count();
    summary.structure_time = stats.structure_time;
    summary.preparation_time = stats.preparation_time;
    summary.codebook_time = stats.codebook_time;
    log_stage_done(index.name(), "train", summary.train_total);

    if (structure_save_path.has_value() &&
        index.can_save_epq_structure() &&
        !std::filesystem::exists(*structure_save_path)) {
        std::filesystem::create_directories(structure_save_path->parent_path());
        index.save_epq_structure(*structure_save_path);
        nlohmann::json saved = {
                {"path", structure_save_path->string()},
                {"owner", index.name()},
        };
        std::cout << "meta.structure_saved " << saved.dump() << '\n';
    }

    if (train_only) {
        return summary;
    }
    if (!index.can_add_search()) {
        fail(index.name() + " does not support add/search but train_only is false");
    }
    std::cout << "----- " << index.name() << ": add\n";
    index.add(ds.xb);
    const auto t2 = std::chrono::steady_clock::now();
    summary.add_time = std::chrono::duration<double>(t2 - t1).count();
    log_stage_done(index.name(), "add", summary.add_time);
    const size_t serialized_bytes = index.serialized_payload_bytes();
    if (serialized_bytes > 0) {
        summary.index_size_mib =
                static_cast<double>(serialized_bytes) / (1024.0 * 1024.0);
    }
    summary.encode_per_vector = summary.add_time / ds.xb.rows();

    auto compute_reconstruction_sample = [&]() {
        if (!index.supports_reconstruction()) {
            return;
        }
        const auto ids = sample_ids(ds.xb.rows(), recon_sample);
        if (ids.empty()) {
            return;
        }
        RowMatrixXf xb_sample(ids.size(), ds.d);
        for (size_t i = 0; i < ids.size(); ++i) {
            xb_sample.row(static_cast<Eigen::Index>(i)) = ds.xb.row(ids[i]);
        }
        RowMatrixXf xb_recons;
        index.reconstruct_rows(ids, xb_recons);
        summary.reconstruction_error =
                (xb_sample - xb_recons).array().square().sum() / xb_sample.rows();
        if (effective_group_stats_env_enabled()) {
            index.print_effective_group_stats(std::cout, xb_sample, xb_recons);
        }
    };

    if (skip_search) {
        compute_reconstruction_sample();
        return summary;
    }
    std::cout << "----- " << index.name() << ": search\n";
    std::vector<float> distances;
    std::vector<faiss::idx_t> labels;
    const auto search_t0 = std::chrono::steady_clock::now();
    index.search(ds.xq, topk, mode, distances, labels);
    const auto t3 = std::chrono::steady_clock::now();
    summary.search_time = std::chrono::duration<double>(t3 - search_t0).count();
    log_stage_done(index.name(), "search", summary.search_time);
    summary.search_per_query = summary.search_time / ds.xq.rows();
    summary.qps = ds.xq.rows() / summary.search_time;
    summary.recall1 =
            recall_at_k(labels, ds.xq.rows(), topk, ds.gt, ds.gt_k, 1);
    summary.recall10 =
            recall_at_k(labels, ds.xq.rows(), topk, ds.gt, ds.gt_k, 10);
    summary.recall100 =
            recall_at_k(labels, ds.xq.rows(), topk, ds.gt, ds.gt_k, 100);
    summary.recall1000 =
            recall_at_k(labels, ds.xq.rows(), topk, ds.gt, ds.gt_k, 1000);
    summary.overlap1000 =
            overlap_at_k(labels, ds.xq.rows(), topk, ds.gt, ds.gt_k, 1000, 1000);

    compute_reconstruction_sample();

    return summary;
}

void print_summary(
        const BenchIndex& index,
        const BenchmarkSummary& summary) {
    const auto method_meta = index.method_metadata();
    if (!method_meta.empty()) {
        std::cout << "meta.method " << method_meta.dump() << '\n';
    }
    std::cout << "===== " << index.name() << '\n';
    std::cout << "\tM: " << index.component_count() << '\n';
    std::cout << "\tstructure time: " << std::fixed << std::setprecision(3)
              << summary.structure_time << " s\n";
    std::cout << "\tpreparation time: " << summary.preparation_time << " s\n";
    std::cout << "\tcodebook time: " << summary.codebook_time << " s\n";
    std::cout << "\ttraining total: " << summary.train_total << " s\n";
    if (summary.budget_bits > 0) {
        std::cout << "\teffective budget bits: " << summary.budget_bits << '\n';
    }
    if (summary.train_only) {
        index.print_diagnostics(std::cout);
        return;
    }
    std::cout << "\tadd/encode time: " << summary.add_time << " s\n";
    std::cout << "\tencode per vector: " << std::setprecision(9)
              << summary.encode_per_vector << " s/vector\n";
    if (std::isfinite(summary.index_size_mib)) {
        std::cout << std::setprecision(3)
                  << "\tserialized index size: " << summary.index_size_mib
                  << " MiB\n";
    }
    if (std::isfinite(summary.reconstruction_error)) {
        std::cout << "\treconstruction error (sample): "
                  << summary.reconstruction_error << '\n';
    }
    if (summary.skip_search) {
        index.print_diagnostics(std::cout);
        return;
    }
    std::cout << std::setprecision(3);
    std::cout << "\tsearch time: " << summary.search_time << " s\n";
    std::cout << "\tsearch per query: " << std::setprecision(9)
              << summary.search_per_query << " s/query\n";
    std::cout << std::setprecision(3);
    std::cout << "\tQPS: " << summary.qps << '\n';
    std::cout << "\trecall@1: " << std::setprecision(4) << summary.recall1
              << " recall@10: " << summary.recall10
              << " recall@100: " << summary.recall100
              << " recall@1000: " << summary.recall1000 << '\n';
    std::cout << "\toverlap@1000(gt=1000): " << summary.overlap1000 << '\n';
    index.print_diagnostics(std::cout);
}

Args parse_args(int argc, char** argv) {
    if (argc < 4) {
        fail(
                "usage: flat_benchmark <dataset> <bits> <target...> "
                "[--config=PATH] "
                "[--data-root=PATH] [--epq-structure=PATH] [--mode=adc|sdc] "
                "[--topk=1000] [--threads=N] [--recon-sample=N] "
                "[--epq-transform-niter=N] [--epq-kmeans-niter=N] "
                "[--epq-transform-kmeans-niter=N] [--maxtrain=N] "
                "[--vaq-subspaces=N] [--vaq-min-bits=N] [--vaq-max-bits=N] "
                "[--vaq-validation-base=N] [--vaq-validation-queries=N] "
                "[--train-only] [--skip-search]");
    }

    Args args;
    args.dataset = argv[1];
    args.bits = std::stoi(argv[2]);
    for (int i = 3; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg.starts_with("--data-root=")) {
            args.data_root = std::string(arg.substr(12));
        } else if (arg.starts_with("--config=")) {
            args.config_path = std::filesystem::path(std::string(arg.substr(9)));
            args.epq_config = epq::load_json_file(*args.config_path);
        } else if (arg.starts_with("--epq-structure=")) {
            args.epq_structure = std::filesystem::path(std::string(arg.substr(16)));
        } else if (arg.starts_with("--mode=")) {
            const auto value = arg.substr(7);
            if (value == "adc") {
                args.mode = QueryMode::kADC;
            } else if (value == "sdc") {
                args.mode = QueryMode::kSDC;
            } else {
                fail("invalid --mode");
            }
        } else if (arg.starts_with("--topk=")) {
            args.topk = std::stoi(std::string(arg.substr(7)));
        } else if (arg.starts_with("--threads=")) {
            args.threads = std::stoi(std::string(arg.substr(10)));
        } else if (arg.starts_with("--recon-sample=")) {
            args.recon_sample = std::stoi(std::string(arg.substr(15)));
        } else if (arg.starts_with("--epq-transform-niter=")) {
            args.epq_transform_niter = std::stoi(std::string(arg.substr(22)));
        } else if (arg.starts_with("--epq-kmeans-niter=")) {
            args.epq_kmeans_niter = std::stoi(std::string(arg.substr(19)));
        } else if (arg.starts_with("--epq-transform-kmeans-niter=")) {
            args.epq_transform_kmeans_niter =
                    std::stoi(std::string(arg.substr(29)));
        } else if (arg.starts_with("--vaq-subspaces=")) {
            args.vaq_subspaces = std::stoi(std::string(arg.substr(16)));
        } else if (arg.starts_with("--vaq-min-bits=")) {
            args.vaq_min_bits = std::stoi(std::string(arg.substr(15)));
        } else if (arg.starts_with("--vaq-max-bits=")) {
            args.vaq_max_bits = std::stoi(std::string(arg.substr(15)));
        } else if (arg.starts_with("--vaq-validation-base=")) {
            args.vaq_validation_base = std::stoi(std::string(arg.substr(22)));
        } else if (arg.starts_with("--vaq-validation-queries=")) {
            args.vaq_validation_queries = std::stoi(std::string(arg.substr(25)));
        } else if (arg.starts_with("--maxtrain=")) {
            args.maxtrain = std::stoi(std::string(arg.substr(11)));
        } else if (arg == "--train-only") {
            args.train_only = true;
        } else if (arg == "--skip-search") {
            args.skip_search = true;
        } else if (arg.starts_with("--")) {
            fail("unknown flag: " + std::string(arg));
        } else {
            args.targets.emplace_back(arg);
        }
    }
    if (args.targets.empty()) {
        fail("no targets specified");
    }
    if (args.bits <= 0) {
        fail("bits must be positive");
    }
    if (args.topk <= 0) {
        fail("topk must be positive");
    }
    if (args.train_only && args.skip_search) {
        fail("--train-only and --skip-search are mutually exclusive");
    }
    if ((args.vaq_validation_base == 0) !=
        (args.vaq_validation_queries == 0)) {
        fail("VAQ validation base/query flags must be set together");
    }
    return args;
}

std::shared_ptr<epq::StructureBuilder> make_epq_builder(
        const Args& args) {
    auto structure_path = resolve_epq_structure_path(args, true);
    if (structure_path.has_value()) {
        auto structure = epq::Structure::load_json(structure_path->string());
        return std::make_shared<epq::FixedStructureBuilder>(std::move(structure));
    }
    if (args.epq_config.has_value()) {
        const auto base_dir = args.config_path.has_value()
                ? args.config_path->parent_path()
                : std::filesystem::path();
        return epq::make_structure_builder_from_config(
                *args.epq_config,
                base_dir);
    }
    return std::make_shared<epq::RefinedStructureBuilder>();
}

std::unique_ptr<BenchIndex> make_target(
        const Args& args,
        const Dataset& ds,
        const std::string& target) {
    if (target == "pq") {
        if (args.train_only) {
            return std::make_unique<FaissProductQuantizerTrainOnly>(
                    ds.d, args.bits, false, "PQ(train-only)");
        }
        return std::make_unique<FaissPQIndex>(ds.d, args.bits, false, "PQ");
    }
    if (target == "opq") {
        if (args.train_only) {
            return std::make_unique<FaissProductQuantizerTrainOnly>(
                    ds.d, args.bits, true, "OPQ(train-only)");
        }
        return std::make_unique<FaissPQIndex>(ds.d, args.bits, true, "OPQ");
    }
    if (target == "dpopq" || target == "dp_opq" || target == "dp-opq") {
        return std::make_unique<DPOPQBenchIndex>(ds.d, args.bits, "DP-OPQ");
    }
    if (target == "vaq") {
        return std::make_unique<VAQBenchIndex>(
                ds.d,
                args.bits,
                args.vaq_subspaces,
                args.vaq_min_bits,
                args.vaq_max_bits,
                args.vaq_validation_base,
                args.vaq_validation_queries,
                "VAQ");
    }
    if (target == "rabitq") {
        return std::make_unique<FaissRaBitQIndex>(ds.d, args.bits, "RaBitQ");
    }
    if (target == "rq") {
        return std::make_unique<FaissAdditiveQuantizerIndex>(
                ds.d,
                args.bits,
                FaissAdditiveQuantizerIndex::Kind::kRQ,
                FaissAdditiveQuantizerIndex::Storage::kNormQint8,
                "RQ");
    }
    if (target == "rq_lut") {
        return std::make_unique<FaissAdditiveQuantizerIndex>(
                ds.d,
                args.bits,
                FaissAdditiveQuantizerIndex::Kind::kRQ,
                FaissAdditiveQuantizerIndex::Storage::kNormFromLUT,
                "RQ(LUT-norm)");
    }
    if (target == "rq_nonorm") {
        return std::make_unique<FaissAdditiveQuantizerIndex>(
                ds.d,
                args.bits,
                FaissAdditiveQuantizerIndex::Kind::kRQ,
                FaissAdditiveQuantizerIndex::Storage::kLUTNonorm,
                "RQ(nonorm)");
    }
    if (target == "lsq") {
        return std::make_unique<FaissAdditiveQuantizerIndex>(
                ds.d,
                args.bits,
                FaissAdditiveQuantizerIndex::Kind::kLSQ,
                FaissAdditiveQuantizerIndex::Storage::kNormQint8,
                "LSQ");
    }
    if (target == "lsq_lut") {
        return std::make_unique<FaissAdditiveQuantizerIndex>(
                ds.d,
                args.bits,
                FaissAdditiveQuantizerIndex::Kind::kLSQ,
                FaissAdditiveQuantizerIndex::Storage::kNormFromLUT,
                "LSQ(LUT-norm)");
    }
    if (target == "lsq_nonorm") {
        return std::make_unique<FaissAdditiveQuantizerIndex>(
                ds.d,
                args.bits,
                FaissAdditiveQuantizerIndex::Kind::kLSQ,
                FaissAdditiveQuantizerIndex::Storage::kLUTNonorm,
                "LSQ(nonorm)");
    }
    if (target == "epq") {
        return std::make_unique<EPQBenchIndex>(
                ds.d,
                args.bits,
                true,
                make_epq_builder(args),
                args.epq_config,
                args.epq_transform_niter,
                args.epq_kmeans_niter,
                args.epq_transform_kmeans_niter,
                "EPQ");
    }
    if (target == "arepq" || target == "arepq_fixed") {
        const auto tail = resolve_arepq_tail_config(args);
        const int tail_bits = tail.tail_bits;
        const int tail_stages = tail.tail_stages;
        if (args.bits <= tail_bits * tail_stages) {
            fail("arepq requires total bits larger than tail_bits * tail_stages");
        }
        Args main_args = args;
        main_args.bits = args.bits - tail_bits * tail_stages;
        return std::make_unique<AREPQBenchIndex>(
                ds.d,
                args.bits,
                tail_bits,
                tail_stages,
                make_epq_builder(main_args),
                args.epq_config,
                args.epq_transform_niter,
                args.epq_kmeans_niter,
                args.epq_transform_kmeans_niter,
                "AR-EPQ");
    }
    if (target == "repq") {
        return std::make_unique<EPQBenchIndex>(
                ds.d,
                args.bits,
                false,
                make_epq_builder(args),
                args.epq_config,
                args.epq_transform_niter,
                args.epq_kmeans_niter,
                args.epq_transform_kmeans_niter,
                "EPQ(raw)");
    }
    if (target == "bapq") {
        return std::make_unique<BAPQBenchIndex>(
                ds.d,
                args.bits,
                ds.xt.rows(),
                "BAPQ");
    }
    if (target == "avq") {
#if EPQ_ENABLE_AVQ
        return std::make_unique<AVQBenchIndex>(
                ds.d,
                args.bits,
                args.topk,
                args.threads,
                "AVQ");
#else
        fail("AVQ target was requested, but EPQ_ENABLE_AVQ is disabled at build time");
#endif
    }
    fail("unsupported target: " + target);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const int effective_threads =
                args.threads > 0 ? args.threads : omp_get_max_threads();
        if (args.threads > 0) {
            cap_downstream_thread_pools(args.threads);
        }
        if (args.epq_config.has_value()) {
            epq::apply_faiss_runtime_config(*args.epq_config);
        }
        Dataset ds = load_dataset(args.dataset, args.data_root);
        const size_t nt_full = static_cast<size_t>(ds.xt.rows());
        cap_training_set(ds, args.maxtrain);
        std::cout << "dataset=" << ds.name
                  << " d=" << ds.d
                  << " nb=" << ds.xb.rows()
                  << " nq=" << ds.xq.rows()
                  << " nt=" << ds.xt.rows()
                  << " gt_k=" << ds.gt_k
                  << " bits=" << args.bits
                  << " mode=" << (args.mode == QueryMode::kADC ? "adc" : "sdc")
                  << " train_only=" << (args.train_only ? "true" : "false")
                  << " skip_search=" << (args.skip_search ? "true" : "false")
                  << " topk=" << args.topk
                  << " metric_topk=" << args.topk
                  << " recon_sample=" << args.recon_sample
                  << " threads=" << effective_threads
                  << " maxtrain=" << args.maxtrain
                  << " faiss_blas_threshold="
                  << faiss::distance_compute_blas_threshold
                  << " faiss_blas_query_bs="
                  << faiss::distance_compute_blas_query_bs
                  << " faiss_blas_database_bs="
                  << faiss::distance_compute_blas_database_bs
                  << '\n';
        nlohmann::json run_meta = {
                {"benchmark", "flat_benchmark"},
                {"protocol", "flat"},
                {"query_mode", args.mode == QueryMode::kADC ? "adc" : "sdc"},
                {"bits", args.bits},
                {"topk", args.topk},
                {"metric_topk", args.topk},
                {"recon_sample", args.recon_sample},
                {"train_only", args.train_only},
                {"skip_search", args.skip_search},
                {"maxtrain", args.maxtrain},
                {"vaq_validation_base", args.vaq_validation_base},
                {"vaq_validation_queries", args.vaq_validation_queries},
                {"data_root", args.data_root.string()},
                {"targets", args.targets},
                {"faiss_blas_threshold", faiss::distance_compute_blas_threshold},
                {"faiss_blas_query_bs", faiss::distance_compute_blas_query_bs},
                {"faiss_blas_database_bs", faiss::distance_compute_blas_database_bs},
        };
        if (args.config_path.has_value()) {
            run_meta["config_path"] = args.config_path->string();
        }
        if (args.epq_structure.has_value()) {
            run_meta["epq_structure"] = args.epq_structure->string();
        }
        const nlohmann::json dataset_meta = {
                {"name", ds.name},
                {"dim", ds.d},
                {"base_rows", ds.xb.rows()},
                {"base_rows_full", ds.xb.rows()},
                {"query_rows", ds.xq.rows()},
                {"query_rows_full", ds.xq.rows()},
                {"train_rows", ds.xt.rows()},
                {"train_rows_full", nt_full},
                {"gt_k", ds.gt_k},
        };
        epq::benchmark_metadata::print_common_benchmark_metadata(
                std::cout,
                epq::benchmark_metadata::build_common_benchmark_metadata(
                        run_meta,
                        dataset_meta,
                        args.threads,
                        effective_threads,
                        args.epq_config.has_value() ? &*args.epq_config : nullptr));

        for (const auto& target : args.targets) {
            auto index = make_target(args, ds, target);
            const auto structure_save_path =
                    structure_save_path_for_target(args, target);
            const auto summary =
                    run_benchmark(
                            *index,
                            ds,
                            args.topk,
                            args.mode,
                            args.recon_sample,
                            args.train_only,
                            args.skip_search,
                            structure_save_path);
            print_summary(*index, summary);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
