#include "epq/index_bapq.h"
#include "epq/benchmark_ivf_wrappers.h"
#include "epq/benchmark_metadata.h"
#include "epq/index_arepq.h"
#include "epq/index_dpopq.h"
#include "epq/index_epq.h"
#include "epq/index_ivf_codec.h"
#include "epq/index_vaq.h"
#include "epq/serialization_size.h"
#include "epq/structure.h"
#include "epq/structure_builder.h"
#include "epq/training_config.h"

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRefine.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/FaissException.h>
#include <faiss/utils/utils.h>
#include <omp.h>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using RowMatrixXf =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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

enum class MatrixFormat {
    kFvecs,
    kIvecs,
    kFbin,
    kIbin,
};

struct MatrixHeader {
    int dim = 0;
    size_t declared_count = 0;
    size_t available_count = 0;
    bool truncated = false;
};

struct MatrixFile {
    std::filesystem::path path;
    MatrixFormat format;
};

struct DatasetSpec {
    std::string name;
    MatrixFile train;
    MatrixFile base;
    MatrixFile query;
    MatrixFile gt;
};

struct Dataset {
    DatasetSpec spec;
    int d = 0;
    RowMatrixXf xt;
    RowMatrixXf xq;
    std::vector<int> gt;
    int gt_k = 0;
    size_t nb = 0;
    size_t train_rows_full = 0;
    size_t base_rows_full = 0;
    size_t query_rows_full = 0;
    size_t gt_rows_full = 0;
};

struct Args {
    std::string dataset;
    int bits = 0;
    int nlist = 0;
    int nprobe = 0;
    std::string target;
    std::filesystem::path data_root = "data";
    std::filesystem::path deep1b_root = "data/deep1b";
    std::optional<std::filesystem::path> epq_structure;
    std::optional<std::filesystem::path> config_path;
    std::optional<nlohmann::json> epq_config;
    std::optional<std::filesystem::path> json_out;
    int topk = 100;
    int metric_topk = 1000;
    int recon_sample = 200000;
    int threads = 0;
    int train_limit = 0;
    size_t base_limit = 0;
    int query_limit = 0;
    int base_batch_size = 100000;
    bool refine = false;
    float refine_k_factor = 1.0f;
    int coarse_kmeans_niter = 25;
    int coarse_kmeans_nredo = 1;
    std::optional<int> epq_transform_niter;
    std::optional<int> epq_kmeans_niter;
    std::optional<int> epq_transform_kmeans_niter;
    std::optional<int> vaq_subspaces;
    std::optional<int> vaq_min_bits;
    std::optional<int> vaq_max_bits;
};

struct CoarseSummary {
    double train_time = std::numeric_limits<double>::quiet_NaN();
    double add_time = std::numeric_limits<double>::quiet_NaN();
    double assign_time = std::numeric_limits<double>::quiet_NaN();
    double avg_candidates = std::numeric_limits<double>::quiet_NaN();
    double max_candidates = std::numeric_limits<double>::quiet_NaN();
    double candidate_hit_rate = std::numeric_limits<double>::quiet_NaN();
};

struct Summary {
    std::string name;
    int components = 0;
    int budget_bits = 0;
    nlohmann::json method_metadata = nlohmann::json::object();
    double train_time = 0.0;
    double structure_time = std::numeric_limits<double>::quiet_NaN();
    double preparation_time = std::numeric_limits<double>::quiet_NaN();
    double codebook_time = std::numeric_limits<double>::quiet_NaN();
    double add_time = 0.0;
    double encode_per_vector = 0.0;
    double rerank_time = std::numeric_limits<double>::quiet_NaN();
    double refine_time = std::numeric_limits<double>::quiet_NaN();
    double total_query_time = 0.0;
    double search_time = 0.0;
    double search_per_query = 0.0;
    double qps = 0.0;
    double recall1 = 0.0;
    double recall10 = 0.0;
    double recall100 = 0.0;
    double recall1000 = std::numeric_limits<double>::quiet_NaN();
    double overlap1000 = std::numeric_limits<double>::quiet_NaN();
    double reconstruction_error = std::numeric_limits<double>::quiet_NaN();
    double index_size_mib = std::numeric_limits<double>::quiet_NaN();
    nlohmann::json diagnostics = nlohmann::json::object();
    CoarseSummary coarse;
};

struct SearchMetrics {
    double rerank_time = std::numeric_limits<double>::quiet_NaN();
    double refine_time = std::numeric_limits<double>::quiet_NaN();
    double total_query_time = 0.0;
    double search_time = 0.0;
    double search_per_query = 0.0;
    double qps = 0.0;
    double recall1 = 0.0;
    double recall10 = 0.0;
    double recall100 = 0.0;
    double recall1000 = std::numeric_limits<double>::quiet_NaN();
    double overlap1000 = std::numeric_limits<double>::quiet_NaN();
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void reject_retired_query_weighted_training_config(
        const nlohmann::json& config) {
    const auto section = config.find("index");
    if (section == config.end() || !section->is_object()) {
        return;
    }
    constexpr std::string_view prefix = "ivf_query_weighted_sampling";
    for (auto it = section->begin(); it != section->end(); ++it) {
        if (std::string_view(it.key()).starts_with(prefix)) {
            fail(
                    "unsupported joint_benchmark config key 'index." + it.key() +
                    "': evaluation-query-weighted training has been removed");
        }
    }
}

template <typename T>
MatrixHeader inspect_xvecs_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail("failed to open " + path.string());
    }
    int dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(int));
    if (!in || dim <= 0) {
        fail("invalid xvecs header in " + path.string());
    }
    const size_t bytes = std::filesystem::file_size(path);
    const size_t record_size = static_cast<size_t>(dim + 1) * sizeof(T);
    const size_t available_count = bytes / record_size;
    return MatrixHeader{
            .dim = dim,
            .declared_count = available_count,
            .available_count = available_count,
            .truncated = bytes % record_size != 0,
    };
}

template <typename T>
MatrixHeader inspect_bin_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail("failed to open " + path.string());
    }
    uint32_t count = 0;
    uint32_t dim = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
    if (!in || dim == 0) {
        fail("invalid bin header in " + path.string());
    }
    const size_t bytes = std::filesystem::file_size(path);
    if (bytes < 2 * sizeof(uint32_t)) {
        fail("bin file too small: " + path.string());
    }
    const size_t payload_bytes = bytes - 2 * sizeof(uint32_t);
    const size_t row_bytes = static_cast<size_t>(dim) * sizeof(T);
    const size_t available_count = payload_bytes / row_bytes;
    return MatrixHeader{
            .dim = static_cast<int>(dim),
            .declared_count = static_cast<size_t>(count),
            .available_count = std::min(static_cast<size_t>(count), available_count),
            .truncated = available_count < static_cast<size_t>(count) ||
                    payload_bytes % row_bytes != 0,
    };
}

template <typename T>
size_t resolve_limit(
        const MatrixHeader& header,
        size_t limit,
        const std::filesystem::path& path) {
    const size_t available = header.available_count;
    if (available == 0) {
        fail("matrix file contains no complete rows: " + path.string());
    }
    if (limit == 0) {
        return available;
    }
    if (limit > available) {
        fail(
                "requested " + std::to_string(limit) + " rows from " + path.string() +
                " but only " + std::to_string(available) + " complete rows are available");
    }
    return limit;
}

template <typename T>
std::vector<T> load_xvecs_flat(
        const std::filesystem::path& path,
        size_t limit,
        int* out_dim,
        size_t* out_count) {
    const auto header = inspect_xvecs_file<T>(path);
    const size_t count = resolve_limit<T>(header, limit, path);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail("failed to open " + path.string());
    }
    std::vector<T> values(count * static_cast<size_t>(header.dim));
    std::vector<T> row(static_cast<size_t>(header.dim));
    for (size_t i = 0; i < count; ++i) {
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
        *out_count = count;
    }
    return values;
}

template <typename T>
std::vector<T> load_bin_flat(
        const std::filesystem::path& path,
        size_t limit,
        int* out_dim,
        size_t* out_count) {
    const auto header = inspect_bin_file<T>(path);
    const size_t count = resolve_limit<T>(header, limit, path);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail("failed to open " + path.string());
    }
    uint32_t declared_count = 0;
    uint32_t dim = 0;
    in.read(reinterpret_cast<char*>(&declared_count), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
    std::vector<T> values(count * static_cast<size_t>(header.dim));
    in.read(reinterpret_cast<char*>(values.data()), sizeof(T) * values.size());
    if (!in) {
        fail("short read while reading " + path.string());
    }
    if (out_dim != nullptr) {
        *out_dim = header.dim;
    }
    if (out_count != nullptr) {
        *out_count = count;
    }
    return values;
}

RowMatrixXf load_float_matrix(
        const MatrixFile& file,
        size_t limit,
        size_t* out_count = nullptr) {
    int dim = 0;
    size_t count = 0;
    std::vector<float> values;
    switch (file.format) {
        case MatrixFormat::kFvecs:
            values = load_xvecs_flat<float>(file.path, limit, &dim, &count);
            break;
        case MatrixFormat::kFbin:
            values = load_bin_flat<float>(file.path, limit, &dim, &count);
            break;
        default:
            fail("expected float matrix format for " + file.path.string());
    }
    if (out_count != nullptr) {
        *out_count = count;
    }
    Eigen::Map<RowMatrixXf> mapped(values.data(), static_cast<Eigen::Index>(count), dim);
    return mapped;
}

std::vector<int> load_int_matrix_flat(
        const MatrixFile& file,
        size_t limit,
        int* out_dim,
        size_t* out_count) {
    switch (file.format) {
        case MatrixFormat::kIvecs:
            return load_xvecs_flat<int>(file.path, limit, out_dim, out_count);
        case MatrixFormat::kIbin:
            return load_bin_flat<int>(file.path, limit, out_dim, out_count);
        default:
            fail("expected int matrix format for " + file.path.string());
    }
}

template <typename Visitor>
void stream_xvecs_float(
        const std::filesystem::path& path,
        size_t limit,
        int batch_size,
        Visitor&& visitor) {
    const auto header = inspect_xvecs_file<float>(path);
    const size_t count = resolve_limit<float>(header, limit, path);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail("failed to open " + path.string());
    }

    size_t offset = 0;
    while (offset < count) {
        const size_t rows =
                std::min<size_t>(static_cast<size_t>(batch_size), count - offset);
        RowMatrixXf batch(static_cast<Eigen::Index>(rows), header.dim);
        for (size_t i = 0; i < rows; ++i) {
            int dim = 0;
            in.read(reinterpret_cast<char*>(&dim), sizeof(int));
            if (!in || dim != header.dim) {
                fail("invalid record header while streaming " + path.string());
            }
            in.read(
                    reinterpret_cast<char*>(batch.row(static_cast<Eigen::Index>(i)).data()),
                    sizeof(float) * static_cast<size_t>(header.dim));
            if (!in) {
                fail("short read while streaming " + path.string());
            }
        }
        visitor(offset, batch);
        offset += rows;
    }
}

template <typename Visitor>
void stream_bin_float(
        const std::filesystem::path& path,
        size_t limit,
        int batch_size,
        Visitor&& visitor) {
    const auto header = inspect_bin_file<float>(path);
    const size_t count = resolve_limit<float>(header, limit, path);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail("failed to open " + path.string());
    }

    uint32_t declared_count = 0;
    uint32_t dim = 0;
    in.read(reinterpret_cast<char*>(&declared_count), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
    size_t offset = 0;
    while (offset < count) {
        const size_t rows =
                std::min<size_t>(static_cast<size_t>(batch_size), count - offset);
        RowMatrixXf batch(static_cast<Eigen::Index>(rows), header.dim);
        in.read(
                reinterpret_cast<char*>(batch.data()),
                sizeof(float) * rows * static_cast<size_t>(header.dim));
        if (!in) {
            fail("short read while streaming " + path.string());
        }
        visitor(offset, batch);
        offset += rows;
    }
}

template <typename Visitor>
void stream_float_matrix(
        const MatrixFile& file,
        size_t limit,
        int batch_size,
        Visitor&& visitor) {
    switch (file.format) {
        case MatrixFormat::kFvecs:
            stream_xvecs_float(file.path, limit, batch_size, std::forward<Visitor>(visitor));
            return;
        case MatrixFormat::kFbin:
            stream_bin_float(file.path, limit, batch_size, std::forward<Visitor>(visitor));
            return;
        default:
            fail("expected float matrix format while streaming " + file.path.string());
    }
}

class BaseVectorReader {
   public:
    explicit BaseVectorReader(const MatrixFile& file)
            : file_(file), in_(file.path, std::ios::binary) {
        if (!in_) {
            fail("failed to open base vector file for reconstruction: " + file.path.string());
        }
        switch (file_.format) {
            case MatrixFormat::kFvecs: {
                const auto header = inspect_xvecs_file<float>(file_.path);
                dim_ = header.dim;
                row_bytes_ = static_cast<std::streamoff>(sizeof(int)) +
                        static_cast<std::streamoff>(sizeof(float) * dim_);
                data_offset_ = 0;
                count_ = header.available_count;
                break;
            }
            case MatrixFormat::kFbin: {
                const auto header = inspect_bin_file<float>(file_.path);
                dim_ = header.dim;
                row_bytes_ = static_cast<std::streamoff>(sizeof(float) * dim_);
                data_offset_ = static_cast<std::streamoff>(2 * sizeof(uint32_t));
                count_ = header.available_count;
                break;
            }
            default:
                fail("reconstruction requires float base vectors: " + file_.path.string());
        }
    }

    int dim() const {
        return dim_;
    }

    void read_row(faiss::idx_t id, float* out) {
        if (id < 0 || static_cast<size_t>(id) >= count_) {
            fail("reconstruction base vector id out of range");
        }
        const auto pos =
                data_offset_ + static_cast<std::streamoff>(id) * row_bytes_;
        in_.seekg(pos);
        if (!in_) {
            fail("failed to seek base vector file during reconstruction");
        }
        if (file_.format == MatrixFormat::kFvecs) {
            int row_dim = 0;
            in_.read(reinterpret_cast<char*>(&row_dim), sizeof(int));
            if (!in_ || row_dim != dim_) {
                fail("failed to read fvecs row header during reconstruction");
            }
        }
        in_.read(reinterpret_cast<char*>(out), sizeof(float) * static_cast<size_t>(dim_));
        if (!in_) {
            fail("failed to read base vector row during reconstruction");
        }
    }

   private:
    MatrixFile file_;
    std::ifstream in_;
    int dim_ = 0;
    size_t count_ = 0;
    std::streamoff row_bytes_ = 0;
    std::streamoff data_offset_ = 0;
};

std::vector<faiss::idx_t> sample_ids(
        faiss::idx_t n,
        int sample,
        uint32_t seed = 123) {
    if (sample <= 0 || sample >= n) {
        std::vector<faiss::idx_t> ids(static_cast<size_t>(n));
        std::iota(ids.begin(), ids.end(), 0);
        return ids;
    }
    std::vector<faiss::idx_t> ids(static_cast<size_t>(n));
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(ids.begin(), ids.end(), rng);
    ids.resize(static_cast<size_t>(sample));
    std::sort(ids.begin(), ids.end());
    return ids;
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

double exact_position_match_at_k(
        const std::vector<faiss::idx_t>& a,
        const std::vector<faiss::idx_t>& b,
        int nq,
        int topk,
        int k) {
    const int kk = std::min(k, topk);
    size_t matched = 0;
    for (int i = 0; i < nq; ++i) {
        const size_t row = static_cast<size_t>(i) * static_cast<size_t>(topk);
        for (int j = 0; j < kk; ++j) {
            matched += a[row + static_cast<size_t>(j)] ==
                    b[row + static_cast<size_t>(j)];
        }
    }
    return static_cast<double>(matched) /
            static_cast<double>(static_cast<size_t>(nq) * static_cast<size_t>(kk));
}

double exact_query_match_at_k(
        const std::vector<faiss::idx_t>& a,
        const std::vector<faiss::idx_t>& b,
        int nq,
        int topk,
        int k) {
    const int kk = std::min(k, topk);
    int matched_queries = 0;
    for (int i = 0; i < nq; ++i) {
        const size_t row = static_cast<size_t>(i) * static_cast<size_t>(topk);
        bool same = true;
        for (int j = 0; j < kk; ++j) {
            if (a[row + static_cast<size_t>(j)] !=
                b[row + static_cast<size_t>(j)]) {
                same = false;
                break;
            }
        }
        matched_queries += same ? 1 : 0;
    }
    return static_cast<double>(matched_queries) / static_cast<double>(nq);
}

std::string format_name(MatrixFormat format) {
    switch (format) {
        case MatrixFormat::kFvecs:
            return "fvecs";
        case MatrixFormat::kIvecs:
            return "ivecs";
        case MatrixFormat::kFbin:
            return "fbin";
        case MatrixFormat::kIbin:
            return "ibin";
    }
    return "unknown";
}

void print_matrix_status(const std::string& label, const MatrixFile& file) {
    MatrixHeader header;
    if (file.format == MatrixFormat::kFvecs) {
        header = inspect_xvecs_file<float>(file.path);
    } else if (file.format == MatrixFormat::kIvecs) {
        header = inspect_xvecs_file<int>(file.path);
    } else if (file.format == MatrixFormat::kFbin) {
        header = inspect_bin_file<float>(file.path);
    } else {
        header = inspect_bin_file<int>(file.path);
    }
    std::cout << label << "=" << file.path
              << " format=" << format_name(file.format)
              << " dim=" << header.dim
              << " available=" << header.available_count;
    if (file.format == MatrixFormat::kFbin || file.format == MatrixFormat::kIbin) {
        std::cout << " declared=" << header.declared_count;
    }
    if (header.truncated) {
        std::cout << " truncated=true";
    }
    std::cout << '\n';
}

DatasetSpec resolve_dataset_spec(const Args& args) {
    if (args.dataset == "sift1M") {
        const auto root = args.data_root / "sift1M";
        return DatasetSpec{
                .name = args.dataset,
                .train = {root / "sift_learn.fvecs", MatrixFormat::kFvecs},
                .base = {root / "sift_base.fvecs", MatrixFormat::kFvecs},
                .query = {root / "sift_query.fvecs", MatrixFormat::kFvecs},
                .gt = {root / "sift_groundtruth.ivecs", MatrixFormat::kIvecs},
        };
    }
    if (args.dataset == "gist1M") {
        const auto root = args.data_root / "gist1M";
        return DatasetSpec{
                .name = args.dataset,
                .train = {root / "gist_learn.fvecs", MatrixFormat::kFvecs},
                .base = {root / "gist_base.fvecs", MatrixFormat::kFvecs},
                .query = {root / "gist_query.fvecs", MatrixFormat::kFvecs},
                .gt = {root / "gist_groundtruth.ivecs", MatrixFormat::kIvecs},
        };
    }
    if (args.dataset == "deep10M") {
        const auto root = args.data_root / "deep1b";
        return DatasetSpec{
                .name = args.dataset,
                .train = {root / "learn.fvecs", MatrixFormat::kFvecs},
                .base = {root / "base.fvecs", MatrixFormat::kFvecs},
                .query = {root / "deep1B_queries.fvecs", MatrixFormat::kFvecs},
                .gt = {root / "deep10M_groundtruth.ivecs", MatrixFormat::kIvecs},
        };
    }
    if (args.dataset == "deep1b") {
        return DatasetSpec{
                .name = args.dataset,
                .train = {args.deep1b_root / "learn.350M.fbin", MatrixFormat::kFbin},
                .base = {args.deep1b_root / "base.1B.fbin", MatrixFormat::kFbin},
                .query = {args.deep1b_root / "query.public.10K.fbin", MatrixFormat::kFbin},
                .gt = {args.deep1b_root / "groundtruth.public.10K.ibin", MatrixFormat::kIbin},
        };
    }
    fail("unsupported dataset: " + args.dataset);
}

Dataset load_dataset(const Args& args) {
    Dataset ds;
    ds.spec = resolve_dataset_spec(args);

    print_matrix_status("train_file", ds.spec.train);
    print_matrix_status("base_file", ds.spec.base);
    print_matrix_status("query_file", ds.spec.query);
    print_matrix_status("gt_file", ds.spec.gt);

    size_t train_count = 0;
    size_t query_count = 0;
    const size_t train_limit =
            args.train_limit > 0 ? static_cast<size_t>(args.train_limit) : 0;
    const size_t query_limit =
            args.query_limit > 0 ? static_cast<size_t>(args.query_limit) : 0;
    const MatrixHeader train_header = ds.spec.train.format == MatrixFormat::kFvecs
            ? inspect_xvecs_file<float>(ds.spec.train.path)
            : inspect_bin_file<float>(ds.spec.train.path);
    const MatrixHeader query_header = ds.spec.query.format == MatrixFormat::kFvecs
            ? inspect_xvecs_file<float>(ds.spec.query.path)
            : inspect_bin_file<float>(ds.spec.query.path);

    ds.xt = load_float_matrix(ds.spec.train, train_limit, &train_count);
    ds.xq = load_float_matrix(ds.spec.query, query_limit, &query_count);
    ds.train_rows_full = train_header.available_count;
    ds.query_rows_full = query_header.available_count;
    ds.d = static_cast<int>(ds.xt.cols());
    if (ds.xq.cols() != ds.d) {
        fail("train/query dimension mismatch");
    }

    size_t gt_count = 0;
    ds.gt = load_int_matrix_flat(ds.spec.gt, query_count, &ds.gt_k, &gt_count);
    const MatrixHeader gt_header = ds.spec.gt.format == MatrixFormat::kIvecs
            ? inspect_xvecs_file<int>(ds.spec.gt.path)
            : inspect_bin_file<int>(ds.spec.gt.path);
    ds.gt_rows_full = gt_header.available_count;
    if (gt_count != static_cast<size_t>(ds.xq.rows())) {
        fail("groundtruth/query size mismatch");
    }

    MatrixHeader base_header;
    if (ds.spec.base.format == MatrixFormat::kFvecs) {
        base_header = inspect_xvecs_file<float>(ds.spec.base.path);
    } else {
        base_header = inspect_bin_file<float>(ds.spec.base.path);
    }
    ds.base_rows_full = base_header.available_count;
    ds.nb = resolve_limit<float>(base_header, args.base_limit, ds.spec.base.path);

    return ds;
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

std::filesystem::path default_epq_structure_path(
        const std::filesystem::path& data_root,
        std::string_view dataset,
        int bits,
        int nlist,
        std::string_view config_fingerprint = {}) {
    const std::string prefix(dataset);
    std::string suffix;
    if (!config_fingerprint.empty()) {
        suffix = "_cfg" + std::string(config_fingerprint);
    }
    return data_root / "structures" /
            (prefix + "_" + std::to_string(bits) + "B_nlist" +
             std::to_string(nlist) + suffix + "_joint_ivf_epq_structure.json");
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
        if (!require_existing || std::filesystem::exists(*args.epq_structure)) {
            return args.epq_structure;
        }
        return std::nullopt;
    }
    if (!should_auto_reuse_epq_structure(args)) {
        return std::nullopt;
    }
    const auto fingerprint = epq_config_fingerprint(args);
    const auto candidate = default_epq_structure_path(
            args.data_root,
            args.dataset,
            args.bits,
            args.nlist,
            fingerprint.value_or(""));
    if (!require_existing || std::filesystem::exists(candidate)) {
        return candidate;
    }
    return std::nullopt;
}

bool is_arepq_target(std::string_view target) {
    return target == "arepq" || target == "arepq_fixed";
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

Args arepq_main_args(const Args& args) {
    const auto tail = resolve_arepq_tail_config(args);
    Args main_args = args;
    main_args.bits = args.bits - tail.tail_bits * tail.tail_stages;
    return main_args;
}

std::optional<std::filesystem::path> resolve_structure_path_for_target(
        const Args& args,
        bool require_existing) {
    if (args.target == "epq") {
        return resolve_epq_structure_path(args, require_existing);
    }
    if (is_arepq_target(args.target)) {
        Args main_args = arepq_main_args(args);
        if (main_args.bits <= 0) {
            return std::nullopt;
        }
        return resolve_epq_structure_path(main_args, require_existing);
    }
    return std::nullopt;
}

std::shared_ptr<epq::StructureBuilder> make_epq_builder(
        const Args& args,
        bool require_existing = false) {
    auto structure_path = resolve_epq_structure_path(args, true);
    if (structure_path.has_value()) {
        auto structure = epq::Structure::load_json(structure_path->string());
        return std::make_shared<epq::FixedStructureBuilder>(std::move(structure));
    }
    if (require_existing) {
        fail(
                "target 'arepq_fixed' requires an existing --epq-structure or "
                "matching reusable structure");
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

void configure_arepq_codec(
        epq::IndexAREPQ& codec,
        const Args& args,
        const std::optional<nlohmann::json>& config) {
    if (config.has_value()) {
        epq::apply_index_training_config(codec.main_index(), *config);
    }
    if (args.epq_transform_niter.has_value()) {
        codec.main_index().transform_niter = *args.epq_transform_niter;
    }
    if (args.epq_kmeans_niter.has_value()) {
        codec.main_index().kmeans_niter = *args.epq_kmeans_niter;
    }
    if (args.epq_transform_kmeans_niter.has_value()) {
        codec.main_index().transform_kmeans_niter =
                *args.epq_transform_kmeans_niter;
    }
    codec.icm_iters = std::max(0, getenv_int_or("EPQ_AREPQ_ICM_ITERS", 2));
    codec.final_main_reassign =
            getenv_int_or("EPQ_AREPQ_FINAL_MAIN_REASSIGN", 0) != 0;
    codec.skip_stable_tail_reassign =
            getenv_int_or("EPQ_AREPQ_SKIP_STABLE_TAIL_REASSIGN", 1) != 0;
    const int legacy_tail_refine_iters =
            getenv_int_or("EPQ_AREPQ_TAIL_REFINE_ITERS", 1);
    codec.tail_alt_iters = std::max(
            0,
            getenv_int_or(
                    "EPQ_AREPQ_TAIL_ALT_ITERS",
                    legacy_tail_refine_iters));
    codec.tail_alt_update_weight = std::clamp(
            getenv_float_or("EPQ_AREPQ_TAIL_ALT_UPDATE_WEIGHT", 0.5f),
            0.0f,
            1.0f);
    codec.tail_kmeans_niter =
            std::max(1, getenv_int_or("EPQ_AREPQ_TAIL_KMEANS_NITER", 25));
    codec.tail_kmeans_nredo =
            std::max(1, getenv_int_or("EPQ_AREPQ_TAIL_KMEANS_NREDO", 1));
    codec.tail_beam_candidates = std::max(
            1,
            get_config_int_or_env(
                    config,
                    "arepq",
                    "tail_beam_candidates",
                    "EPQ_AREPQ_TAIL_BEAM",
                    1));
    codec.add_batch_rows =
            std::max(1, getenv_int_or("EPQ_AREPQ_ADD_BATCH_ROWS", 100000));
    codec.search_query_batch =
            std::max(1, getenv_int_or("EPQ_AREPQ_SEARCH_QUERY_BATCH", 4));
    codec.search_db_chunk =
            std::max(1024, getenv_int_or("EPQ_AREPQ_SEARCH_DB_CHUNK", 65536));
}

nlohmann::json summarize_arepq_method(const epq::IndexAREPQ& index) {
    const auto tail_memory = index.tail_memory_stats();
    nlohmann::json meta = {
            {"family", "arepq"},
            {"impl", "cpp"},
            {"total_bits", index.total_bits},
            {"main_bits", index.main_bits},
            {"tail_bits", index.tail_bits},
            {"tail_stages", index.tail_stages},
            {"tail_ksub", index.tail_ksub},
            {"tail_type", "full_dim_transformed_residual"},
            {"assignment", "additive_icm"},
            {"tail_training", "residual_tail_bcd"},
            {"tail_update", "relaxed_centroid_mean"},
            {"tail_update_acceptance", "monotone_train_mse"},
            {"search", "ivf_additive_adc"},
            {"icm_iters", index.icm_iters},
            {"final_main_reassign", index.final_main_reassign},
            {"skip_stable_tail_reassign", index.skip_stable_tail_reassign},
            {"tail_alt_iters", index.tail_alt_iters},
            {"tail_alt_update_weight", index.tail_alt_update_weight},
            {"tail_alt_initial_mse", index.tail_alt_initial_mse()},
            {"tail_alt_best_mse", index.tail_alt_best_mse()},
            {"tail_alt_final_mse", index.tail_alt_final_mse()},
            {"tail_kmeans_niter", index.tail_kmeans_niter},
            {"tail_kmeans_nredo", index.tail_kmeans_nredo},
            {"tail_beam_candidates", index.tail_beam_candidates},
            {"add_batch_rows", index.add_batch_rows},
            {"search_query_batch", index.search_query_batch},
            {"search_db_chunk", index.search_db_chunk},
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
            index.main_index(),
            "arepq_main");
    return meta;
}

struct AssignedLists {
    int nprobe = 0;
    std::vector<faiss::idx_t> list_ids;
    std::vector<float> centroid_distances;
};

AssignedLists build_query_assignments(
        const faiss::IndexIVF& ivf,
        const Dataset& ds,
        const Args& args,
        CoarseSummary& summary) {
    AssignedLists assigned;
    assigned.nprobe = args.nprobe;
    assigned.list_ids.resize(
            static_cast<size_t>(ds.xq.rows()) * static_cast<size_t>(args.nprobe));
    assigned.centroid_distances.resize(
            static_cast<size_t>(ds.xq.rows()) * static_cast<size_t>(args.nprobe));
    const auto t0 = std::chrono::steady_clock::now();
    ivf.quantizer->search(
            ds.xq.rows(),
            ds.xq.data(),
            args.nprobe,
            assigned.centroid_distances.data(),
            assigned.list_ids.data());
    summary.assign_time = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();
    return assigned;
}

void compute_candidate_stats(
        const faiss::IndexIVF& ivf,
        const AssignedLists& assigned,
        const Dataset& ds,
        CoarseSummary& summary) {
    size_t total_candidates = 0;
    size_t max_candidates = 0;
    size_t hit_queries = 0;
    for (int qi = 0; qi < ds.xq.rows(); ++qi) {
        size_t count = 0;
        bool hit = false;
        const auto truth = static_cast<faiss::idx_t>(
                ds.gt[static_cast<size_t>(qi) * static_cast<size_t>(ds.gt_k)]);
        for (int p = 0; p < assigned.nprobe; ++p) {
            const auto list_no =
                    assigned.list_ids[static_cast<size_t>(qi) * assigned.nprobe + p];
            if (list_no < 0) {
                continue;
            }
            const size_t list_size = ivf.invlists->list_size(
                    static_cast<size_t>(list_no));
            count += list_size;
            if (!hit) {
                const faiss::InvertedLists::ScopedIds ids(
                        ivf.invlists,
                        static_cast<size_t>(list_no));
                for (size_t i = 0; i < list_size; ++i) {
                    if (ids[i] == truth) {
                        hit = true;
                        break;
                    }
                }
            }
        }
        total_candidates += count;
        max_candidates = std::max(max_candidates, count);
        hit_queries += hit ? 1 : 0;
    }
    summary.avg_candidates = static_cast<double>(total_candidates) / ds.xq.rows();
    summary.max_candidates = static_cast<double>(max_candidates);
    summary.candidate_hit_rate = static_cast<double>(hit_queries) / ds.xq.rows();
}

bool compare_epq_ivf_three_way() {
    static const bool enabled = [] {
        const char* value = std::getenv("EPQ_COMPARE_IVF_THREE_WAY");
        return value != nullptr && std::string_view(value) == "1";
    }();
    return enabled;
}

bool compare_epq_ivf_scanners() {
    static const bool enabled = [] {
        const char* value = std::getenv("EPQ_COMPARE_IVF_SCANNERS");
        return value != nullptr && std::string_view(value) == "1";
    }();
    return enabled;
}

struct EpqIvfTrainSummary {
    double total_time = 0.0;
    double coarse_train_time = std::numeric_limits<double>::quiet_NaN();
};

template <typename Codec>
EpqIvfTrainSummary train_epq_ivf_index(
        epq::IndexIVFCodec<Codec>& index,
        const Dataset& ds) {
    EpqIvfTrainSummary summary;
    const auto total_t0 = std::chrono::steady_clock::now();

    const auto coarse_t0 = std::chrono::steady_clock::now();
    index.train_q1(ds.xt.rows(), ds.xt.data(), index.verbose, index.metric_type);
    summary.coarse_train_time =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - coarse_t0)
                    .count();

    size_t train_n_size = static_cast<size_t>(ds.xt.rows());
    faiss::idx_t max_nt = index.train_encoder_num_vectors();
    if (max_nt <= 0) {
        max_nt = static_cast<faiss::idx_t>((size_t)1 << 35);
    }
    faiss::TransformedVectors tv(
            ds.xt.data(),
            faiss::fvecs_maybe_subsample(
                    index.d,
                    &train_n_size,
                    max_nt,
                    ds.xt.data(),
                    index.verbose));
    const faiss::idx_t train_n = static_cast<faiss::idx_t>(train_n_size);

    std::vector<faiss::idx_t> assign(static_cast<size_t>(train_n));
    index.quantizer->assign(train_n, tv.x, assign.data());
    std::vector<float> residuals(
            static_cast<size_t>(train_n) * static_cast<size_t>(index.d));
    index.quantizer->compute_residual_n(
            train_n, tv.x, residuals.data(), assign.data());
    index.train_encoder(train_n, residuals.data(), nullptr);

    index.is_trained = true;
    summary.total_time =
            std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - total_t0)
                    .count();
    return summary;
}

struct BuiltIndex {
    std::unique_ptr<faiss::Index> index;
    faiss::Index* storage_index = nullptr;
    faiss::IndexIVF* ivf = nullptr;
    std::string name;
};

BuiltIndex build_index(const Args& args, const Dataset& ds) {
    const int M = args.bits / 8;
    if (args.bits % 8 != 0) {
        fail("joint_benchmark MVP currently requires bits divisible by 8");
    }
    if (M <= 0) {
        fail("invalid bits for PQ-style targets");
    }

    if (args.target == "pq") {
        auto* quantizer = new faiss::IndexFlatL2(ds.d);
        auto base = std::make_unique<epq::TimedIndexIVFPQ>(
                quantizer, ds.d, args.nlist, M, 8, faiss::METRIC_L2);
        base->own_fields = true;
        base->cp.niter = args.coarse_kmeans_niter;
        base->cp.nredo = args.coarse_kmeans_nredo;
        BuiltIndex out;
        out.storage_index = base.get();
        out.ivf = base.get();
        out.name = "IVF+PQ";
        out.index = std::move(base);
        return out;
    }

    if (args.target == "opq") {
        const int d2 = ((ds.d + M - 1) / M) * M;
        auto* quantizer = new faiss::IndexFlatL2(d2);
        auto* ivfpq = new epq::TimedIndexIVFPQ(
                quantizer, d2, args.nlist, M, 8, faiss::METRIC_L2);
        ivfpq->own_fields = true;
        ivfpq->cp.niter = args.coarse_kmeans_niter;
        ivfpq->cp.nredo = args.coarse_kmeans_nredo;
        auto* opq = new faiss::OPQMatrix(ds.d, M, d2);
        auto pre = std::make_unique<epq::TimedIndexPreTransform>(opq, ivfpq);
        pre->own_fields = true;
        BuiltIndex out;
        out.storage_index = pre.get();
        out.ivf = ivfpq;
        out.name = "IVF+OPQ";
        out.index = std::move(pre);
        return out;
    }

    if (args.target == "rq") {
        if (M <= 1) {
            fail("IVF+RQ requires at least 16 total bits");
        }
        auto* quantizer = new faiss::IndexFlatL2(ds.d);
        auto base = std::make_unique<epq::TimedIndexIVFResidualQuantizer>(
                quantizer,
                static_cast<size_t>(ds.d),
                static_cast<size_t>(args.nlist),
                static_cast<size_t>(M - 1),
                static_cast<size_t>(8),
                faiss::METRIC_L2,
                faiss::AdditiveQuantizer::ST_norm_qint8);
        base->own_fields = true;
        base->cp.niter = args.coarse_kmeans_niter;
        base->cp.nredo = args.coarse_kmeans_nredo;
        base->rq.max_beam_size = 8;
        BuiltIndex out;
        out.storage_index = base.get();
        out.ivf = base.get();
        out.name = "IVF+RQ";
        out.index = std::move(base);
        return out;
    }

    if (args.target == "lsq") {
        if (M <= 1) {
            fail("IVF+LSQ requires at least 16 total bits");
        }
        auto* quantizer = new faiss::IndexFlatL2(ds.d);
        auto base = std::make_unique<epq::TimedIndexIVFLocalSearchQuantizer>(
                quantizer,
                static_cast<size_t>(ds.d),
                static_cast<size_t>(args.nlist),
                static_cast<size_t>(M - 1),
                static_cast<size_t>(8),
                faiss::METRIC_L2,
                faiss::AdditiveQuantizer::ST_norm_qint8);
        base->own_fields = true;
        base->cp.niter = args.coarse_kmeans_niter;
        base->cp.nredo = args.coarse_kmeans_nredo;
        BuiltIndex out;
        out.storage_index = base.get();
        out.ivf = base.get();
        out.name = "IVF+LSQ";
        out.index = std::move(base);
        return out;
    }

    if (args.target == "rabitq") {
        auto* quantizer = new faiss::IndexFlatL2(ds.d);
        auto base = std::make_unique<epq::TimedIndexIVFRaBitQ>(
                quantizer,
                static_cast<size_t>(ds.d),
                static_cast<size_t>(args.nlist),
                faiss::METRIC_L2,
                true,
                epq::resolve_rabitq_nb_bits(ds.d, args.bits));
        base->own_fields = true;
        base->cp.niter = args.coarse_kmeans_niter;
        base->cp.nredo = args.coarse_kmeans_nredo;
        BuiltIndex out;
        out.storage_index = base.get();
        out.ivf = base.get();
        out.name = "IVF+RaBitQ";
        out.index = std::move(base);
        return out;
    }

    if (args.target == "dpopq" || args.target == "dp_opq" ||
        args.target == "dp-opq") {
        auto codec = std::make_unique<epq::IndexDPOPQ>(ds.d, args.bits);
        codec->kmeans_niter = getenv_int_or("EPQ_DPOPQ_KMEANS_NITER", 25);
        codec->kmeans_nredo = getenv_int_or("EPQ_DPOPQ_KMEANS_NREDO", 1);
        codec->dp_max_units = getenv_int_or("EPQ_DPOPQ_DP_MAX_UNITS", 0);
        codec->block_alignment = getenv_int_or("EPQ_DPOPQ_BLOCK_ALIGN", 0) != 0;
        auto* quantizer = new faiss::IndexFlatL2(ds.d);
        auto base = std::make_unique<epq::IndexIVFCodec<epq::IndexDPOPQ>>(
                std::move(codec), quantizer, args.nlist, "IVF+DPOPQ");
        base->cp.niter = args.coarse_kmeans_niter;
        base->cp.nredo = args.coarse_kmeans_nredo;
        BuiltIndex out;
        out.storage_index = base.get();
        out.ivf = base.get();
        out.name = "IVF+DPOPQ";
        out.index = std::move(base);
        return out;
    }

    if (args.target == "vaq") {
        auto codec = std::make_unique<epq::IndexVAQ>(
                ds.d,
                args.bits,
                args.vaq_subspaces.value_or(0),
                args.vaq_min_bits.value_or(1),
                args.vaq_max_bits.value_or(8));
        auto* quantizer = new faiss::IndexFlatL2(ds.d);
        auto base = std::make_unique<epq::IndexIVFCodec<epq::IndexVAQ>>(
                std::move(codec), quantizer, args.nlist, "IVF+VAQ");
        base->cp.niter = args.coarse_kmeans_niter;
        base->cp.nredo = args.coarse_kmeans_nredo;
        BuiltIndex out;
        out.storage_index = base.get();
        out.ivf = base.get();
        out.name = "IVF+VAQ";
        out.index = std::move(base);
        return out;
    }

    if (args.target == "epq") {
        auto codec = std::make_unique<epq::IndexEPQ>(
                ds.d, args.bits, make_epq_builder(args));
        if (args.epq_config.has_value()) {
            epq::apply_index_training_config(*codec, *args.epq_config);
        }
        if (args.epq_transform_niter.has_value()) {
            codec->transform_niter = *args.epq_transform_niter;
        }
        if (args.epq_kmeans_niter.has_value()) {
            codec->kmeans_niter = *args.epq_kmeans_niter;
        }
        if (args.epq_transform_kmeans_niter.has_value()) {
            codec->transform_kmeans_niter = *args.epq_transform_kmeans_niter;
        }
        auto* quantizer = new faiss::IndexFlatL2(ds.d);
        auto base = std::make_unique<epq::IndexIVFCodec<epq::IndexEPQ>>(
                std::move(codec), quantizer, args.nlist, "IVF+EPQ");
        base->cp.niter = args.coarse_kmeans_niter;
        base->cp.nredo = args.coarse_kmeans_nredo;
        BuiltIndex out;
        out.storage_index = base.get();
        out.ivf = base.get();
        out.name = "IVF+EPQ";
        out.index = std::move(base);
        return out;
    }

    if (is_arepq_target(args.target)) {
        const auto tail = resolve_arepq_tail_config(args);
        Args main_args = arepq_main_args(args);
        if (main_args.bits <= 0) {
            fail("AREPQ requires total bits larger than tail_bits * tail_stages");
        }
        auto codec = std::make_unique<epq::IndexAREPQ>(
                ds.d,
                args.bits,
                tail.tail_bits,
                tail.tail_stages,
                make_epq_builder(main_args, args.target == "arepq_fixed"));
        configure_arepq_codec(*codec, args, args.epq_config);
        auto* quantizer = new faiss::IndexFlatL2(ds.d);
        auto base = std::make_unique<epq::IndexIVFCodec<epq::IndexAREPQ>>(
                std::move(codec), quantizer, args.nlist, "IVF+AREPQ");
        base->cp.niter = args.coarse_kmeans_niter;
        base->cp.nredo = args.coarse_kmeans_nredo;
        BuiltIndex out;
        out.storage_index = base.get();
        out.ivf = base.get();
        out.name = "IVF+AREPQ";
        out.index = std::move(base);
        return out;
    }

    if (args.target == "bapq") {
        auto codec = std::make_unique<epq::IndexBAPQ>(ds.d, args.bits, 4);
        codec->bmax = 12;
        codec->seed = 123;
        codec->max_train_rows = ds.xt.rows();
        codec->pca_max_train_rows = ds.xt.rows();
        codec->kmeans_niter = 50;
        codec->kmeans_nredo = 3;
        auto* quantizer = new faiss::IndexFlatL2(ds.d);
        auto base = std::make_unique<epq::IndexIVFCodec<epq::IndexBAPQ>>(
                std::move(codec), quantizer, args.nlist, "IVF+BAPQ");
        base->cp.niter = args.coarse_kmeans_niter;
        base->cp.nredo = args.coarse_kmeans_nredo;
        BuiltIndex out;
        out.storage_index = base.get();
        out.ivf = base.get();
        out.name = "IVF+BAPQ";
        out.index = std::move(base);
        return out;
    }

    fail("unsupported target: " + args.target);
}

void wrap_refine(BuiltIndex& built) {
    auto wrapper = std::make_unique<faiss::IndexRefineFlat>(built.index.release());
    wrapper->own_fields = true;
    built.index = std::move(wrapper);
    built.name += "+RefineFlat";
}

void fill_summary_metadata(
        const Args& args,
        const BuiltIndex& built,
        Summary& summary) {
    summary.budget_bits = args.bits;
    if (args.target == "pq") {
        summary.components = args.bits / 8;
        auto* ivfpq = dynamic_cast<const epq::TimedIndexIVFPQ*>(built.ivf);
        if (ivfpq != nullptr) {
            const auto& stats = ivfpq->train_stats();
            summary.structure_time = 0.0;
            summary.preparation_time = 0.0;
            summary.codebook_time = stats.encoder_train_time;
            summary.train_time = stats.encoder_train_time;
            summary.coarse.train_time = stats.coarse_train_time;
        }
        return;
    }
    if (args.target == "opq") {
        summary.components = args.bits / 8;
        auto* ivfpq = dynamic_cast<const epq::TimedIndexIVFPQ*>(built.ivf);
        auto* pre =
                dynamic_cast<const epq::TimedIndexPreTransform*>(built.storage_index);
        if (ivfpq != nullptr) {
            summary.coarse.train_time = ivfpq->train_stats().coarse_train_time;
            summary.codebook_time = ivfpq->train_stats().encoder_train_time;
        }
        if (pre != nullptr) {
            const auto& stats = pre->train_stats();
            summary.structure_time = 0.0;
            summary.preparation_time =
                    stats.transform_train_time + stats.transform_apply_time;
            if (ivfpq == nullptr) {
                summary.train_time = stats.total_train_time;
            } else {
                summary.train_time =
                        summary.preparation_time + summary.codebook_time;
            }
        }
        return;
    }
    if (args.target == "rabitq") {
        auto* ivf_rabitq =
                dynamic_cast<const epq::TimedIndexIVFRaBitQ*>(built.ivf);
        if (ivf_rabitq != nullptr) {
            const auto& stats = ivf_rabitq->train_stats();
            summary.components = built.ivf->d;
            summary.budget_bits = static_cast<int>(ivf_rabitq->rabitq.code_size * 8);
            summary.structure_time = 0.0;
            summary.preparation_time = 0.0;
            summary.codebook_time = stats.encoder_train_time;
            summary.train_time = stats.total_train_time;
            summary.coarse.train_time = stats.coarse_train_time;
        }
        return;
    }
    if (args.target == "rq") {
        auto* ivf_rq =
                dynamic_cast<const epq::TimedIndexIVFResidualQuantizer*>(built.ivf);
        if (ivf_rq != nullptr) {
            const auto& stats = ivf_rq->train_stats();
            summary.components = static_cast<int>(ivf_rq->rq.M);
            summary.budget_bits = static_cast<int>(ivf_rq->code_size * 8);
            summary.structure_time = 0.0;
            summary.preparation_time = 0.0;
            summary.codebook_time = stats.encoder_train_time;
            summary.train_time = stats.total_train_time;
            summary.coarse.train_time = stats.coarse_train_time;
        }
        return;
    }
    if (args.target == "lsq") {
        auto* ivf_lsq = dynamic_cast<const epq::TimedIndexIVFLocalSearchQuantizer*>(
                built.ivf);
        if (ivf_lsq != nullptr) {
            const auto& stats = ivf_lsq->train_stats();
            summary.components = static_cast<int>(ivf_lsq->lsq.M);
            summary.budget_bits = static_cast<int>(ivf_lsq->code_size * 8);
            summary.structure_time = 0.0;
            summary.preparation_time = 0.0;
            summary.codebook_time = stats.encoder_train_time;
            summary.train_time = stats.total_train_time;
            summary.coarse.train_time = stats.coarse_train_time;
        }
        return;
    }
    if (args.target == "epq") {
        auto* ivf_epq =
                dynamic_cast<epq::IndexIVFCodec<epq::IndexEPQ>*>(built.ivf);
        if (ivf_epq != nullptr) {
            const auto& codec = ivf_epq->codec();
            summary.components = static_cast<int>(codec.structure().group_count());
            summary.budget_bits = codec.total_bits;
            const auto& stats = codec.training_stats();
            summary.structure_time = stats.structure_time;
            summary.preparation_time = stats.preparation_time;
            summary.codebook_time = stats.codebook_time;
        }
        return;
    }
    if (args.target == "dpopq" || args.target == "dp_opq" ||
        args.target == "dp-opq") {
        auto* ivf_dpopq =
                dynamic_cast<epq::IndexIVFCodec<epq::IndexDPOPQ>*>(built.ivf);
        if (ivf_dpopq != nullptr) {
            const auto& codec = ivf_dpopq->codec();
            summary.components = codec.component_count();
            summary.budget_bits = codec.total_bits;
            const auto& stats = codec.training_stats();
            summary.structure_time = stats.structure_time;
            summary.preparation_time = stats.preparation_time;
            summary.codebook_time = stats.codebook_time;
        }
        return;
    }
    if (args.target == "vaq") {
        auto* ivf_vaq =
                dynamic_cast<epq::IndexIVFCodec<epq::IndexVAQ>*>(built.ivf);
        if (ivf_vaq != nullptr) {
            const auto& codec = ivf_vaq->codec();
            summary.components = codec.component_count();
            summary.budget_bits = codec.total_bits;
            const auto& stats = codec.training_stats();
            summary.structure_time = stats.structure_time;
            summary.preparation_time = stats.preparation_time;
            summary.codebook_time = stats.codebook_time;
        }
        return;
    }
    if (is_arepq_target(args.target)) {
        auto* ivf_arepq =
                dynamic_cast<epq::IndexIVFCodec<epq::IndexAREPQ>*>(built.ivf);
        if (ivf_arepq != nullptr) {
            const auto& codec = ivf_arepq->codec();
            summary.components = codec.component_count();
            summary.budget_bits = codec.total_bits;
            const auto& stats = codec.training_stats();
            summary.structure_time = stats.structure_time;
            summary.preparation_time = stats.preparation_time;
            summary.codebook_time = stats.codebook_time;
        }
        return;
    }
    if (args.target == "bapq") {
        auto* ivf_bapq =
                dynamic_cast<epq::IndexIVFCodec<epq::IndexBAPQ>*>(built.ivf);
        if (ivf_bapq != nullptr) {
            const auto& codec = ivf_bapq->codec();
            summary.components = codec.component_count();
            summary.budget_bits = codec.total_bits;
            const auto& stats = codec.training_stats();
            summary.structure_time = stats.structure_time;
            summary.preparation_time = stats.preparation_time;
            summary.codebook_time = stats.codebook_time;
        }
    }
}

nlohmann::json build_method_metadata(
        const Args& args,
        const BuiltIndex& built,
        const Dataset& ds) {
    if (args.target == "pq") {
        const int M = args.bits / 8;
        return {
                {"family", "pq"},
                {"impl", "faiss"},
                {"index_family", "IVF+PQ"},
                {"d", ds.d},
                {"M", M},
                {"nbits", 8},
                {"total_bits", args.bits},
                {"metric", "L2"},
                {"by_residual", true},
        };
    }

    if (args.target == "opq") {
        const int M = args.bits / 8;
        const int d2 = ((ds.d + M - 1) / M) * M;
        return {
                {"family", "opq"},
                {"impl", "faiss"},
                {"index_family", "IVF+OPQ"},
                {"d", ds.d},
                {"d2", d2},
                {"M", M},
                {"nbits", 8},
                {"total_bits", args.bits},
                {"metric", "L2"},
                {"by_residual", true},
                {"opq",
                 epq::benchmark_metadata::default_opq_metadata(ds.d, M, d2)},
        };
    }

    if (args.target == "rabitq") {
        auto* ivf_rabitq = dynamic_cast<const faiss::IndexIVFRaBitQ*>(built.ivf);
        if (ivf_rabitq != nullptr) {
            return {
                    {"family", "rabitq"},
                    {"impl", "faiss"},
                    {"index_family", "IVF+RaBitQ"},
                    {"native_index", "IndexIVFRaBitQ"},
                    {"d", ds.d},
                    {"nb_bits", static_cast<int>(ivf_rabitq->rabitq.nb_bits)},
                    {"nominal_budget_bits", args.bits},
                    {"effective_budget_bits",
                     static_cast<int>(ivf_rabitq->rabitq.code_size * 8)},
                    {"metric", "L2"},
                    {"by_residual", true},
                    {"qb", static_cast<int>(ivf_rabitq->qb)},
            };
        }
    }

    if (args.target == "rq") {
        auto* ivf_rq =
                dynamic_cast<const epq::TimedIndexIVFResidualQuantizer*>(built.ivf);
        if (ivf_rq != nullptr) {
            return {
                    {"family", "rq"},
                    {"impl", "faiss"},
                    {"index_family", "IVF+RQ"},
                    {"native_index", "IndexIVFResidualQuantizer"},
                    {"d", ds.d},
                    {"M", static_cast<int>(ivf_rq->rq.M)},
                    {"nbits", 8},
                    {"total_bits", args.bits},
                    {"effective_budget_bits", static_cast<int>(ivf_rq->code_size * 8)},
                    {"metric", "L2"},
                    {"by_residual", true},
                    {"search_type", "ST_norm_qint8"},
                    {"max_beam_size", ivf_rq->rq.max_beam_size},
                    {"train_type", ivf_rq->rq.train_type},
                    {"use_beam_LUT", ivf_rq->rq.use_beam_LUT},
            };
        }
    }

    if (args.target == "lsq") {
        auto* ivf_lsq = dynamic_cast<const epq::TimedIndexIVFLocalSearchQuantizer*>(
                built.ivf);
        if (ivf_lsq != nullptr) {
            return {
                    {"family", "lsq"},
                    {"impl", "faiss"},
                    {"index_family", "IVF+LSQ"},
                    {"native_index", "IndexIVFLocalSearchQuantizer"},
                    {"d", ds.d},
                    {"M", static_cast<int>(ivf_lsq->lsq.M)},
                    {"nbits", 8},
                    {"total_bits", args.bits},
                    {"effective_budget_bits", static_cast<int>(ivf_lsq->code_size * 8)},
                    {"metric", "L2"},
                    {"by_residual", true},
                    {"search_type", "ST_norm_qint8"},
                    {"train_iters", static_cast<int>(ivf_lsq->lsq.train_iters)},
                    {"encode_ils_iters",
                     static_cast<int>(ivf_lsq->lsq.encode_ils_iters)},
                    {"train_ils_iters",
                     static_cast<int>(ivf_lsq->lsq.train_ils_iters)},
                    {"icm_iters", static_cast<int>(ivf_lsq->lsq.icm_iters)},
            };
        }
    }

    if (args.target == "epq") {
        auto* ivf_epq =
                dynamic_cast<epq::IndexIVFCodec<epq::IndexEPQ>*>(built.ivf);
        if (ivf_epq != nullptr) {
            auto meta = epq::benchmark_metadata::summarize_index_epq(
                    ivf_epq->codec(), "epq");
            meta["impl"] = "cpp";
            meta["index_family"] = "IVF+EPQ";
            meta["by_residual"] = true;
            return meta;
        }
    }

    if (args.target == "dpopq" || args.target == "dp_opq" ||
        args.target == "dp-opq") {
        auto* ivf_dpopq =
                dynamic_cast<epq::IndexIVFCodec<epq::IndexDPOPQ>*>(built.ivf);
        if (ivf_dpopq != nullptr) {
            auto meta = ivf_dpopq->codec().metadata();
            meta["impl"] = "paper_based_implementation";
            meta["index_family"] = "IVF+DPOPQ";
            meta["by_residual"] = true;
            meta["metric"] = "L2";
            return meta;
        }
    }

    if (args.target == "vaq") {
        auto* ivf_vaq =
                dynamic_cast<epq::IndexIVFCodec<epq::IndexVAQ>*>(built.ivf);
        if (ivf_vaq != nullptr) {
            auto meta = ivf_vaq->codec().metadata();
            meta["index_family"] = "IVF+VAQ";
            meta["by_residual"] = true;
            meta["metric"] = "L2";
            return meta;
        }
    }

    if (is_arepq_target(args.target)) {
        auto* ivf_arepq =
                dynamic_cast<epq::IndexIVFCodec<epq::IndexAREPQ>*>(built.ivf);
        if (ivf_arepq != nullptr) {
            auto meta = summarize_arepq_method(ivf_arepq->codec());
            meta["impl"] = "cpp";
            meta["index_family"] = "IVF+AREPQ";
            meta["by_residual"] = true;
            return meta;
        }
    }

    if (args.target == "bapq") {
        auto* ivf_bapq =
                dynamic_cast<epq::IndexIVFCodec<epq::IndexBAPQ>*>(built.ivf);
        if (ivf_bapq != nullptr) {
            auto meta =
                    epq::benchmark_metadata::summarize_index_bapq(ivf_bapq->codec());
            meta["impl"] = "cpp";
            meta["index_family"] = "IVF+BAPQ";
            meta["by_residual"] = true;
            return meta;
        }
    }

    return nlohmann::json::object();
}

size_t estimate_storage_index_bytes(
        const Args& args,
        const BuiltIndex& built) {
    if (args.target == "pq" || args.target == "opq" || args.target == "rabitq" ||
        args.target == "rq" || args.target == "lsq") {
        return epq::serialized_faiss_index_bytes(*built.storage_index);
    }
    const size_t quantizer_bytes =
            epq::serialized_faiss_index_bytes(*built.ivf->quantizer);
    const size_t invlists_bytes =
            static_cast<size_t>(built.ivf->ntotal) *
                    (sizeof(faiss::idx_t) + built.ivf->code_size) +
            built.ivf->nlist * (sizeof(size_t) + sizeof(void*));
    if (args.target == "epq") {
        auto* ivf_epq =
                dynamic_cast<epq::IndexIVFCodec<epq::IndexEPQ>*>(built.ivf);
        if (ivf_epq != nullptr) {
            return quantizer_bytes + invlists_bytes +
                    ivf_epq->codec().serialized_payload_bytes();
        }
    }
    if (args.target == "dpopq" || args.target == "dp_opq" ||
        args.target == "dp-opq") {
        auto* ivf_dpopq =
                dynamic_cast<epq::IndexIVFCodec<epq::IndexDPOPQ>*>(built.ivf);
        if (ivf_dpopq != nullptr) {
            return quantizer_bytes + invlists_bytes +
                    ivf_dpopq->codec().serialized_payload_bytes();
        }
    }
    if (args.target == "vaq") {
        auto* ivf_vaq =
                dynamic_cast<epq::IndexIVFCodec<epq::IndexVAQ>*>(built.ivf);
        if (ivf_vaq != nullptr) {
            return quantizer_bytes + invlists_bytes +
                    ivf_vaq->codec().serialized_payload_bytes();
        }
    }
    if (is_arepq_target(args.target)) {
        auto* ivf_arepq =
                dynamic_cast<epq::IndexIVFCodec<epq::IndexAREPQ>*>(built.ivf);
        if (ivf_arepq != nullptr) {
            return quantizer_bytes + invlists_bytes +
                    ivf_arepq->codec().serialized_payload_bytes();
        }
    }
    if (args.target == "bapq") {
        auto* ivf_bapq =
                dynamic_cast<epq::IndexIVFCodec<epq::IndexBAPQ>*>(built.ivf);
        if (ivf_bapq != nullptr) {
            return quantizer_bytes + invlists_bytes +
                    ivf_bapq->codec().serialized_payload_bytes();
        }
    }
    return quantizer_bytes + invlists_bytes;
}

double compute_reconstruction_error(
        const Dataset& ds,
        const Args& args,
        faiss::IndexIVF& ivf) {
    const auto ids = sample_ids(ds.nb, args.recon_sample);
    if (ids.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    BaseVectorReader reader(ds.spec.base);
    if (reader.dim() != ds.d) {
        fail("reconstruction base vector dimension mismatch");
    }
    const size_t sample_n = ids.size();
    RowMatrixXf original(static_cast<Eigen::Index>(sample_n), ds.d);
    RowMatrixXf recons(static_cast<Eigen::Index>(sample_n), ds.d);
    ivf.make_direct_map(true);
    std::vector<float> row(static_cast<size_t>(ds.d));
    for (size_t i = 0; i < sample_n; ++i) {
        reader.read_row(ids[i], row.data());
        for (int j = 0; j < ds.d; ++j) {
            original(static_cast<Eigen::Index>(i), j) = row[static_cast<size_t>(j)];
        }
        ivf.reconstruct(ids[i], recons.row(static_cast<Eigen::Index>(i)).data());
    }
    return (original - recons).array().square().sum() / original.rows();
}

void add_base_streamed(
        const Dataset& ds,
        const Args& args,
        faiss::Index& index) {
    stream_float_matrix(
            ds.spec.base,
            ds.nb,
            args.base_batch_size,
            [&](size_t, const RowMatrixXf& batch) {
                index.add(batch.rows(), batch.data());
            });
}

Summary run_benchmark(const Args& args, Dataset ds) {
    BuiltIndex built = build_index(args, ds);
    if (args.refine) {
        wrap_refine(built);
    }
    built.ivf->nprobe = static_cast<size_t>(args.nprobe);

    Summary summary;
    summary.name = built.name;

    double epq_coarse_train_time = std::numeric_limits<double>::quiet_NaN();
    if (args.target == "epq") {
        if (auto* ivf_epq =
                    dynamic_cast<epq::IndexIVFCodec<epq::IndexEPQ>*>(built.ivf);
            ivf_epq != nullptr && !args.refine) {
            const auto train_summary = train_epq_ivf_index(*ivf_epq, ds);
            summary.train_time = train_summary.total_time;
            epq_coarse_train_time = train_summary.coarse_train_time;
        } else {
            const auto train_t0 = std::chrono::steady_clock::now();
            built.index->train(ds.xt.rows(), ds.xt.data());
            summary.train_time = std::chrono::duration<double>(
                                         std::chrono::steady_clock::now() - train_t0)
                                         .count();
        }
    } else if (is_arepq_target(args.target)) {
        if (auto* ivf_arepq =
                    dynamic_cast<epq::IndexIVFCodec<epq::IndexAREPQ>*>(built.ivf);
            ivf_arepq != nullptr && !args.refine) {
            const auto train_summary = train_epq_ivf_index(*ivf_arepq, ds);
            summary.train_time = train_summary.total_time;
            epq_coarse_train_time = train_summary.coarse_train_time;
        } else {
            const auto train_t0 = std::chrono::steady_clock::now();
            built.index->train(ds.xt.rows(), ds.xt.data());
            summary.train_time = std::chrono::duration<double>(
                                         std::chrono::steady_clock::now() - train_t0)
                                         .count();
        }
    } else {
        const auto train_t0 = std::chrono::steady_clock::now();
        built.index->train(ds.xt.rows(), ds.xt.data());
        summary.train_time = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - train_t0)
                                     .count();
    }
    fill_summary_metadata(args, built, summary);
    summary.method_metadata = build_method_metadata(args, built, ds);
    if (std::isfinite(epq_coarse_train_time)) {
        summary.coarse.train_time = epq_coarse_train_time;
    }
    if (args.target == "epq" || is_arepq_target(args.target)) {
        if (const auto structure_path = resolve_structure_path_for_target(args, false);
            structure_path.has_value()) {
            if (auto* ivf_epq =
                        dynamic_cast<epq::IndexIVFCodec<epq::IndexEPQ>*>(built.ivf);
                ivf_epq != nullptr) {
                std::filesystem::create_directories(structure_path->parent_path());
                ivf_epq->codec().structure().save_json(structure_path->string());
            } else if (auto* ivf_arepq =
                               dynamic_cast<epq::IndexIVFCodec<epq::IndexAREPQ>*>(
                                       built.ivf);
                       ivf_arepq != nullptr) {
                std::filesystem::create_directories(structure_path->parent_path());
                ivf_arepq->codec().main_index().structure().save_json(
                        structure_path->string());
            }
        }
    }

    const auto add_t0 = std::chrono::steady_clock::now();
    add_base_streamed(ds, args, *built.index);
    summary.add_time = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - add_t0)
                               .count();
    summary.encode_per_vector = summary.add_time / ds.nb;
    summary.index_size_mib = static_cast<double>(estimate_storage_index_bytes(args, built)) /
            (1024.0 * 1024.0);
    summary.reconstruction_error = compute_reconstruction_error(ds, args, *built.ivf);

    const double coarse_train_time = summary.coarse.train_time;
    summary.coarse = {};
    if (std::isfinite(coarse_train_time) && coarse_train_time > 0.0) {
        summary.coarse.train_time = coarse_train_time;
    } else if (args.target != "rabitq") {
        // Fallback for targets without explicit coarse-train instrumentation.
        summary.coarse.train_time = summary.train_time;
    }
    // Official Faiss IVF still does not expose coarse-only add timing, so we
    // report the aggregate add stage here to keep the schema complete.
    summary.coarse.add_time = summary.add_time;
    const auto assigned = build_query_assignments(*built.ivf, ds, args, summary.coarse);
    compute_candidate_stats(*built.ivf, assigned, ds, summary.coarse);

    const int eval_topk = std::max(args.topk, args.metric_topk);
    std::vector<float> distances(
            static_cast<size_t>(ds.xq.rows()) * static_cast<size_t>(eval_topk));
    std::vector<faiss::idx_t> labels(
            static_cast<size_t>(ds.xq.rows()) * static_cast<size_t>(eval_topk));
    auto evaluate_search = [&](std::vector<float>& out_distances,
                               std::vector<faiss::idx_t>& out_labels) {
        SearchMetrics metrics;
        double approx_search_time = std::numeric_limits<double>::quiet_NaN();

        if (args.refine) {
            const faiss::idx_t k_base = std::max<faiss::idx_t>(
                    eval_topk,
                    static_cast<faiss::idx_t>(eval_topk * args.refine_k_factor));
            std::vector<float> base_distances(
                    static_cast<size_t>(ds.xq.rows()) * static_cast<size_t>(k_base));
            std::vector<faiss::idx_t> base_labels(
                    static_cast<size_t>(ds.xq.rows()) * static_cast<size_t>(k_base));
            const auto approx_t0 = std::chrono::steady_clock::now();
            built.ivf->search(
                    ds.xq.rows(),
                    ds.xq.data(),
                    k_base,
                    base_distances.data(),
                    base_labels.data());
            approx_search_time = std::chrono::duration<double>(
                                         std::chrono::steady_clock::now() - approx_t0)
                                         .count();
        }

        const auto search_t0 = std::chrono::steady_clock::now();
        if (args.refine) {
            auto* refine_index =
                    dynamic_cast<faiss::IndexRefineFlat*>(built.index.get());
            if (refine_index == nullptr) {
                fail("internal error: refine wrapper missing");
            }
            faiss::IndexRefineSearchParameters refine_params;
            refine_params.k_factor = args.refine_k_factor;
            refine_index->search(
                    ds.xq.rows(),
                    ds.xq.data(),
                    eval_topk,
                    out_distances.data(),
                    out_labels.data(),
                    &refine_params);
        } else {
            built.index->search(
                    ds.xq.rows(),
                    ds.xq.data(),
                    eval_topk,
                    out_distances.data(),
                    out_labels.data());
        }
        metrics.search_time = std::chrono::duration<double>(
                                      std::chrono::steady_clock::now() - search_t0)
                                      .count();
        metrics.total_query_time = metrics.search_time;
        if (args.refine) {
            const double approx_total = std::isfinite(approx_search_time)
                    ? std::min(
                              metrics.total_query_time,
                              std::max(0.0, approx_search_time))
                    : metrics.total_query_time;
            if (std::isfinite(summary.coarse.assign_time)) {
                metrics.rerank_time =
                        std::max(0.0, approx_total - summary.coarse.assign_time);
            } else {
                metrics.rerank_time = approx_total;
            }
            metrics.refine_time =
                    std::max(0.0, metrics.total_query_time - approx_total);
        } else {
            if (std::isfinite(summary.coarse.assign_time)) {
                metrics.rerank_time = std::max(
                        0.0,
                        metrics.total_query_time - summary.coarse.assign_time);
            }
            metrics.refine_time = 0.0;
        }
        metrics.search_per_query = metrics.total_query_time / ds.xq.rows();
        metrics.qps = ds.xq.rows() / metrics.total_query_time;
        metrics.recall1 =
                recall_at_k(out_labels, ds.xq.rows(), eval_topk, ds.gt, ds.gt_k, 1);
        metrics.recall10 =
                recall_at_k(out_labels, ds.xq.rows(), eval_topk, ds.gt, ds.gt_k, 10);
        metrics.recall100 =
                recall_at_k(out_labels, ds.xq.rows(), eval_topk, ds.gt, ds.gt_k, 100);
        if (eval_topk >= 1000) {
            metrics.recall1000 = recall_at_k(
                    out_labels,
                    ds.xq.rows(),
                    eval_topk,
                    ds.gt,
                    ds.gt_k,
                    1000);
            metrics.overlap1000 = overlap_at_k(
                    out_labels,
                    ds.xq.rows(),
                    eval_topk,
                    ds.gt,
                    ds.gt_k,
                    1000,
                    1000);
        }
        return metrics;
    };

    const epq::EpqIvfSearchMode original_search_mode = epq::epq_ivf_search_mode();
    const bool run_three_way_compare =
            compare_epq_ivf_three_way() && args.target == "epq" && !args.refine;
    const bool run_scanner_compare =
            compare_epq_ivf_scanners() && args.target == "epq" && !args.refine;
    const SearchMetrics primary_metrics = evaluate_search(distances, labels);
    summary.search_time = primary_metrics.search_time;
    summary.total_query_time = primary_metrics.total_query_time;
    summary.rerank_time = primary_metrics.rerank_time;
    summary.refine_time = primary_metrics.refine_time;
    summary.search_per_query = primary_metrics.search_per_query;
    summary.qps = primary_metrics.qps;
    summary.recall1 = primary_metrics.recall1;
    summary.recall10 = primary_metrics.recall10;
    summary.recall100 = primary_metrics.recall100;
    summary.recall1000 = primary_metrics.recall1000;
    summary.overlap1000 = primary_metrics.overlap1000;
    if (args.target == "epq" && !args.refine) {
        if (auto* ivf_epq =
                    dynamic_cast<epq::IndexIVFCodec<epq::IndexEPQ>*>(built.ivf);
            ivf_epq != nullptr) {
            const auto diag = ivf_epq->last_epq_ivf_diagnostics();
            if (!diag.empty()) {
                summary.diagnostics["epq_ivf_fast_path"] = diag;
            }
        }
    }

    if (run_three_way_compare) {
        auto build_metrics_json = [](const SearchMetrics& metrics) {
            return nlohmann::json{
                    {"search_time", metrics.search_time},
                    {"search_per_query", metrics.search_per_query},
                    {"qps", metrics.qps},
                    {"recall1", metrics.recall1},
                    {"recall10", metrics.recall10},
                    {"recall100", metrics.recall100},
                    {"recall1000", metrics.recall1000},
                    {"overlap1000", metrics.overlap1000},
            };
        };
        auto build_match_json = [&](const std::vector<faiss::idx_t>& a,
                                    const std::vector<faiss::idx_t>& b) {
            return nlohmann::json{
                    {"exact_position_match",
                     {
                             {"top1",
                              exact_position_match_at_k(
                                      a, b, ds.xq.rows(), eval_topk, 1)},
                             {"top10",
                              exact_position_match_at_k(
                                      a, b, ds.xq.rows(), eval_topk, 10)},
                             {"top100",
                              exact_position_match_at_k(
                                      a, b, ds.xq.rows(), eval_topk, 100)},
                             {"top1000",
                              exact_position_match_at_k(
                                      a, b, ds.xq.rows(), eval_topk, 1000)},
                     }},
                    {"exact_query_match",
                     {
                             {"top1",
                              exact_query_match_at_k(
                                      a, b, ds.xq.rows(), eval_topk, 1)},
                             {"top10",
                              exact_query_match_at_k(
                                      a, b, ds.xq.rows(), eval_topk, 10)},
                             {"top100",
                              exact_query_match_at_k(
                                      a, b, ds.xq.rows(), eval_topk, 100)},
                             {"top1000",
                              exact_query_match_at_k(
                                      a, b, ds.xq.rows(), eval_topk, 1000)},
                     }},
            };
        };

        SearchMetrics scalar_metrics = primary_metrics;
        std::vector<float> scalar_distances = distances;
        std::vector<faiss::idx_t> scalar_labels = labels;
        if (original_search_mode != epq::EpqIvfSearchMode::kScalarLut) {
            epq::set_epq_ivf_search_mode(epq::EpqIvfSearchMode::kScalarLut);
            scalar_metrics = evaluate_search(scalar_distances, scalar_labels);
        }

        std::vector<float> exact_distances(
                static_cast<size_t>(ds.xq.rows()) * static_cast<size_t>(eval_topk));
        std::vector<faiss::idx_t> exact_labels(
                static_cast<size_t>(ds.xq.rows()) * static_cast<size_t>(eval_topk));
        epq::set_epq_ivf_search_mode(epq::EpqIvfSearchMode::kExactDecode);
        const SearchMetrics exact_metrics =
                evaluate_search(exact_distances, exact_labels);
        epq::set_epq_ivf_search_mode(original_search_mode);

        summary.diagnostics["three_way_compare"] = {
                {"primary_mode",
                 epq::epq_ivf_mode_name(original_search_mode)},
                {"primary", build_metrics_json(primary_metrics)},
                {"scalar_lut", build_metrics_json(scalar_metrics)},
                {"exact_decode", build_metrics_json(exact_metrics)},
                {"primary_vs_scalar", build_match_json(labels, scalar_labels)},
                {"primary_vs_exact", build_match_json(labels, exact_labels)},
                {"scalar_vs_exact", build_match_json(scalar_labels, exact_labels)},
        };
    }

    if (run_scanner_compare) {
        std::vector<float> fallback_distances(
                static_cast<size_t>(ds.xq.rows()) * static_cast<size_t>(eval_topk));
        std::vector<faiss::idx_t> fallback_labels(
                static_cast<size_t>(ds.xq.rows()) * static_cast<size_t>(eval_topk));
        epq::set_epq_ivf_search_mode(epq::EpqIvfSearchMode::kFallbackScanner);
        const SearchMetrics fallback_metrics =
                evaluate_search(fallback_distances, fallback_labels);
        epq::set_epq_ivf_search_mode(original_search_mode);

        summary.diagnostics["scanner_compare"] = {
                {"primary_mode", epq::epq_ivf_mode_name(original_search_mode)},
                {"primary",
                 {
                         {"search_time", primary_metrics.search_time},
                         {"search_per_query", primary_metrics.search_per_query},
                         {"qps", primary_metrics.qps},
                         {"recall1", primary_metrics.recall1},
                         {"recall10", primary_metrics.recall10},
                         {"recall100", primary_metrics.recall100},
                         {"recall1000", primary_metrics.recall1000},
                         {"overlap1000", primary_metrics.overlap1000},
                 }},
                {"fallback",
                 {
                         {"search_time", fallback_metrics.search_time},
                         {"search_per_query", fallback_metrics.search_per_query},
                         {"qps", fallback_metrics.qps},
                         {"recall1", fallback_metrics.recall1},
                         {"recall10", fallback_metrics.recall10},
                         {"recall100", fallback_metrics.recall100},
                         {"recall1000", fallback_metrics.recall1000},
                         {"overlap1000", fallback_metrics.overlap1000},
                 }},
                {"exact_position_match",
                 {
                         {"top1",
                          exact_position_match_at_k(
                                  labels, fallback_labels, ds.xq.rows(), eval_topk, 1)},
                         {"top10",
                          exact_position_match_at_k(
                                  labels, fallback_labels, ds.xq.rows(), eval_topk, 10)},
                         {"top100",
                          exact_position_match_at_k(
                                  labels, fallback_labels, ds.xq.rows(), eval_topk, 100)},
                         {"top1000",
                          exact_position_match_at_k(
                                  labels, fallback_labels, ds.xq.rows(), eval_topk, 1000)},
                 }},
                {"exact_query_match",
                 {
                         {"top1",
                          exact_query_match_at_k(
                                  labels, fallback_labels, ds.xq.rows(), eval_topk, 1)},
                         {"top10",
                          exact_query_match_at_k(
                                  labels, fallback_labels, ds.xq.rows(), eval_topk, 10)},
                         {"top100",
                          exact_query_match_at_k(
                                  labels, fallback_labels, ds.xq.rows(), eval_topk, 100)},
                         {"top1000",
                          exact_query_match_at_k(
                                  labels, fallback_labels, ds.xq.rows(), eval_topk, 1000)},
                 }},
        };
    } else {
        epq::set_epq_ivf_search_mode(original_search_mode);
    }

    if (!summary.method_metadata.empty()) {
        std::cout << "meta.method " << summary.method_metadata.dump() << '\n';
    }
    std::cout << "===== " << summary.name << '\n';
    std::cout << "\tcomponents: " << summary.components << '\n';
    if (summary.budget_bits > 0) {
        std::cout << "\teffective budget bits: " << summary.budget_bits << '\n';
    }
    std::cout << "\ttrain time: " << std::fixed << std::setprecision(3)
              << summary.train_time << " s\n";
    if (std::isfinite(summary.structure_time)) {
        std::cout << "\tstructure time: " << summary.structure_time << " s\n";
    }
    if (std::isfinite(summary.preparation_time)) {
        std::cout << "\tpreparation time: " << summary.preparation_time << " s\n";
    }
    if (std::isfinite(summary.codebook_time)) {
        std::cout << "\tcodebook time: " << summary.codebook_time << " s\n";
    }
    std::cout << "\tadd time: " << summary.add_time << " s\n";
    std::cout << "\tencode per vector: " << std::setprecision(9)
              << summary.encode_per_vector << " s/vector\n";
    std::cout << std::setprecision(3);
    if (std::isfinite(summary.coarse.assign_time)) {
        std::cout << "\tcoarse assign time: " << summary.coarse.assign_time << " s\n";
    }
    if (std::isfinite(summary.coarse.avg_candidates)) {
        std::cout << "\tavg candidates/query: " << summary.coarse.avg_candidates << '\n';
    }
    if (std::isfinite(summary.coarse.max_candidates)) {
        std::cout << "\tmax candidates/query: " << summary.coarse.max_candidates << '\n';
    }
    if (std::isfinite(summary.coarse.candidate_hit_rate)) {
        std::cout << "\tcandidate hit@1: " << std::setprecision(4)
                  << summary.coarse.candidate_hit_rate << '\n';
        std::cout << std::setprecision(3);
    }
    if (std::isfinite(summary.rerank_time)) {
        std::cout << "\tpost-coarse query time: " << summary.rerank_time << " s\n";
    }
    if (std::isfinite(summary.refine_time)) {
        std::cout << "\trefine time: " << summary.refine_time << " s\n";
    }
    std::cout << "\tsearch time: " << summary.search_time << " s\n";
    std::cout << "\ttotal query time: " << summary.total_query_time << " s\n";
    std::cout << "\tsearch per query: " << std::setprecision(9)
              << summary.search_per_query << " s/query\n";
    std::cout << std::setprecision(3);
    std::cout << "\tQPS: " << summary.qps << '\n';
    std::cout << "\trecall@1: " << std::setprecision(4) << summary.recall1
              << " recall@10: " << summary.recall10
              << " recall@100: " << summary.recall100 << '\n';
    if (std::isfinite(summary.recall1000)) {
        std::cout << "\trecall@1000: " << summary.recall1000 << '\n';
    }
    if (std::isfinite(summary.overlap1000)) {
        std::cout << "\toverlap@1000(gt=1000): " << summary.overlap1000 << '\n';
    }
    if (!summary.diagnostics.empty()) {
        std::cout << "meta.diagnostics " << summary.diagnostics.dump() << '\n';
    }
    if (std::isfinite(summary.reconstruction_error)) {
        std::cout << "\treconstruction MSE/sample: "
                  << summary.reconstruction_error << '\n';
    }
    if (std::isfinite(summary.index_size_mib)) {
        std::cout << "\tindex size (MiB): " << summary.index_size_mib << '\n';
    }
    return summary;
}

nlohmann::json to_json(
        const Dataset& ds,
        const Args& args,
        const Summary& summary,
        const nlohmann::json& metadata) {
    nlohmann::json j;
    j["dataset"] = ds.spec.name;
    j["dim"] = ds.d;
    j["nb"] = ds.nb;
    j["nq"] = ds.xq.rows();
    j["nt"] = ds.xt.rows();
    j["gt_k"] = ds.gt_k;
    j["nominal_bits"] = args.bits;
    j["bits"] = summary.budget_bits > 0 ? summary.budget_bits : args.bits;
    j["nlist"] = args.nlist;
    j["nprobe"] = args.nprobe;
    j["target"] = args.target;
    j["refine"] = args.refine;
    j["refine_k_factor"] = args.refine_k_factor;
    j["topk"] = args.topk;
    j["metric_topk"] = args.metric_topk;
    j["base_batch_size"] = args.base_batch_size;
    j["name"] = summary.name;
    j["train_time"] = summary.train_time;
    j["add_time"] = summary.add_time;
    j["search_time"] = summary.search_time;
    j["search_per_query"] = summary.search_per_query;
    j["qps"] = summary.qps;
    j["recall1"] = summary.recall1;
    j["recall10"] = summary.recall10;
    j["recall100"] = summary.recall100;
    j["recall1000"] = summary.recall1000;
    j["overlap1000"] = summary.overlap1000;
    j["metadata"] = metadata;
    j["method"] = summary.method_metadata;
    if (!summary.diagnostics.empty()) {
        j["diagnostics"] = summary.diagnostics;
    }
    j["coarse"] = {
            {"train_time", summary.coarse.train_time},
            {"add_time", summary.coarse.add_time},
            {"assign_time", summary.coarse.assign_time},
            {"avg_candidates", summary.coarse.avg_candidates},
            {"max_candidates", summary.coarse.max_candidates},
            {"candidate_hit_rate", summary.coarse.candidate_hit_rate},
    };
    j["targets"] = nlohmann::json::array({
            {
                    {"name", summary.name},
                    {"budget_bits", summary.budget_bits > 0 ? summary.budget_bits : args.bits},
                    {"components", summary.components},
                    {"train_total", summary.train_time},
                    {"structure_time", summary.structure_time},
                    {"preparation_time", summary.preparation_time},
                    {"codebook_time", summary.codebook_time},
                    {"add_time", summary.add_time},
                    {"encode_per_vector", summary.encode_per_vector},
                    {"rerank_time", summary.rerank_time},
                    {"refine_time", summary.refine_time},
                    {"total_query_time", summary.total_query_time},
                    {"search_per_query", summary.search_per_query},
                    {"qps", summary.qps},
                    {"recall1", summary.recall1},
                    {"recall10", summary.recall10},
                    {"recall100", summary.recall100},
                    {"recall1000", summary.recall1000},
                    {"overlap1000", summary.overlap1000},
                    {"reconstruction_error", summary.reconstruction_error},
                    {"index_size_mib", summary.index_size_mib},
                    {"method", summary.method_metadata},
            },
    });
    return j;
}

Args parse_args(int argc, char** argv) {
    if (argc < 6) {
        fail(
                "usage: joint_benchmark <dataset> <bits> <nlist> <nprobe> <target> "
                "[--config=PATH] [--data-root=PATH] [--deep1b-root=PATH] "
                "[--epq-structure=PATH] [--json-out=PATH] [--topk=N] [--metric-topk=N] "
                "[--recon-sample=N] [--threads=N] [--train-limit=N] [--base-limit=N] [--query-limit=N] "
                "[--base-batch-size=N] [--refine] [--refine-k-factor=F] "
                "[--coarse-kmeans-niter=N] [--coarse-kmeans-nredo=N] "
                "[--epq-transform-niter=N] [--epq-kmeans-niter=N] "
                "[--epq-transform-kmeans-niter=N] "
                "[--vaq-subspaces=N] [--vaq-min-bits=N] [--vaq-max-bits=N]");
    }

    Args args;
    args.dataset = argv[1];
    args.bits = std::stoi(argv[2]);
    args.nlist = std::stoi(argv[3]);
    args.nprobe = std::stoi(argv[4]);
    args.target = argv[5];
    for (int i = 6; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg.starts_with("--data-root=")) {
            args.data_root = std::string(arg.substr(12));
        } else if (arg.starts_with("--config=")) {
            args.config_path = std::filesystem::path(std::string(arg.substr(9)));
            args.epq_config = epq::load_json_file(*args.config_path);
            reject_retired_query_weighted_training_config(*args.epq_config);
        } else if (arg.starts_with("--deep1b-root=")) {
            args.deep1b_root = std::string(arg.substr(14));
        } else if (arg.starts_with("--epq-structure=")) {
            args.epq_structure = std::filesystem::path(std::string(arg.substr(16)));
        } else if (arg.starts_with("--json-out=")) {
            args.json_out = std::filesystem::path(std::string(arg.substr(11)));
        } else if (arg.starts_with("--topk=")) {
            args.topk = std::stoi(std::string(arg.substr(7)));
        } else if (arg.starts_with("--metric-topk=")) {
            args.metric_topk = std::stoi(std::string(arg.substr(14)));
        } else if (arg.starts_with("--recon-sample=")) {
            args.recon_sample = std::stoi(std::string(arg.substr(15)));
        } else if (arg.starts_with("--threads=")) {
            args.threads = std::stoi(std::string(arg.substr(10)));
        } else if (arg.starts_with("--train-limit=")) {
            args.train_limit = std::stoi(std::string(arg.substr(14)));
        } else if (arg.starts_with("--base-limit=")) {
            args.base_limit = static_cast<size_t>(
                    std::stoull(std::string(arg.substr(13))));
        } else if (arg.starts_with("--query-limit=")) {
            args.query_limit = std::stoi(std::string(arg.substr(14)));
        } else if (arg.starts_with("--base-batch-size=")) {
            args.base_batch_size = std::stoi(std::string(arg.substr(18)));
        } else if (arg == "--refine") {
            args.refine = true;
        } else if (arg.starts_with("--refine-k-factor=")) {
            args.refine_k_factor = std::stof(std::string(arg.substr(18)));
        } else if (arg.starts_with("--coarse-kmeans-niter=")) {
            args.coarse_kmeans_niter = std::stoi(std::string(arg.substr(22)));
        } else if (arg.starts_with("--coarse-kmeans-nredo=")) {
            args.coarse_kmeans_nredo = std::stoi(std::string(arg.substr(22)));
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
        } else {
            fail("unknown flag: " + std::string(arg));
        }
    }

    if (args.bits <= 0) {
        fail("bits must be positive");
    }
    if (args.nlist <= 0) {
        fail("nlist must be positive");
    }
    if (args.nprobe <= 0 || args.nprobe > args.nlist) {
        fail("nprobe must be in [1, nlist]");
    }
    if (args.topk <= 0) {
        fail("topk must be positive");
    }
    if (args.metric_topk < args.topk) {
        args.metric_topk = args.topk;
    }
    if (args.base_batch_size <= 0) {
        fail("base-batch-size must be positive");
    }
    if (args.recon_sample < 0) {
        fail("recon-sample must be non-negative");
    }
    if (args.refine_k_factor < 1.0f) {
        fail("refine-k-factor must be >= 1");
    }
    if (args.dataset == "deep1b" && args.train_limit <= 0) {
        args.train_limit = 200000;
    }
    return args;
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

        Dataset ds = load_dataset(args);
        std::cout << "dataset=" << ds.spec.name
                  << " d=" << ds.d
                  << " nb=" << ds.nb
                  << " nq=" << ds.xq.rows()
                  << " nt=" << ds.xt.rows()
                  << " bits=" << args.bits
                  << " nlist=" << args.nlist
                  << " nprobe=" << args.nprobe
                  << " target=" << args.target
                  << " refine=" << std::boolalpha << args.refine << std::noboolalpha
                  << " refine_k_factor=" << args.refine_k_factor
                  << " topk=" << args.topk
                  << " metric_topk=" << args.metric_topk
                  << " recon_sample=" << args.recon_sample
                  << " threads=" << effective_threads
                  << " train_limit=" << args.train_limit
                  << " base_limit=" << args.base_limit
                  << " query_limit=" << args.query_limit
                  << " base_batch_size=" << args.base_batch_size
                  << " coarse_kmeans_niter=" << args.coarse_kmeans_niter
                  << " coarse_kmeans_nredo=" << args.coarse_kmeans_nredo
                  << '\n';
        nlohmann::json run_meta = {
                {"benchmark", "joint_benchmark"},
                {"protocol", "ivf"},
                {"target", args.target},
                {"bits", args.bits},
                {"nlist", args.nlist},
                {"nprobe", args.nprobe},
                {"refine", args.refine},
                {"refine_k_factor", args.refine_k_factor},
                {"topk", args.topk},
                {"metric_topk", args.metric_topk},
                {"recon_sample", args.recon_sample},
                {"train_limit", args.train_limit},
                {"base_limit", args.base_limit},
                {"query_limit", args.query_limit},
                {"base_batch_size", args.base_batch_size},
                {"coarse_kmeans_niter", args.coarse_kmeans_niter},
                {"coarse_kmeans_nredo", args.coarse_kmeans_nredo},
                {"data_root", args.data_root.string()},
                {"deep1b_root", args.deep1b_root.string()},
                {"shared_coarse_protocol", true},
                {"rerank_depth", "all"},
        };
        if (args.config_path.has_value()) {
            run_meta["config_path"] = args.config_path->string();
        }
        if (args.epq_structure.has_value()) {
            run_meta["epq_structure"] = args.epq_structure->string();
        }
        if (args.json_out.has_value()) {
            run_meta["json_out"] = args.json_out->string();
        }
        const nlohmann::json dataset_meta = {
                {"name", ds.spec.name},
                {"dim", ds.d},
                {"base_rows", ds.nb},
                {"base_rows_full", ds.base_rows_full},
                {"query_rows", ds.xq.rows()},
                {"query_rows_full", ds.query_rows_full},
                {"train_rows", ds.xt.rows()},
                {"train_rows_full", ds.train_rows_full},
                {"gt_k", ds.gt_k},
                {"gt_rows_full", ds.gt_rows_full},
        };
        const nlohmann::json metadata =
                epq::benchmark_metadata::build_common_benchmark_metadata(
                        run_meta,
                        dataset_meta,
                        args.threads,
                        effective_threads,
                        args.epq_config.has_value() ? &*args.epq_config : nullptr);
        epq::benchmark_metadata::print_common_benchmark_metadata(
                std::cout, metadata);

        const Summary summary = run_benchmark(args, ds);
        if (args.json_out.has_value()) {
            const auto parent = args.json_out->parent_path();
            if (!parent.empty()) {
                std::filesystem::create_directories(parent);
            }
            std::ofstream out(*args.json_out);
            if (!out) {
                fail("failed to open json output path: " + args.json_out->string());
            }
            out << std::setw(2) << to_json(ds, args, summary, metadata) << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
