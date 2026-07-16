#include "epq/index_arepq.h"
#include "epq/index_bapq.h"
#include "epq/index_epq.h"
#include "epq/structure.h"
#include "epq/structure_builder.h"
#include "epq/training_config.h"

#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/ProductQuantizer.h>
#include <omp.h>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

using RowMatrixXf =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
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
    if (config.has_value()) {
        const auto sec_it = config->find(section);
        if (sec_it != config->end() && sec_it->is_object()) {
            const auto key_it = sec_it->find(key);
            if (key_it != sec_it->end() && key_it->is_number_integer()) {
                return key_it->get<int>();
            }
        }
    }
    return getenv_int_or(env_name, fallback);
}

struct AREPQTailConfig {
    int tail_bits = 8;
    int tail_stages = 1;
};

struct Args {
    std::filesystem::path bundle;
    std::filesystem::path output_dir = "mmeb_v2_bench/cpp_runs/latest";
    std::string target = "epq";
    int bits = 128;
    int topk = 10;
    std::vector<int> k_values{1, 5, 10};
    int threads = 0;
    int max_train_rows = 0;
    bool train_only = false;
    std::vector<std::string> tasks;
    std::optional<std::filesystem::path> config_path;
    std::optional<nlohmann::json> config;
    std::optional<std::filesystem::path> epq_structure;
    int epq_transform_niter = -1;
    int epq_kmeans_niter = -1;
    int epq_transform_kmeans_niter = -1;
    int bapq_subspace_dim = 4;
    int bapq_bmax = 12;
    int bapq_max_train_rows = 200000;
    int arepq_tail_bits = -1;
    int arepq_tail_stages = -1;
};

struct TaskData {
    std::string name;
    std::filesystem::path dir;
    int d = 0;
    RowMatrixXf train;
    RowMatrixXf corpus;
    RowMatrixXf queries;
    std::vector<std::vector<int>> labels;
    nlohmann::json manifest;
};

struct Metrics {
    std::map<std::string, double> values;
};

struct RunSummary {
    std::string task_name;
    std::string target;
    int bits = 0;
    int dim = 0;
    int n_queries = 0;
    int n_candidates = 0;
    int n_train_vectors = 0;
    bool train_reused = false;
    bool train_only = false;
    double train_time = 0.0;
    double add_time = 0.0;
    double search_time = 0.0;
    double qps = 0.0;
    Metrics metrics;
};

std::vector<int> parse_k_values(std::string_view raw) {
    std::vector<int> out;
    size_t start = 0;
    while (start <= raw.size()) {
        const size_t comma = raw.find(',', start);
        const auto piece = raw.substr(start, comma == std::string_view::npos
                                                     ? std::string_view::npos
                                                     : comma - start);
        if (!piece.empty()) {
            out.push_back(std::stoi(std::string(piece)));
        }
        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1;
    }
    if (out.empty()) {
        fail("--k-values must contain at least one positive integer");
    }
    for (const int k : out) {
        if (k <= 0) {
            fail("--k-values must be positive");
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg.starts_with("--bundle=")) {
            args.bundle = std::string(arg.substr(9));
        } else if (arg.starts_with("--output-dir=")) {
            args.output_dir = std::string(arg.substr(13));
        } else if (arg.starts_with("--target=")) {
            args.target = std::string(arg.substr(9));
        } else if (arg.starts_with("--bits=")) {
            args.bits = std::stoi(std::string(arg.substr(7)));
        } else if (arg.starts_with("--topk=")) {
            args.topk = std::stoi(std::string(arg.substr(7)));
        } else if (arg.starts_with("--k-values=")) {
            args.k_values = parse_k_values(arg.substr(11));
        } else if (arg.starts_with("--threads=")) {
            args.threads = std::stoi(std::string(arg.substr(10)));
        } else if (arg.starts_with("--max-train-rows=")) {
            args.max_train_rows = std::stoi(std::string(arg.substr(17)));
        } else if (arg.starts_with("--maxtrain=")) {
            args.max_train_rows = std::stoi(std::string(arg.substr(11)));
        } else if (arg == "--train-only") {
            args.train_only = true;
        } else if (arg.starts_with("--task=")) {
            args.tasks.push_back(std::string(arg.substr(7)));
        } else if (arg.starts_with("--config=")) {
            args.config_path = std::filesystem::path(std::string(arg.substr(9)));
            args.config = epq::load_json_file(*args.config_path);
        } else if (arg.starts_with("--epq-structure=")) {
            args.epq_structure = std::filesystem::path(std::string(arg.substr(16)));
        } else if (arg.starts_with("--epq-transform-niter=")) {
            args.epq_transform_niter = std::stoi(std::string(arg.substr(22)));
        } else if (arg.starts_with("--epq-kmeans-niter=")) {
            args.epq_kmeans_niter = std::stoi(std::string(arg.substr(19)));
        } else if (arg.starts_with("--epq-transform-kmeans-niter=")) {
            args.epq_transform_kmeans_niter =
                    std::stoi(std::string(arg.substr(29)));
        } else if (arg.starts_with("--bapq-subspace-dim=")) {
            args.bapq_subspace_dim = std::stoi(std::string(arg.substr(21)));
        } else if (arg.starts_with("--bapq-bmax=")) {
            args.bapq_bmax = std::stoi(std::string(arg.substr(12)));
        } else if (arg.starts_with("--bapq-max-train-rows=")) {
            args.bapq_max_train_rows = std::stoi(std::string(arg.substr(22)));
        } else if (arg.starts_with("--arepq-tail-bits=")) {
            args.arepq_tail_bits = std::stoi(std::string(arg.substr(19)));
        } else if (arg.starts_with("--arepq-tail-stages=")) {
            args.arepq_tail_stages = std::stoi(std::string(arg.substr(21)));
        } else if (arg == "--help" || arg == "-h") {
            fail(
                    "usage: mmeb_vector_benchmark --bundle=DIR --target=exact|pq|opq|rq|lsq|epq|repq|bapq|arepq "
                    "--bits=N [--config=PATH] [--task=NAME] [--topk=N] [--k-values=1,5,10] "
                    "[--threads=N] [--max-train-rows=N] [--train-only] [--output-dir=DIR]");
        } else {
            fail("unknown argument: " + std::string(arg));
        }
    }
    if (args.bundle.empty()) {
        fail("--bundle is required");
    }
    if (args.bits <= 0) {
        fail("--bits must be positive");
    }
    if (args.topk <= 0) {
        fail("--topk must be positive");
    }
    if (args.max_train_rows < 0) {
        fail("--max-train-rows must be non-negative");
    }
    const int max_metric_k = *std::max_element(args.k_values.begin(), args.k_values.end());
    args.topk = std::max(args.topk, max_metric_k);
    return args;
}

std::filesystem::path resolve_path(
        const std::filesystem::path& base,
        const std::string& raw) {
    std::filesystem::path path(raw);
    if (path.is_absolute()) {
        return path;
    }
    return base / path;
}

RowMatrixXf read_f32_matrix(
        const std::filesystem::path& path,
        int rows,
        int dim) {
    if (rows < 0 || dim <= 0) {
        fail("invalid matrix shape for " + path.string());
    }
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        fail("failed to open matrix: " + path.string());
    }
    const auto bytes = in.tellg();
    const size_t expected =
            static_cast<size_t>(rows) * static_cast<size_t>(dim) * sizeof(float);
    if (bytes < 0 || static_cast<size_t>(bytes) != expected) {
        fail("matrix byte size mismatch for " + path.string());
    }
    in.seekg(0);
    RowMatrixXf x(rows, dim);
    in.read(reinterpret_cast<char*>(x.data()), static_cast<std::streamsize>(expected));
    if (!in) {
        fail("failed to read matrix: " + path.string());
    }
    return x;
}

nlohmann::json read_json(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        fail("failed to open JSON: " + path.string());
    }
    nlohmann::json j;
    in >> j;
    return j;
}

bool task_selected(
        const nlohmann::json& task,
        const std::unordered_set<std::string>& selected) {
    if (selected.empty()) {
        return true;
    }
    const std::string task_name = task.value("task_name", "");
    const std::string task_dir = task.value("task_dir", "");
    return selected.contains(task_name) || selected.contains(task_dir);
}

std::vector<TaskData> load_tasks(const Args& args) {
    const auto metadata_path = args.bundle / "metadata.json";
    const auto metadata = read_json(metadata_path);
    if (metadata.value("format", "") != "mmeb-vector-bundle-v1") {
        fail("unsupported bundle metadata format: " + metadata_path.string());
    }
    std::unordered_set<std::string> selected(args.tasks.begin(), args.tasks.end());
    std::vector<TaskData> out;
    for (const auto& task_meta : metadata.at("tasks")) {
        if (!task_selected(task_meta, selected)) {
            continue;
        }
        const auto task_dir = args.bundle / task_meta.at("task_dir").get<std::string>();
        const auto manifest = read_json(task_dir / "manifest.json");
        TaskData task;
        task.name = manifest.at("task_name").get<std::string>();
        task.dir = task_dir;
        task.d = manifest.at("dim").get<int>();
        const auto& corpus = manifest.at("corpus");
        const auto& queries = manifest.at("queries");
        const auto& train = manifest.at("train");
        task.corpus = read_f32_matrix(
                resolve_path(task_dir, corpus.at("path").get<std::string>()),
                corpus.at("rows").get<int>(),
                corpus.at("dim").get<int>());
        task.queries = read_f32_matrix(
                resolve_path(task_dir, queries.at("path").get<std::string>()),
                queries.at("rows").get<int>(),
                queries.at("dim").get<int>());
        task.train = read_f32_matrix(
                resolve_path(task_dir, train.at("path").get<std::string>()),
                train.at("rows").get<int>(),
                train.at("dim").get<int>());
        for (const auto& row : manifest.at("labels")) {
            std::vector<int> labels;
            for (const auto& value : row) {
                labels.push_back(value.get<int>());
            }
            task.labels.push_back(std::move(labels));
        }
        if (task.queries.rows() != static_cast<Eigen::Index>(task.labels.size())) {
            fail("query/label row mismatch for task " + task.name);
        }
        if (task.corpus.cols() != task.d || task.queries.cols() != task.d ||
            task.train.cols() != task.d) {
            fail("matrix dim mismatch for task " + task.name);
        }
        task.manifest = manifest;
        out.push_back(std::move(task));
    }
    if (out.empty()) {
        fail("no bundle tasks matched the requested selection");
    }
    return out;
}

class VectorIndex {
   public:
    virtual ~VectorIndex() = default;
    virtual std::string name() const = 0;
    virtual void train(const RowMatrixXf& train) = 0;
    virtual void add(const RowMatrixXf& corpus) = 0;
    virtual void reset() = 0;
    virtual void search(
            const RowMatrixXf& queries,
            int k,
            std::vector<float>& distances,
            std::vector<faiss::idx_t>& labels) const = 0;
};

class ExactIndex final : public VectorIndex {
   public:
    explicit ExactIndex(int d) : index_(d, faiss::METRIC_INNER_PRODUCT) {}
    std::string name() const override {
        return "exact";
    }
    void train(const RowMatrixXf&) override {}
    void add(const RowMatrixXf& corpus) override {
        index_.add(corpus.rows(), corpus.data());
    }
    void reset() override {
        index_.reset();
    }
    void search(
            const RowMatrixXf& queries,
            int k,
            std::vector<float>& distances,
            std::vector<faiss::idx_t>& labels) const override {
        distances.resize(static_cast<size_t>(queries.rows()) * static_cast<size_t>(k));
        labels.resize(static_cast<size_t>(queries.rows()) * static_cast<size_t>(k));
        index_.search(queries.rows(), queries.data(), k, distances.data(), labels.data());
    }

   private:
    faiss::IndexFlat index_;
};

class FaissPQIndex final : public VectorIndex {
   public:
    FaissPQIndex(int d, int bits, bool opq)
            : d_(d),
              bits_(bits),
              m_(bits / 8),
              d2_(opq ? ((d + m_ - 1) / m_) * m_ : d),
              use_opq_(opq) {
        if (bits % 8 != 0 || m_ <= 0) {
            fail("pq/opq target requires bits divisible by 8");
        }
    }
    std::string name() const override {
        return use_opq_ ? "opq" : "pq";
    }
    void train(const RowMatrixXf& train) override {
        if (use_opq_) {
            auto* opq = new faiss::OPQMatrix(d_, m_, d2_);
            opq_ = opq;
            opq->niter = std::max(1, getenv_int_or("EPQ_OPQ_NITER", opq->niter));
            opq->niter_pq =
                    std::max(1, getenv_int_or("EPQ_OPQ_NITER_PQ", opq->niter_pq));
            opq->niter_pq_0 = std::max(
                    1,
                    getenv_int_or("EPQ_OPQ_NITER_PQ0", opq->niter_pq_0));
            opq->train(train.rows(), train.data());
            std::vector<float> transformed(
                    static_cast<size_t>(train.rows()) * static_cast<size_t>(d2_));
            opq->apply_noalloc(train.rows(), train.data(), transformed.data());
            Eigen::Map<const RowMatrixXf> mapped(transformed.data(), train.rows(), d2_);
            auto* pq = new faiss::IndexPQ(d2_, m_, 8, faiss::METRIC_L2);
            pq->train(mapped.rows(), mapped.data());
            auto pre = std::make_unique<faiss::IndexPreTransform>(opq, pq);
            pre->own_fields = true;
            index_ = std::move(pre);
            return;
        }
        index_ = std::make_unique<faiss::IndexPQ>(d_, m_, 8, faiss::METRIC_L2);
        index_->train(train.rows(), train.data());
    }
    void add(const RowMatrixXf& corpus) override {
        index_->add(corpus.rows(), corpus.data());
    }
    void reset() override {
        index_->reset();
    }
    void search(
            const RowMatrixXf& queries,
            int k,
            std::vector<float>& distances,
            std::vector<faiss::idx_t>& labels) const override {
        distances.resize(static_cast<size_t>(queries.rows()) * static_cast<size_t>(k));
        labels.resize(static_cast<size_t>(queries.rows()) * static_cast<size_t>(k));
        index_->search(queries.rows(), queries.data(), k, distances.data(), labels.data());
    }

   private:
    int d_ = 0;
    int bits_ = 0;
    int m_ = 0;
    int d2_ = 0;
    bool use_opq_ = false;
    faiss::OPQMatrix* opq_ = nullptr;
    std::unique_ptr<faiss::Index> index_;
};

class FaissPQTrainOnlyIndex final : public VectorIndex {
   public:
    FaissPQTrainOnlyIndex(int d, int bits, bool opq)
            : d_(d),
              bits_(bits),
              m_(bits / 8),
              d2_(opq ? ((d + m_ - 1) / m_) * m_ : d),
              use_opq_(opq) {
        if (bits % 8 != 0 || m_ <= 0) {
            fail("pq/opq train-only target requires bits divisible by 8");
        }
    }

    std::string name() const override {
        return use_opq_ ? "opq" : "pq";
    }

    void train(const RowMatrixXf& train) override {
        RowMatrixXf x_train = train;
        if (use_opq_) {
            opq_ = std::make_unique<faiss::OPQMatrix>(d_, m_, d2_);
            opq_->niter = std::max(1, getenv_int_or("EPQ_OPQ_NITER", opq_->niter));
            opq_->niter_pq = std::max(
                    1,
                    getenv_int_or("EPQ_OPQ_NITER_PQ", opq_->niter_pq));
            opq_->niter_pq_0 = std::max(
                    1,
                    getenv_int_or("EPQ_OPQ_NITER_PQ0", opq_->niter_pq_0));
            opq_->train(train.rows(), train.data());
            std::vector<float> transformed(
                    static_cast<size_t>(train.rows()) * static_cast<size_t>(d2_));
            opq_->apply_noalloc(train.rows(), train.data(), transformed.data());
            Eigen::Map<const RowMatrixXf> mapped(transformed.data(), train.rows(), d2_);
            x_train = mapped;
        } else {
            opq_.reset();
        }
        pq_ = std::make_unique<faiss::ProductQuantizer>(d2_, m_, 8);
        pq_->train(x_train.rows(), x_train.data());
    }

    void add(const RowMatrixXf&) override {
        fail(name() + " train-only index cannot add vectors");
    }

    void reset() override {}

    void search(
            const RowMatrixXf&,
            int,
            std::vector<float>&,
            std::vector<faiss::idx_t>&) const override {
        fail(name() + " train-only index cannot search");
    }

   private:
    int d_ = 0;
    int bits_ = 0;
    int m_ = 0;
    int d2_ = 0;
    bool use_opq_ = false;
    std::unique_ptr<faiss::OPQMatrix> opq_;
    std::unique_ptr<faiss::ProductQuantizer> pq_;
};

class EPQIndex final : public VectorIndex {
   public:
    EPQIndex(
            int d,
            int bits,
            bool use_transform,
            std::shared_ptr<epq::StructureBuilder> builder,
            const Args& args)
            : index_(d, bits, std::move(builder)),
              name_(use_transform ? "epq" : "repq") {
        if (args.config.has_value()) {
            epq::apply_index_training_config(index_, *args.config);
        }
        index_.use_uneven_transform = use_transform;
        if (args.epq_transform_niter >= 0) {
            index_.transform_niter = args.epq_transform_niter;
        }
        if (args.epq_kmeans_niter >= 0) {
            index_.kmeans_niter = args.epq_kmeans_niter;
        }
        if (args.epq_transform_kmeans_niter >= 0) {
            index_.transform_kmeans_niter = args.epq_transform_kmeans_niter;
        }
    }
    std::string name() const override {
        return name_;
    }
    void train(const RowMatrixXf& train) override {
        index_.train(train.rows(), train.data());
    }
    void add(const RowMatrixXf& corpus) override {
        index_.add(corpus.rows(), corpus.data());
    }
    void reset() override {
        index_.reset();
    }
    void search(
            const RowMatrixXf& queries,
            int k,
            std::vector<float>& distances,
            std::vector<faiss::idx_t>& labels) const override {
        distances.resize(static_cast<size_t>(queries.rows()) * static_cast<size_t>(k));
        labels.resize(static_cast<size_t>(queries.rows()) * static_cast<size_t>(k));
        epq::SearchParametersEPQ params;
        params.mode = epq::SearchMode::kADC;
        index_.search(
                queries.rows(),
                queries.data(),
                k,
                distances.data(),
                labels.data(),
                &params);
    }

   private:
    epq::IndexEPQ index_;
    std::string name_;
};

class BAPQIndex final : public VectorIndex {
   public:
    BAPQIndex(int d, int bits, const Args& args)
            : index_(d, bits, args.bapq_subspace_dim) {
        index_.bmax = args.bapq_bmax;
        index_.max_train_rows = args.bapq_max_train_rows;
    }
    std::string name() const override {
        return "bapq";
    }
    void train(const RowMatrixXf& train) override {
        index_.train(train.rows(), train.data());
    }
    void add(const RowMatrixXf& corpus) override {
        index_.add(corpus.rows(), corpus.data());
    }
    void reset() override {
        index_.reset();
    }
    void search(
            const RowMatrixXf& queries,
            int k,
            std::vector<float>& distances,
            std::vector<faiss::idx_t>& labels) const override {
        distances.resize(static_cast<size_t>(queries.rows()) * static_cast<size_t>(k));
        labels.resize(static_cast<size_t>(queries.rows()) * static_cast<size_t>(k));
        index_.search(queries.rows(), queries.data(), k, distances.data(), labels.data());
    }

   private:
    epq::IndexBAPQ index_;
};

std::shared_ptr<epq::StructureBuilder> make_epq_builder(const Args& args);

class FaissAdditiveQuantizerIndex final : public VectorIndex {
   public:
    enum class Kind {
        kRQ,
        kLSQ,
    };

    FaissAdditiveQuantizerIndex(int d, int bits, Kind kind)
            : d_(d), bits_(bits), kind_(kind), name_(kind == Kind::kRQ ? "rq" : "lsq") {
        if (bits_ % 8 != 0) {
            fail(name_ + " target requires bits divisible by 8");
        }
        code_size_bytes_ = bits_ / 8;
        if (code_size_bytes_ < 2) {
            fail(name_ + " target requires at least 16 total bits");
        }
        m_ = code_size_bytes_ - 1;
        if (m_ <= 0) {
            fail("invalid stage count for " + name_);
        }
    }

    std::string name() const override {
        return name_;
    }

    void train(const RowMatrixXf& train) override {
        constexpr auto search_type = faiss::AdditiveQuantizer::ST_norm_qint8;
        if (kind_ == Kind::kRQ) {
            auto index = std::make_unique<faiss::IndexResidualQuantizer>(
                    d_,
                    static_cast<size_t>(m_),
                    8,
                    faiss::METRIC_L2,
                    search_type);
            index->rq.max_beam_size = getenv_int_or("EPQ_RQ_MAX_BEAM_SIZE", 8);
            index_ = std::move(index);
        } else {
            index_ = std::make_unique<faiss::IndexLocalSearchQuantizer>(
                    d_,
                    static_cast<size_t>(m_),
                    8,
                    faiss::METRIC_L2,
                    search_type);
        }
        index_->train(train.rows(), train.data());
    }

    void add(const RowMatrixXf& corpus) override {
        index_->add(corpus.rows(), corpus.data());
    }
    void reset() override {
        index_->reset();
    }

    void search(
            const RowMatrixXf& queries,
            int k,
            std::vector<float>& distances,
            std::vector<faiss::idx_t>& labels) const override {
        distances.resize(static_cast<size_t>(queries.rows()) * static_cast<size_t>(k));
        labels.resize(static_cast<size_t>(queries.rows()) * static_cast<size_t>(k));
        index_->search(queries.rows(), queries.data(), k, distances.data(), labels.data());
    }

   private:
    int d_ = 0;
    int bits_ = 0;
    int code_size_bytes_ = 0;
    int m_ = 0;
    Kind kind_ = Kind::kRQ;
    std::string name_;
    std::unique_ptr<faiss::Index> index_;
};

AREPQTailConfig resolve_arepq_tail_config(const Args& args) {
    AREPQTailConfig cfg;
    cfg.tail_bits = args.arepq_tail_bits > 0
            ? args.arepq_tail_bits
            : std::max(
                      1,
                      get_config_int_or_env(
                              args.config,
                              "arepq",
                              "tail_bits",
                              "EPQ_AREPQ_TAIL_BITS",
                              8));
    cfg.tail_stages = args.arepq_tail_stages > 0
            ? args.arepq_tail_stages
            : std::max(
                      1,
                      get_config_int_or_env(
                              args.config,
                              "arepq",
                              "tail_stages",
                              "EPQ_AREPQ_TAIL_STAGES",
                              1));
    return cfg;
}

class AREPQIndex final : public VectorIndex {
   public:
    AREPQIndex(int d, int bits, const Args& args)
            : tail_(resolve_arepq_tail_config(args)),
              index_(d, bits, tail_.tail_bits, tail_.tail_stages, make_epq_builder(main_args(args))),
              name_("arepq") {
        if (bits <= tail_.tail_bits * tail_.tail_stages) {
            fail("arepq requires total bits larger than tail_bits * tail_stages");
        }
        if (args.config.has_value()) {
            epq::apply_index_training_config(index_.main_index(), *args.config);
        }
        if (args.epq_transform_niter >= 0) {
            index_.main_index().transform_niter = args.epq_transform_niter;
        }
        if (args.epq_kmeans_niter >= 0) {
            index_.main_index().kmeans_niter = args.epq_kmeans_niter;
        }
        if (args.epq_transform_kmeans_niter >= 0) {
            index_.main_index().transform_kmeans_niter =
                    args.epq_transform_kmeans_niter;
        }
        index_.icm_iters = std::max(0, getenv_int_or("EPQ_AREPQ_ICM_ITERS", 2));
        index_.final_main_reassign =
                getenv_int_or("EPQ_AREPQ_FINAL_MAIN_REASSIGN", 0) != 0;
        index_.skip_stable_tail_reassign =
                getenv_int_or("EPQ_AREPQ_SKIP_STABLE_TAIL_REASSIGN", 1) != 0;
        const int legacy_tail_refine_iters =
                getenv_int_or("EPQ_AREPQ_TAIL_REFINE_ITERS", 1);
        index_.tail_alt_iters = std::max(
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
                        args.config,
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

    void train(const RowMatrixXf& train) override {
        index_.train(train.rows(), train.data());
    }

    void add(const RowMatrixXf& corpus) override {
        index_.add(corpus.rows(), corpus.data());
    }
    void reset() override {
        index_.reset();
    }

    void search(
            const RowMatrixXf& queries,
            int k,
            std::vector<float>& distances,
            std::vector<faiss::idx_t>& labels) const override {
        distances.resize(static_cast<size_t>(queries.rows()) * static_cast<size_t>(k));
        labels.resize(static_cast<size_t>(queries.rows()) * static_cast<size_t>(k));
        epq::SearchParametersEPQ params;
        params.mode = epq::SearchMode::kADC;
        index_.search(
                queries.rows(),
                queries.data(),
                k,
                distances.data(),
                labels.data(),
                &params);
    }

   private:
    static Args main_args(const Args& args) {
        Args adjusted = args;
        const auto tail = resolve_arepq_tail_config(args);
        adjusted.bits = args.bits - tail.tail_bits * tail.tail_stages;
        return adjusted;
    }

    AREPQTailConfig tail_;
    epq::IndexAREPQ index_;
    std::string name_;
};

std::shared_ptr<epq::StructureBuilder> make_epq_builder(const Args& args) {
    if (args.epq_structure.has_value()) {
        auto structure = epq::Structure::load_json(args.epq_structure->string());
        return std::make_shared<epq::FixedStructureBuilder>(std::move(structure));
    }
    if (args.config.has_value()) {
        const auto base_dir = args.config_path.has_value()
                ? args.config_path->parent_path()
                : std::filesystem::path();
        return epq::make_structure_builder_from_config(*args.config, base_dir);
    }
    return std::make_shared<epq::RefinedStructureBuilder>();
}

std::unique_ptr<VectorIndex> make_index(const Args& args, int d) {
    if (args.target == "exact") {
        return std::make_unique<ExactIndex>(d);
    }
    if (args.target == "pq") {
        if (args.train_only) {
            return std::make_unique<FaissPQTrainOnlyIndex>(d, args.bits, false);
        }
        return std::make_unique<FaissPQIndex>(d, args.bits, false);
    }
    if (args.target == "opq") {
        if (args.train_only) {
            return std::make_unique<FaissPQTrainOnlyIndex>(d, args.bits, true);
        }
        return std::make_unique<FaissPQIndex>(d, args.bits, true);
    }
    if (args.target == "epq") {
        return std::make_unique<EPQIndex>(d, args.bits, true, make_epq_builder(args), args);
    }
    if (args.target == "repq") {
        return std::make_unique<EPQIndex>(d, args.bits, false, make_epq_builder(args), args);
    }
    if (args.target == "bapq") {
        return std::make_unique<BAPQIndex>(d, args.bits, args);
    }
    if (args.target == "rq") {
        return std::make_unique<FaissAdditiveQuantizerIndex>(
                d,
                args.bits,
                FaissAdditiveQuantizerIndex::Kind::kRQ);
    }
    if (args.target == "lsq") {
        return std::make_unique<FaissAdditiveQuantizerIndex>(
                d,
                args.bits,
                FaissAdditiveQuantizerIndex::Kind::kLSQ);
    }
    if (args.target == "arepq") {
        return std::make_unique<AREPQIndex>(d, args.bits, args);
    }
    fail("unsupported target: " + args.target);
}

Metrics evaluate_metrics(
        const std::vector<faiss::idx_t>& predictions,
        int nq,
        int topk,
        const std::vector<std::vector<int>>& labels,
        const std::vector<int>& k_values) {
    Metrics metrics;
    for (const int k : k_values) {
        double hit_sum = 0.0;
        double precision_sum = 0.0;
        double recall_sum = 0.0;
        double mrr_sum = 0.0;
        const int eval_k = std::min(k, topk);
        for (int qi = 0; qi < nq; ++qi) {
            std::unordered_set<int> label_set(labels[qi].begin(), labels[qi].end());
            int hits = 0;
            double rr = 0.0;
            for (int rank = 0; rank < eval_k; ++rank) {
                const auto id = predictions[static_cast<size_t>(qi) * topk + rank];
                if (id < 0) {
                    continue;
                }
                if (label_set.contains(static_cast<int>(id))) {
                    ++hits;
                    if (rr == 0.0) {
                        rr = 1.0 / static_cast<double>(rank + 1);
                    }
                }
            }
            hit_sum += hits > 0 ? 1.0 : 0.0;
            precision_sum += static_cast<double>(hits) / static_cast<double>(k);
            recall_sum += label_set.empty()
                    ? 0.0
                    : static_cast<double>(hits) / static_cast<double>(label_set.size());
            mrr_sum += rr;
        }
        const double denom = nq > 0 ? static_cast<double>(nq) : 1.0;
        metrics.values["hit@" + std::to_string(k)] = hit_sum / denom;
        metrics.values["precision@" + std::to_string(k)] = precision_sum / denom;
        metrics.values["recall@" + std::to_string(k)] = recall_sum / denom;
        metrics.values["mrr@" + std::to_string(k)] = mrr_sum / denom;
    }
    return metrics;
}

RowMatrixXf select_train_rows(const RowMatrixXf& train, int max_train_rows) {
    if (max_train_rows <= 0 ||
        train.rows() <= static_cast<Eigen::Index>(max_train_rows)) {
        return train;
    }
    return train.topRows(max_train_rows);
}

RunSummary run_one(
        const Args& args,
        VectorIndex& index,
        const TaskData& task,
        int train_rows_used,
        double shared_train_time,
        bool train_reused) {
    RunSummary summary;
    summary.task_name = task.name;
    summary.target = index.name();
    summary.bits = args.bits;
    summary.dim = task.d;
    summary.n_queries = static_cast<int>(task.queries.rows());
    summary.n_candidates = static_cast<int>(task.corpus.rows());
    summary.n_train_vectors = train_rows_used;
    summary.train_reused = train_reused;

    std::cout << "task=" << task.name << " target=" << summary.target
              << " d=" << task.d << " train=" << train_rows_used
              << " corpus=" << task.corpus.rows()
              << " queries=" << task.queries.rows()
              << " bits=" << args.bits
              << " train_reused=" << train_reused << '\n';

    auto t1 = std::chrono::steady_clock::now();
    index.add(task.corpus);
    auto t2 = std::chrono::steady_clock::now();
    const int effective_topk = std::min(args.topk, static_cast<int>(task.corpus.rows()));
    std::vector<float> distances;
    std::vector<faiss::idx_t> labels;
    index.search(task.queries, effective_topk, distances, labels);
    auto t3 = std::chrono::steady_clock::now();

    summary.train_time = shared_train_time;
    summary.add_time = std::chrono::duration<double>(t2 - t1).count();
    summary.search_time = std::chrono::duration<double>(t3 - t2).count();
    summary.qps = task.queries.rows() / std::max(summary.search_time, 1e-12);
    summary.metrics = evaluate_metrics(
            labels,
            static_cast<int>(task.queries.rows()),
            effective_topk,
            task.labels,
            args.k_values);
    index.reset();
    return summary;
}

nlohmann::json summary_to_json(const RunSummary& summary) {
    nlohmann::json j = {
            {"task", summary.task_name},
            {"target", summary.target},
            {"bits", summary.bits},
            {"dim", summary.dim},
            {"n_queries", summary.n_queries},
            {"n_candidates", summary.n_candidates},
            {"n_train_vectors", summary.n_train_vectors},
            {"train_reused", summary.train_reused},
            {"train_only", summary.train_only},
            {"train", summary.train_time},
            {"add", summary.add_time},
            {"search", summary.search_time},
            {"QPS", summary.qps},
    };
    for (const auto& [name, value] : summary.metrics.values) {
        j[name] = value;
    }
    return j;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (args.threads > 0) {
            omp_set_num_threads(args.threads);
            setenv("OMP_NUM_THREADS", std::to_string(args.threads).c_str(), 1);
            setenv("OPENBLAS_NUM_THREADS", "1", 1);
            setenv("MKL_NUM_THREADS", "1", 1);
            setenv("BLIS_NUM_THREADS", "1", 1);
            setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
        }
        if (args.config.has_value()) {
            epq::apply_faiss_runtime_config(*args.config);
        }
        std::filesystem::create_directories(args.output_dir);
        const auto tasks = load_tasks(args);
        const int d = tasks.front().d;
        for (const auto& task : tasks) {
            if (task.d != d) {
                fail("all tasks in one benchmark run must have the same embedding dim");
            }
        }
        auto index = make_index(args, d);
        const RowMatrixXf train = select_train_rows(tasks.front().train, args.max_train_rows);
        std::cout << "train_once target=" << index->name()
                  << " d=" << d
                  << " train=" << train.rows()
                  << " bits=" << args.bits << '\n';
        const auto train_t0 = std::chrono::steady_clock::now();
        index->train(train);
        const auto train_t1 = std::chrono::steady_clock::now();
        const double train_time =
                std::chrono::duration<double>(train_t1 - train_t0).count();
        nlohmann::json rows = nlohmann::json::array();
        if (args.train_only) {
            RunSummary summary;
            summary.task_name = "__train__";
            summary.target = index->name();
            summary.bits = args.bits;
            summary.dim = d;
            summary.n_train_vectors = static_cast<int>(train.rows());
            summary.train_only = true;
            summary.train_time = train_time;
            const auto row = summary_to_json(summary);
            rows.push_back(row);
            std::ofstream out(args.output_dir / "train.summary.json");
            out << row.dump(2) << '\n';
            std::ofstream summary_out(args.output_dir / "summary.json");
            summary_out << rows.dump(2) << '\n';
            std::cout << row.dump() << '\n';
            return 0;
        }
        bool first_task = true;
        for (const auto& task : tasks) {
            const auto summary = run_one(
                    args,
                    *index,
                    task,
                    static_cast<int>(train.rows()),
                    first_task ? train_time : 0.0,
                    !first_task);
            first_task = false;
            const auto row = summary_to_json(summary);
            rows.push_back(row);
            const auto out_path = args.output_dir / (task.name + ".summary.json");
            std::ofstream out(out_path);
            out << row.dump(2) << '\n';
            std::cout << row.dump() << '\n';
        }
        std::ofstream summary_out(args.output_dir / "summary.json");
        summary_out << rows.dump(2) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
