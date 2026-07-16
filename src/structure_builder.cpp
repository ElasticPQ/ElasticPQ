#include "structure_builder_internal.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <Eigen/Eigenvalues>
#include <nlohmann/json.hpp>

namespace {

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    const std::string s(value);
    return !(s.empty() || s == "0" || s == "false" || s == "FALSE" ||
             s == "off" || s == "OFF" || s == "no" || s == "NO");
}

}  // namespace

namespace epq::structure_builder_internal {

int min_feasible_groups(const BuildContext& ctx) {
    if (ctx.total_bits <= 0) {
        return 1;
    }
    if (ctx.max_bits <= 0) {
        throw std::invalid_argument(
                "epq::StructureBuilder: max_bits must be positive when total_bits > 0");
    }
    return std::max(1, (ctx.total_bits + ctx.max_bits - 1) / ctx.max_bits);
}

void validate_build_context(const BuildContext& ctx) {
    if (ctx.d <= 0) {
        throw std::invalid_argument("epq::StructureBuilder: d must be positive");
    }
    if (ctx.total_bits < 0) {
        throw std::invalid_argument(
                "epq::StructureBuilder: total_bits must be non-negative");
    }
    if (ctx.min_bits < 0 || ctx.max_bits < ctx.min_bits) {
        throw std::invalid_argument("epq::StructureBuilder: invalid bit bounds");
    }
}

std::vector<int> distribute_bits_evenly(int total_bits, int groups) {
    std::vector<int> bits(static_cast<size_t>(groups), 0);
    if (groups <= 0) {
        return bits;
    }
    const int base = total_bits / groups;
    const int rem = total_bits % groups;
    for (int i = 0; i < groups; ++i) {
        bits[static_cast<size_t>(i)] = base + (i < rem ? 1 : 0);
    }
    return bits;
}

std::vector<std::vector<int>> balanced_groups(int d, int groups) {
    std::vector<std::vector<int>> out;
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

RowMatrixXf sample_rows(
        faiss::idx_t n,
        const float* x,
        int d,
        int max_rows,
        int seed) {
    Eigen::Map<const RowMatrixXf> all(x, static_cast<Eigen::Index>(n), d);
    if (max_rows <= 0 || n <= max_rows) {
        return all;
    }
    std::vector<faiss::idx_t> ids(static_cast<size_t>(n));
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(ids.begin(), ids.end(), rng);

    RowMatrixXf out(max_rows, d);
    for (int i = 0; i < max_rows; ++i) {
        out.row(i) = all.row(static_cast<Eigen::Index>(ids[static_cast<size_t>(i)]));
    }
    return out;
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

uint64_t stable_hash_dims(const std::vector<int>& dims) {
    uint64_t h = 1469598103934665603ULL;
    for (int v : dims) {
        uint32_t x = static_cast<uint32_t>(v);
        for (int i = 0; i < 4; ++i) {
            h ^= static_cast<uint8_t>((x >> (i * 8)) & 0xFFU);
            h *= 1099511628211ULL;
        }
    }
    return h;
}

double median_value(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<ptrdiff_t>(mid), values.end());
    double med = values[mid];
    if (values.size() % 2 == 0) {
        std::nth_element(
                values.begin(),
                values.begin() + static_cast<ptrdiff_t>(mid - 1),
                values.begin() + static_cast<ptrdiff_t>(mid));
        med = 0.5 * (med + values[mid - 1]);
    }
    return med;
}

RowMatrixXf train_kmeans_seeded(
        const RowMatrixXf& x,
        int k,
        int niter,
        int nredo,
        int seed,
        int min_points_per_centroid) {
    if (x.rows() <= 0 || x.cols() <= 0) {
        throw std::invalid_argument(
                "epq::StructureBuilder: cannot train k-means on empty matrix");
    }
    const int effective_k = std::min<int>(k, x.rows());
    faiss::ClusteringParameters cp;
    cp.niter = niter;
    cp.nredo = nredo;
    cp.seed = seed;
    cp.verbose = false;
    cp.min_points_per_centroid = min_points_per_centroid;
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

double kmeans_recon_mse_holdout(
        const RowMatrixXf& x_train,
        const RowMatrixXf& x_eval,
        int k,
        int niter,
        int nredo,
        int seed,
        int min_points_per_centroid) {
    if (x_train.rows() <= 0 || x_train.cols() <= 0 || x_eval.rows() <= 0) {
        return 0.0;
    }
    const int kk = std::min<int>(k, std::max<int>(1, x_train.rows()));
    if (kk == 1) {
        const Eigen::RowVectorXf mu = x_train.colwise().mean();
        const RowMatrixXf diff = x_eval.rowwise() - mu;
        return diff.rowwise().squaredNorm().mean();
    }

    std::vector<double> mses;
    mses.reserve(static_cast<size_t>(std::max(1, nredo)));
    for (int r = 0; r < std::max(1, nredo); ++r) {
        const int run_seed = seed + 10007 * r;
        const RowMatrixXf codebook = train_kmeans_seeded(
                x_train,
                kk,
                niter,
                1,
                run_seed,
                min_points_per_centroid);
        faiss::IndexFlatL2 index(codebook.cols());
        index.add(codebook.rows(), codebook.data());
        std::vector<float> distances(static_cast<size_t>(x_eval.rows()));
        std::vector<faiss::idx_t> labels(static_cast<size_t>(x_eval.rows()));
        index.search(
                x_eval.rows(),
                x_eval.data(),
                1,
                distances.data(),
                labels.data());
        double mse = 0.0;
        for (float value : distances) {
            mse += static_cast<double>(value);
        }
        mses.push_back(mse / static_cast<double>(distances.size()));
    }
    return median_value(std::move(mses));
}

ProxyPcaSlice build_pca_approx_slice(
        const RowMatrixXf& x_train,
        const RowMatrixXf& x_eval,
        int top_dims) {
    ProxyPcaSlice slice;
    slice.full_dims = static_cast<int>(x_train.cols());
    slice.proj_dims = std::min(top_dims, slice.full_dims);
    if (slice.proj_dims <= 0 || slice.proj_dims >= slice.full_dims) {
        slice.train_proj = x_train;
        slice.eval_proj = x_eval;
        slice.tail_eval_mse = 0.0;
        slice.proj_dims = slice.full_dims;
        return slice;
    }

    const Eigen::RowVectorXf mu = x_train.colwise().mean();
    const RowMatrixXf centered_train = x_train.rowwise() - mu;
    const RowMatrixXf centered_eval = x_eval.rowwise() - mu;
    const float denom = std::max(1.0f, static_cast<float>(x_train.rows() - 1));
    const Eigen::MatrixXf cov =
            (centered_train.adjoint() * centered_train) / denom;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(cov);
    if (eig.info() != Eigen::Success) {
        throw std::runtime_error(
                "epq::StructureBuilder: PCA eigendecomposition failed");
    }

    const Eigen::MatrixXf basis = eig.eigenvectors().rightCols(slice.proj_dims);
    slice.train_proj = centered_train * basis;
    slice.eval_proj = centered_eval * basis;
    const double total_eval_energy =
            static_cast<double>(centered_eval.rowwise().squaredNorm().mean());
    const double head_eval_energy =
            static_cast<double>(slice.eval_proj.rowwise().squaredNorm().mean());
    slice.tail_eval_mse = std::max(0.0, total_eval_energy - head_eval_energy);
    return slice;
}

TrainEvalSplit split_train_eval_rows(
        const RowMatrixXf& x,
        int max_train,
        int max_eval,
        float eval_frac,
        int seed) {
    const int n0 = static_cast<int>(x.rows());
    if (n0 <= 1) {
        return {.train = x, .eval = x};
    }

    std::mt19937 rng(seed);
    const int want_train = std::max(1, max_train);
    const int want_eval = std::max(1, max_eval);

    if (n0 >= want_train + want_eval) {
        std::vector<int> perm(static_cast<size_t>(n0));
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        RowMatrixXf train(want_train, x.cols());
        RowMatrixXf eval(want_eval, x.cols());
        for (int i = 0; i < want_train; ++i) {
            train.row(i) = x.row(perm[static_cast<size_t>(i)]);
        }
        for (int i = 0; i < want_eval; ++i) {
            eval.row(i) = x.row(perm[static_cast<size_t>(want_train + i)]);
        }
        return {.train = std::move(train), .eval = std::move(eval)};
    }

    float frac = std::clamp(eval_frac, 0.05f, 0.5f);
    std::vector<int> perm(static_cast<size_t>(n0));
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    int ne = std::max(1, static_cast<int>(std::lround(n0 * frac)));
    ne = std::min(ne, n0 - 1);
    int nt = n0 - ne;
    nt = std::min(nt, want_train);
    ne = std::min(ne, want_eval);

    RowMatrixXf train(nt, x.cols());
    RowMatrixXf eval(ne, x.cols());
    for (int i = 0; i < ne; ++i) {
        eval.row(i) = x.row(perm[static_cast<size_t>(i)]);
    }
    for (int i = 0; i < nt; ++i) {
        train.row(i) = x.row(perm[static_cast<size_t>(ne + i)]);
    }
    return {.train = std::move(train), .eval = std::move(eval)};
}

void validate_partition(
        const Groups& groups,
        int d,
        bool require_cover,
        bool allow_empty_group) {
    if (groups.empty()) {
        throw std::invalid_argument("epq::StructureBuilder: groups is empty");
    }
    std::vector<char> seen(static_cast<size_t>(d), 0);
    int count = 0;
    for (const auto& group : groups) {
        if (!allow_empty_group && group.empty()) {
            throw std::invalid_argument(
                    "epq::StructureBuilder: groups contains an empty group");
        }
        for (int dim : group) {
            if (dim < 0 || dim >= d) {
                throw std::invalid_argument(
                        "epq::StructureBuilder: invalid dimension id in groups");
            }
            if (seen[static_cast<size_t>(dim)]) {
                throw std::invalid_argument(
                        "epq::StructureBuilder: duplicated dimension in groups");
            }
            seen[static_cast<size_t>(dim)] = 1;
            ++count;
        }
    }
    if (require_cover && count != d) {
        throw std::invalid_argument(
                "epq::StructureBuilder: groups must cover all dimensions");
    }
}

PartitionKey canonical_partition_key(const Groups& groups) {
    PartitionKey parts;
    parts.reserve(groups.size());
    for (auto group : groups) {
        std::sort(group.begin(), group.end());
        parts.push_back(std::move(group));
    }
    std::sort(parts.begin(), parts.end());
    return parts;
}

std::vector<int> remove_one(const std::vector<int>& group, int v) {
    std::vector<int> out;
    out.reserve(group.size());
    bool removed = false;
    for (int dim : group) {
        if (!removed && dim == v) {
            removed = true;
            continue;
        }
        out.push_back(dim);
    }
    return out;
}

Structure make_structure(
        const Groups& groups,
        const Bits& bits,
        const BuildContext& ctx,
        const std::string& builder_name) {
    if (groups.size() != bits.size()) {
        throw std::invalid_argument(
                "epq::StructureBuilder: groups/bits size mismatch");
    }
    Structure structure;
    structure.d = ctx.d;
    structure.total_bits = ctx.total_bits;
    structure.meta = {{"builder", builder_name}};
    for (size_t i = 0; i < groups.size(); ++i) {
        auto dims = groups[i];
        std::sort(dims.begin(), dims.end());
        structure.groups.push_back(GroupSpec{std::move(dims), bits[i]});
    }
    structure.validate(ctx.min_bits, ctx.max_bits);
    return structure;
}

#if EPQ_ENABLE_STRUCTURE_TRACE
namespace {

int env_int_or_default(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

std::string compact_stage_name(const std::string& stage) {
    std::string out;
    out.reserve(stage.size());
    for (char ch : stage) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') || ch == '_' || ch == '-') {
            out.push_back(ch);
        } else {
            out.push_back('_');
        }
    }
    return out.empty() ? "stage" : out;
}

struct StructureTraceWriter {
    bool initialized = false;
    bool enabled = false;
    std::filesystem::path dir;
    std::ofstream manifest;
    uint64_t seen = 0;
    uint64_t written = 0;
    int stride = 1;
    int limit = 64;

    void init_once() {
        if (initialized) {
            return;
        }
        initialized = true;
        const char* dir_env = std::getenv("EPQ_STRUCTURE_TRACE_DIR");
        if (dir_env == nullptr || dir_env[0] == '\0') {
            return;
        }
        dir = std::filesystem::path(dir_env);
        stride = std::max(1, env_int_or_default("EPQ_STRUCTURE_TRACE_STRIDE", 1));
        limit = env_int_or_default("EPQ_STRUCTURE_TRACE_LIMIT", 64);
        std::filesystem::create_directories(dir);
        manifest.open(dir / "manifest.jsonl", std::ios::app);
        if (!manifest) {
            throw std::runtime_error(
                    "epq::StructureBuilder: failed to open structure trace manifest");
        }
        enabled = true;
    }

    bool should_write() {
        init_once();
        if (!enabled) {
            return false;
        }
        const uint64_t index = seen++;
        if (index % static_cast<uint64_t>(stride) != 0) {
            return false;
        }
        if (limit > 0 && written >= static_cast<uint64_t>(limit)) {
            return false;
        }
        return true;
    }

    void write(
            const BuildContext& ctx,
            const std::string& stage,
            int step,
            const Groups& groups,
            const Bits& bits,
            double j_star,
            const std::string& source) {
        if (!should_write()) {
            return;
        }
        const uint64_t id = written++;
        std::ostringstream stem;
        stem << "structure_" << std::setw(4) << std::setfill('0') << id << "_"
             << compact_stage_name(stage) << "_" << step << ".json";
        const auto path = dir / stem.str();

        Structure structure = make_structure(groups, bits, ctx, "TraceStructure");
        structure.meta["trace"] = {
                {"id", id},
                {"stage", stage},
                {"step", step},
                {"source", source},
                {"j_star", j_star},
                {"groups", groups.size()},
                {"total_bits", ctx.total_bits},
                {"min_bits", ctx.min_bits},
                {"max_bits", ctx.max_bits},
        };
        structure.save_json(path.string());

        std::vector<int> group_sizes;
        group_sizes.reserve(groups.size());
        for (const auto& group : groups) {
            group_sizes.push_back(static_cast<int>(group.size()));
        }
        const nlohmann::json record = {
                {"id", id},
                {"stage", stage},
                {"step", step},
                {"source", source},
                {"j_star", j_star},
                {"groups", groups.size()},
                {"group_sizes", group_sizes},
                {"bits", bits},
                {"path", path.string()},
        };
        manifest << record.dump() << '\n';
        manifest.flush();
    }
};

StructureTraceWriter& structure_trace_writer() {
    static StructureTraceWriter writer;
    return writer;
}

}  // namespace

void trace_structure_candidate(
        const BuildContext& ctx,
        const std::string& stage,
        int step,
        const Groups& groups,
        const Bits& bits,
        double j_star,
        const std::string& source) {
    structure_trace_writer().write(ctx, stage, step, groups, bits, j_star, source);
}
#endif

namespace {

double proxy_D_impl(
        ProxyContext& proxy,
        const std::vector<int>& dims,
        int bits,
        bool fast_mode) {
    auto& d_calls = fast_mode ? proxy.work_stats.d_fast_calls : proxy.work_stats.d_calls;
    auto& d_empty_calls = fast_mode ? proxy.work_stats.d_fast_empty_calls
                                    : proxy.work_stats.d_empty_calls;
    auto& d_hits = fast_mode ? proxy.cache_stats.d_fast_hits : proxy.cache_stats.d_hits;
    auto& d_misses = fast_mode ? proxy.cache_stats.d_fast_misses
                               : proxy.cache_stats.d_misses;
    auto& pca_hits = fast_mode ? proxy.cache_stats.pca_fast_hits
                               : proxy.cache_stats.pca_hits;
    auto& pca_misses = fast_mode ? proxy.cache_stats.pca_fast_misses
                                 : proxy.cache_stats.pca_misses;
    auto& kmeans_calls = fast_mode ? proxy.work_stats.kmeans_fast_calls
                                   : proxy.work_stats.kmeans_calls;
    auto& kmeans_k_total = fast_mode ? proxy.work_stats.kmeans_fast_k_total
                                     : proxy.work_stats.kmeans_k_total;
    auto& kmeans_dims_total = fast_mode ? proxy.work_stats.kmeans_fast_dims_total
                                        : proxy.work_stats.kmeans_dims_total;
    auto& kmeans_train_rows_total =
            fast_mode ? proxy.work_stats.kmeans_fast_train_rows_total
                      : proxy.work_stats.kmeans_train_rows_total;
    auto& kmeans_eval_rows_total =
            fast_mode ? proxy.work_stats.kmeans_fast_eval_rows_total
                      : proxy.work_stats.kmeans_eval_rows_total;
    auto& pca_approx_calls = fast_mode ? proxy.work_stats.pca_fast_approx_calls
                                       : proxy.work_stats.pca_approx_calls;
    auto& pca_fits = fast_mode ? proxy.work_stats.pca_fast_fits
                               : proxy.work_stats.pca_fits;
    auto& pca_full_dims_total = fast_mode ? proxy.work_stats.pca_fast_full_dims_total
                                          : proxy.work_stats.pca_full_dims_total;
    auto& pca_proj_dims_total = fast_mode ? proxy.work_stats.pca_fast_proj_dims_total
                                          : proxy.work_stats.pca_proj_dims_total;
    auto& pca_tail_dims_total = fast_mode ? proxy.work_stats.pca_fast_tail_dims_total
                                          : proxy.work_stats.pca_tail_dims_total;
    auto& d_cache = fast_mode ? proxy.d_fast_cache : proxy.d_cache;
    auto& pca_cache = fast_mode ? proxy.pca_fast_cache : proxy.pca_cache;
    const int pca_top_dims = fast_mode ? proxy.fast_pca_top_dims : proxy.pca_top_dims;

    ++d_calls;
    if (dims.empty()) {
        ++d_empty_calls;
        return 0.0;
    }
    if (bits < 0 || bits > proxy.build_ctx.max_bits) {
        throw std::invalid_argument("epq::StructureBuilder: proxy bits out of range");
    }
    DimsKey key_dims = dims;
    std::sort(key_dims.begin(), key_dims.end());
    const auto key = std::make_pair(key_dims, bits);
    if (const double* hit = d_cache.get(key)) {
        ++d_hits;
        return *hit;
    }
    ++d_misses;

    RowMatrixXf xtr_local;
    RowMatrixXf xev_local;
    const RowMatrixXf* xtr = nullptr;
    const RowMatrixXf* xev = nullptr;

    const bool use_slice_cache =
            proxy.cache_slices &&
            proxy.xtr_cache.enabled() &&
            proxy.xev_cache.enabled();
    if (use_slice_cache) {
        xtr = proxy.xtr_cache.get(key_dims);
        if (xtr != nullptr) {
            ++proxy.cache_stats.xtr_hits;
        } else {
            ++proxy.cache_stats.xtr_misses;
            proxy.xtr_cache.set(key_dims, gather_columns(proxy.xt_train, key_dims));
            xtr = proxy.xtr_cache.get(key_dims);
        }

        xev = proxy.xev_cache.get(key_dims);
        if (xev != nullptr) {
            ++proxy.cache_stats.xev_hits;
        } else {
            ++proxy.cache_stats.xev_misses;
            proxy.xev_cache.set(key_dims, gather_columns(proxy.xt_eval, key_dims));
            xev = proxy.xev_cache.get(key_dims);
        }
    } else {
        ++proxy.cache_stats.xtr_misses;
        ++proxy.cache_stats.xev_misses;
        xtr_local = gather_columns(proxy.xt_train, key_dims);
        xev_local = gather_columns(proxy.xt_eval, key_dims);
        xtr = &xtr_local;
        xev = &xev_local;
    }

    const RowMatrixXf* xtr_km = xtr;
    const RowMatrixXf* xev_km = xev;
    double tail_eval_mse = 0.0;
    if (pca_top_dims > 0 && xtr->cols() > pca_top_dims) {
        const ProxyPcaSlice* pca = nullptr;
        ProxyPcaSlice pca_local;
        if (pca_cache.max_size > 0) {
            pca = pca_cache.get(key_dims);
            if (pca != nullptr) {
                ++pca_hits;
            } else {
                ++pca_misses;
                ++pca_fits;
                pca_cache.set(key_dims, build_pca_approx_slice(*xtr, *xev, pca_top_dims));
                pca = pca_cache.get(key_dims);
            }
        } else {
            ++pca_misses;
            ++pca_fits;
            pca_local = build_pca_approx_slice(*xtr, *xev, pca_top_dims);
            pca = &pca_local;
        }
        xtr_km = &pca->train_proj;
        xev_km = &pca->eval_proj;
        tail_eval_mse = pca->tail_eval_mse;
        ++pca_approx_calls;
        pca_full_dims_total += static_cast<uint64_t>(pca->full_dims);
        pca_proj_dims_total += static_cast<uint64_t>(pca->proj_dims);
        pca_tail_dims_total +=
                static_cast<uint64_t>(pca->full_dims - pca->proj_dims);
    }

    const int run_seed = proxy.seed +
            static_cast<int>(stable_hash_dims(key_dims) % 100000ULL) +
            7919 * bits;
    const int k = 1 << bits;
    ++kmeans_calls;
    kmeans_k_total += static_cast<uint64_t>(k);
    kmeans_dims_total += static_cast<uint64_t>(xtr_km->cols());
    kmeans_train_rows_total += static_cast<uint64_t>(xtr_km->rows());
    kmeans_eval_rows_total += static_cast<uint64_t>(xev_km->rows());
    const double value = kmeans_recon_mse_holdout(
            *xtr_km,
            *xev_km,
            k,
            proxy.km_niter,
            proxy.km_nredo,
            run_seed,
            proxy.min_points_per_centroid) +
            tail_eval_mse;
    d_cache.set(key, value);
    return value;
}

}  // namespace

double ProxyContext::D(const std::vector<int>& dims, int bits) {
    return proxy_D_impl(*this, dims, bits, false);
}

double ProxyContext::D_fast(const std::vector<int>& dims, int bits) {
    if (fast_pca_top_dims <= 0) {
        return D(dims, bits);
    }
    return proxy_D_impl(*this, dims, bits, true);
}

BitAllocResult ProxyContext::solve_bits(
        const Groups& groups,
        bool allow_partial) {
    ++work_stats.solve_bits_calls;
    validate_partition(groups, build_ctx.d, !allow_partial);
    const int M = static_cast<int>(groups.size());
    work_stats.solve_bits_groups_total += static_cast<uint64_t>(M);
    const int B = build_ctx.total_bits;
    const int min_bits = build_ctx.min_bits;
    const int bmax = build_ctx.max_bits;
    if (B < M * min_bits || B > M * bmax) {
        throw std::runtime_error("epq::StructureBuilder: infeasible bit budget");
    }
    const int rem_budget = B - M * min_bits;
    constexpr double kInf = 1e100;
    std::vector<std::vector<double>> dp(
            static_cast<size_t>(M + 1),
            std::vector<double>(static_cast<size_t>(rem_budget + 1), kInf));
    std::vector<std::vector<int>> choice(
            static_cast<size_t>(M + 1),
            std::vector<int>(static_cast<size_t>(rem_budget + 1), -1));
    dp[0][0] = 0.0;
    for (int i = 1; i <= M; ++i) {
        const auto& group = groups[static_cast<size_t>(i - 1)];
        const int cap = bmax - min_bits;
        work_stats.solve_bits_cost_evals += static_cast<uint64_t>(cap + 1);
        std::vector<double> costs(static_cast<size_t>(cap + 1), 0.0);
        for (int extra = 0; extra <= cap; ++extra) {
            costs[static_cast<size_t>(extra)] = D(group, min_bits + extra);
        }
        for (int t = 0; t <= rem_budget; ++t) {
            const double base = dp[static_cast<size_t>(i - 1)][static_cast<size_t>(t)];
            if (base >= kInf / 2) {
                continue;
            }
            ++work_stats.solve_bits_dp_states;
            const int max_add = std::min(cap, rem_budget - t);
            work_stats.solve_bits_dp_transitions +=
                    static_cast<uint64_t>(max_add + 1);
            for (int add = 0; add <= max_add; ++add) {
                const int tt = t + add;
                const double value = base + costs[static_cast<size_t>(add)];
                if (value < dp[static_cast<size_t>(i)][static_cast<size_t>(tt)]) {
                    dp[static_cast<size_t>(i)][static_cast<size_t>(tt)] = value;
                    choice[static_cast<size_t>(i)][static_cast<size_t>(tt)] = add;
                }
            }
        }
    }
    double J = dp[static_cast<size_t>(M)][static_cast<size_t>(rem_budget)];
    if (!std::isfinite(J) || J >= kInf / 2) {
        throw std::runtime_error("epq::StructureBuilder: DP allocation failed");
    }
    Bits bits(static_cast<size_t>(M), min_bits);
    int t = rem_budget;
    for (int i = M; i >= 1; --i) {
        const int add = choice[static_cast<size_t>(i)][static_cast<size_t>(t)];
        if (add < 0) {
            throw std::runtime_error("epq::StructureBuilder: DP backtrack failed");
        }
        bits[static_cast<size_t>(i - 1)] += add;
        t -= add;
    }
    return {.J = J, .bits = std::move(bits)};
}

bool group_stats_env_enabled() {
    return env_flag_enabled("EPQ_PRINT_GROUP_STATS");
}

void print_group_proxy_stats(
        std::ostream& os,
        const std::string& quantizer_name,
        const std::string& space_label,
        const Groups& groups,
        const Bits& bits,
        const BuildContext& ctx,
        ProxyContext& proxy) {
    if (groups.size() != bits.size()) {
        throw std::invalid_argument(
                "epq::StructureBuilder: group stats groups/bits size mismatch");
    }
    int total_dims = 0;
    int total_bits = 0;
    for (size_t i = 0; i < groups.size(); ++i) {
        total_dims += static_cast<int>(groups[i].size());
        total_bits += bits[i];
    }

    const auto old_flags = os.flags();
    const auto old_precision = os.precision();
    os << "\t[group-stats] quantizer=" << quantizer_name
       << " space=" << space_label
       << " entries=" << groups.size()
       << " total_dims=" << total_dims
       << " total_bits=" << total_bits << '\n';

    double j_proxy = 0.0;
    for (size_t i = 0; i < groups.size(); ++i) {
        const double d_proxy = proxy.D(groups[i], bits[i]);
        j_proxy += d_proxy;
        os << "\t[group-stats] group[" << std::setw(3) << std::setfill('0')
           << i << std::setfill(' ')
           << "] ndims=" << groups[i].size()
           << " bits=" << bits[i]
           << " D_proxy=" << std::fixed << std::setprecision(6)
           << d_proxy << '\n';
    }
    os << "\t[group-stats] J_proxy=" << std::fixed << std::setprecision(6)
       << j_proxy << '\n';
    os.flags(old_flags);
    os.precision(old_precision);

    (void)ctx;
}

void print_group_proxy_stats_from_matrix(
        std::ostream& os,
        const std::string& quantizer_name,
        const std::string& space_label,
        const Groups& groups,
        const Bits& bits,
        const RowMatrixXf& xt,
        const BuildContext& ctx,
        int proxy_max_train_rows,
        int proxy_max_eval_rows,
        float proxy_eval_frac,
        int proxy_kmeans_niter,
        int proxy_kmeans_nredo,
        int proxy_min_points_per_centroid,
        int seed) {
    const auto split = split_train_eval_rows(
            xt,
            proxy_max_train_rows,
            proxy_max_eval_rows,
            proxy_eval_frac,
            seed);
    ProxyContext proxy{
            .build_ctx = ctx,
            .xt_train = split.train,
            .xt_eval = split.eval,
            .km_niter = proxy_kmeans_niter,
            .km_nredo = proxy_kmeans_nredo,
            .min_points_per_centroid = proxy_min_points_per_centroid,
            .seed = seed,
    };
    print_group_proxy_stats(
            os,
            quantizer_name,
            space_label,
            groups,
            bits,
            ctx,
            proxy);
}

int score_bits_for_group(
        int d,
        int B,
        int proxy_bmax,
        int fixed_bits) {
    (void)d;
    (void)B;
    return std::clamp(fixed_bits, 0, proxy_bmax);
}

std::vector<std::vector<std::pair<int, float>>> build_dim_neighbors_by_corr_weighted(
        const RowMatrixXf& xt,
        int knn,
        bool abs_corr,
        int max_rows,
        int seed,
        float edge_tau) {
    RowMatrixXf x = xt;
    if (max_rows > 0 && x.rows() > max_rows) {
        std::vector<int> ids(static_cast<size_t>(x.rows()));
        std::iota(ids.begin(), ids.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(ids.begin(), ids.end(), rng);
        RowMatrixXf sample(max_rows, x.cols());
        for (int i = 0; i < max_rows; ++i) {
            sample.row(i) = x.row(ids[static_cast<size_t>(i)]);
        }
        x = std::move(sample);
    }

    const int n = static_cast<int>(x.rows());
    const int d = static_cast<int>(x.cols());
    if (d <= 1) {
        return std::vector<std::vector<std::pair<int, float>>>(
                static_cast<size_t>(d));
    }
    const int kk = std::min(knn, std::max(1, d - 1));
    if (n <= 1) {
        std::vector<std::vector<std::pair<int, float>>> out(
                static_cast<size_t>(d));
        const int half = std::max(1, kk / 2);
        for (int i = 0; i < d; ++i) {
            for (int j = std::max(0, i - half); j < std::min(d, i + half + 1);
                 ++j) {
                if (j != i) {
                    out[static_cast<size_t>(i)].push_back({j, 1.0f});
                }
            }
        }
        return out;
    }

    RowMatrixXf centered = x;
    const Eigen::RowVectorXf mean = centered.colwise().mean();
    centered.rowwise() -= mean;
    Eigen::RowVectorXf stddev =
            centered.array().square().colwise().mean().array().sqrt();
    for (int i = 0; i < stddev.size(); ++i) {
        if (!(stddev(i) > 0.0f)) {
            stddev(i) = 1e-6f;
        }
    }
    centered.array().rowwise() /= stddev.array();
    RowMatrixXf corr =
            (centered.transpose() * centered) / static_cast<float>(n);
    corr = corr.array().max(-1.0f).min(1.0f);

    std::vector<std::vector<std::pair<int, float>>> out(
            static_cast<size_t>(d));
    const float tau = std::max(0.0f, edge_tau);
    for (int i = 0; i < d; ++i) {
        std::vector<std::pair<int, float>> neighbors;
        neighbors.reserve(static_cast<size_t>(d - 1));
        for (int j = 0; j < d; ++j) {
            if (i == j) {
                continue;
            }
            float score = corr(i, j);
            if (abs_corr) {
                score = std::abs(score);
            }
            if (tau > 0.0f && score < tau) {
                continue;
            }
            neighbors.push_back({j, score});
        }
        std::partial_sort(
                neighbors.begin(),
                neighbors.begin() + std::min<int>(kk, neighbors.size()),
                neighbors.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.second > rhs.second;
                });
        if (static_cast<int>(neighbors.size()) > kk) {
            neighbors.resize(static_cast<size_t>(kk));
        }
        out[static_cast<size_t>(i)] = std::move(neighbors);
    }
    return out;
}

std::vector<std::vector<int>> build_dim_neighbors_by_corr(
        const RowMatrixXf& xt,
        int knn,
        bool abs_corr,
        int max_rows,
        int seed) {
    const auto weighted = build_dim_neighbors_by_corr_weighted(
            xt,
            knn,
            abs_corr,
            max_rows,
            seed,
            0.0f);
    std::vector<std::vector<int>> out(weighted.size());
    for (size_t i = 0; i < weighted.size(); ++i) {
        for (const auto& [j, _] : weighted[i]) {
            out[i].push_back(j);
        }
    }
    return out;
}

Groups singleton_groups(int d) {
    Groups groups;
    groups.reserve(static_cast<size_t>(d));
    for (int i = 0; i < d; ++i) {
        groups.push_back({i});
    }
    return groups;
}

std::vector<int> greedy_allocate_bits(
        const std::vector<float>& weights,
        const BuildContext& ctx) {
    const int groups = static_cast<int>(weights.size());
    if (groups <= 0) {
        throw std::invalid_argument(
                "epq::StructureBuilder: no groups to allocate bits");
    }
    if (ctx.min_bits > 0 && ctx.total_bits < groups * ctx.min_bits) {
        throw std::invalid_argument(
                "epq::StructureBuilder: total_bits smaller than groups * min_bits");
    }

    std::vector<int> bits(static_cast<size_t>(groups), ctx.min_bits);
    int remaining = ctx.total_bits - groups * ctx.min_bits;
    while (remaining > 0) {
        int best = -1;
        double best_gain = -1.0;
        for (int i = 0; i < groups; ++i) {
            if (bits[static_cast<size_t>(i)] >= ctx.max_bits) {
                continue;
            }
            const double scale =
                    std::ldexp(1.0, bits[static_cast<size_t>(i)] + 1);
            const double gain =
                    static_cast<double>(weights[static_cast<size_t>(i)]) / scale;
            if (gain > best_gain) {
                best_gain = gain;
                best = i;
            }
        }
        if (best < 0) {
            throw std::invalid_argument(
                    "epq::StructureBuilder: failed to allocate requested bit budget");
        }
        ++bits[static_cast<size_t>(best)];
        --remaining;
    }
    return bits;
}

}  // namespace epq::structure_builder_internal

namespace epq {
namespace sbi = structure_builder_internal;

FixedStructureBuilder::FixedStructureBuilder(Structure structure)
        : structure_(std::move(structure)) {
    structure_.validate();
}

const Structure& FixedStructureBuilder::structure() const noexcept {
    return structure_;
}

Structure FixedStructureBuilder::build(
        faiss::idx_t n,
        const float* x,
        const BuildContext& ctx) const {
    sbi::validate_build_context(ctx);
    if (structure_.d != ctx.d) {
        throw std::invalid_argument("epq::FixedStructureBuilder: dimension mismatch");
    }
    if (structure_.total_bits != ctx.total_bits) {
        throw std::invalid_argument("epq::FixedStructureBuilder: bit-budget mismatch");
    }
    structure_.validate(ctx.min_bits, ctx.max_bits);
    if (sbi::group_stats_env_enabled()) {
        if (n <= 0 || x == nullptr) {
            throw std::invalid_argument(
                    "epq::FixedStructureBuilder: training data is empty");
        }
        const Eigen::Map<const sbi::RowMatrixXf> xt(
                x,
                static_cast<Eigen::Index>(n),
                ctx.d);
        sbi::Groups groups;
        sbi::Bits bits;
        groups.reserve(structure_.groups.size());
        bits.reserve(structure_.groups.size());
        for (const auto& group : structure_.groups) {
            groups.push_back(group.dims);
            bits.push_back(group.nbits);
        }
        sbi::print_group_proxy_stats_from_matrix(
                std::cout,
                name(),
                "main",
                groups,
                bits,
                xt,
                ctx);
    }
    return structure_;
}

std::unique_ptr<StructureBuilder> FixedStructureBuilder::clone() const {
    return std::make_unique<FixedStructureBuilder>(structure_);
}

std::string FixedStructureBuilder::name() const {
    return "FixedStructureBuilder";
}

Structure BalancedStructureBuilder::build(
        faiss::idx_t,
        const float*,
        const BuildContext& ctx) const {
    sbi::validate_build_context(ctx);

    const int min_groups = sbi::min_feasible_groups(ctx);
    const int nominal = std::max(1, nominal_group_bits);
    int groups =
            target_groups > 0 ? target_groups : std::max(1, ctx.total_bits / nominal);
    groups = std::max(groups, min_groups);
    groups = std::min(groups, ctx.d);
    if (ctx.min_bits > 0) {
        groups = std::min(groups, std::max(1, ctx.total_bits / ctx.min_bits));
        groups = std::max(groups, min_groups);
    }

    const auto group_dims = sbi::balanced_groups(ctx.d, groups);
    const auto bits = sbi::distribute_bits_evenly(ctx.total_bits, groups);
    return sbi::make_structure(group_dims, bits, ctx, name());
}

std::unique_ptr<StructureBuilder> BalancedStructureBuilder::clone() const {
    return std::make_unique<BalancedStructureBuilder>(*this);
}

std::string BalancedStructureBuilder::name() const {
    return "BalancedStructureBuilder";
}

Structure VarianceAwareStructureBuilder::build(
        faiss::idx_t n,
        const float* x,
        const BuildContext& ctx) const {
    sbi::validate_build_context(ctx);
    if (n <= 0 || x == nullptr) {
        throw std::invalid_argument(
                "epq::VarianceAwareStructureBuilder: training data is empty");
    }

    const int min_groups_needed = sbi::min_feasible_groups(ctx);
    int groups = target_groups > 0
            ? target_groups
            : static_cast<int>(
                      std::lround(alpha_groups * std::max(1, ctx.total_bits / 8)));
    groups = std::max(groups, std::max(1, min_groups_needed));
    if (max_groups > 0) {
        groups = std::min(groups, max_groups);
    }
    groups = std::max(groups, std::min(min_groups, ctx.d));
    groups = std::min(groups, ctx.d);
    if (ctx.min_bits > 0) {
        groups = std::min(groups, std::max(1, ctx.total_bits / ctx.min_bits));
        groups = std::max(groups, min_groups_needed);
    }

    const sbi::RowMatrixXf sample = sbi::sample_rows(n, x, ctx.d, corr_sample_rows, seed);
    sbi::RowMatrixXf centered = sample;
    const Eigen::RowVectorXf mean = centered.colwise().mean();
    centered.rowwise() -= mean;
    Eigen::RowVectorXf var = centered.array().square().colwise().mean();
    for (int i = 0; i < var.size(); ++i) {
        if (!(var(i) > 0.0f)) {
            var(i) = 1e-6f;
        }
    }

    const Eigen::RowVectorXf stddev = var.array().sqrt();
    sbi::RowMatrixXf normed = centered;
    normed.array().rowwise() /= stddev.array();
    sbi::RowMatrixXf corr =
            (normed.transpose() * normed) / static_cast<float>(normed.rows());
    if (abs_correlation) {
        corr = corr.cwiseAbs();
    }
    corr.diagonal().setZero();

    std::vector<int> order(static_cast<size_t>(ctx.d));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return var(lhs) > var(rhs);
    });

    std::vector<std::vector<int>> group_dims(static_cast<size_t>(groups));
    std::vector<float> group_weight(static_cast<size_t>(groups), 0.0f);
    std::vector<char> assigned(static_cast<size_t>(ctx.d), 0);

    int seeds = 0;
    for (int dim : order) {
        if (seeds >= groups) {
            break;
        }
        group_dims[static_cast<size_t>(seeds)].push_back(dim);
        group_weight[static_cast<size_t>(seeds)] = var(dim);
        assigned[static_cast<size_t>(dim)] = 1;
        ++seeds;
    }

    for (int dim : order) {
        if (assigned[static_cast<size_t>(dim)]) {
            continue;
        }
        int best_group = 0;
        float best_score = -1e30f;
        for (int g = 0; g < groups; ++g) {
            const auto& dims = group_dims[static_cast<size_t>(g)];
            float mean_corr = 0.0f;
            for (int other : dims) {
                mean_corr += corr(dim, other);
            }
            mean_corr /= static_cast<float>(dims.size());
            const float score =
                    mean_corr - size_penalty * static_cast<float>(dims.size());
            if (score > best_score) {
                best_score = score;
                best_group = g;
            }
        }
        group_dims[static_cast<size_t>(best_group)].push_back(dim);
        group_weight[static_cast<size_t>(best_group)] += var(dim);
        assigned[static_cast<size_t>(dim)] = 1;
    }

    const auto bits = sbi::greedy_allocate_bits(group_weight, ctx);
    Structure structure = sbi::make_structure(group_dims, bits, ctx, name());
    structure.meta["alpha_groups"] = alpha_groups;
    structure.meta["corr_sample_rows"] = corr_sample_rows;
    return structure;
}

std::unique_ptr<StructureBuilder> VarianceAwareStructureBuilder::clone() const {
    return std::make_unique<VarianceAwareStructureBuilder>(*this);
}

std::string VarianceAwareStructureBuilder::name() const {
    return "VarianceAwareStructureBuilder";
}

Structure RefinedStructureBuilder::build(
        faiss::idx_t n,
        const float* x,
        const BuildContext& ctx) const {
    sbi::validate_build_context(ctx);
    if (n <= 0 || x == nullptr) {
        throw std::invalid_argument(
                "epq::RefinedStructureBuilder: training data is empty");
    }
    const Eigen::Map<const sbi::RowMatrixXf> xt(x, static_cast<Eigen::Index>(n), ctx.d);
    const auto split = sbi::split_train_eval_rows(
            xt,
            proxy_max_train_rows,
            proxy_max_eval_rows,
            proxy_eval_frac,
            seed);
    sbi::ProxyContext proxy{
            .build_ctx = ctx,
            .xt_train = split.train,
            .xt_eval = split.eval,
            .km_niter = proxy_kmeans_niter,
            .km_nredo = proxy_kmeans_nredo,
            .min_points_per_centroid = proxy_min_points_per_centroid,
            .seed = seed,
    };
    proxy.cache_slices = proxy_cache_slices;
    proxy.pca_top_dims = std::max(0, proxy_pca_top_dims);
    proxy.fast_pca_top_dims = std::max(
            0,
            std::max(
                    chain_tail_fast_proxy_top_dims,
                    crystallize_fast_proxy_top_dims));
    proxy.d_cache.max_size = static_cast<size_t>(std::max(0, proxy_max_d_cache));
    proxy.d_fast_cache.max_size = static_cast<size_t>(std::max(0, proxy_max_d_cache));
    const size_t slice_cache_bytes = static_cast<size_t>(proxy_max_slice_cache_bytes);
    proxy.xtr_cache.max_weight = slice_cache_bytes / 2;
    proxy.xev_cache.max_weight = slice_cache_bytes - proxy.xtr_cache.max_weight;
    proxy.pca_cache.max_size =
            static_cast<size_t>(std::max(0, proxy_max_pca_cache));
    proxy.pca_fast_cache.max_size =
            static_cast<size_t>(std::max(0, proxy_max_pca_cache));
    EPQ_STRUCTURE_DEBUG_LOG(
            1,
            "proxy train_rows=" << proxy.xt_train.rows()
                                << " eval_rows=" << proxy.xt_eval.rows()
                                << " km_niter=" << proxy.km_niter
                                << " km_nredo=" << proxy.km_nredo
                                << " pca_top_dims=" << proxy.pca_top_dims
                                << " fast_pca_top_dims=" << proxy.fast_pca_top_dims
                                << " cache_slices=" << (proxy.cache_slices ? "on" : "off")
                                << " max_d_cache=" << proxy.d_cache.max_size
                                << " max_slice_cache_bytes="
                                << (proxy.xtr_cache.max_weight + proxy.xev_cache.max_weight)
                                << " max_pca_cache=" << proxy.pca_cache.max_size);
    const auto pipeline_t0 = std::chrono::steady_clock::now();
    sbi::Groups groups;
    sbi::Bits bits;
    nlohmann::json stages_meta = nlohmann::json::object();
    const bool run_chain_tail = use_chain_tail;
    const bool run_greedy_tail = use_greedy_tail && !run_chain_tail;
    const bool run_mbeam = use_mbeam && !run_greedy_tail && !run_chain_tail;
    if (use_grow) {
        const auto t0 = std::chrono::steady_clock::now();
        std::tie(groups, bits) = sbi::run_grow_stage(*this, proxy, ctx);
        const auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        stages_meta["grow_time"] = seconds;
        EPQ_STRUCTURE_DEBUG_LOG(
                1,
                "grow groups=" << groups.size()
                               << " time=" << seconds << " s");
    } else {
        groups = sbi::singleton_groups(ctx.d);
        bits = proxy.solve_bits(groups).bits;
        stages_meta["grow_time"] = 0.0;
    }
    if (use_crystallize) {
        const auto t0 = std::chrono::steady_clock::now();
        std::tie(groups, bits) =
                sbi::run_crystallize_stage(*this, proxy, ctx, groups, bits);
        const auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        stages_meta["crystallize_time"] = seconds;
        EPQ_STRUCTURE_DEBUG_LOG(
                1,
                "crystallize groups=" << groups.size()
                                      << " time=" << seconds << " s");
    } else {
        stages_meta["crystallize_time"] = 0.0;
    }
    if (run_greedy_tail) {
        const auto t0 = std::chrono::steady_clock::now();
        std::tie(groups, bits) =
                sbi::run_greedy_tail_stage(*this, proxy, ctx, groups, bits);
        const auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        stages_meta["greedy_tail_time"] = seconds;
        EPQ_STRUCTURE_DEBUG_LOG(
                1,
                "greedy_tail groups=" << groups.size() << " time=" << seconds
                                    << " s");
    } else {
        stages_meta["greedy_tail_time"] = 0.0;
    }
    if (run_chain_tail) {
        const auto t0 = std::chrono::steady_clock::now();
        std::tie(groups, bits) =
                sbi::run_chain_tail_stage(*this, proxy, ctx, groups, bits);
        const auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        stages_meta["chain_tail_time"] = seconds;
        EPQ_STRUCTURE_DEBUG_LOG(
                1,
                "chain_tail groups=" << groups.size() << " time=" << seconds
                                     << " s");
    } else {
        stages_meta["chain_tail_time"] = 0.0;
    }
    if (run_mbeam) {
        const auto t0 = std::chrono::steady_clock::now();
        std::tie(groups, bits) =
                sbi::run_mbeam_stage(*this, proxy, ctx, groups, bits);
        const auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        stages_meta["mbeam_time"] = seconds;
        EPQ_STRUCTURE_DEBUG_LOG(
                1,
                "mbeam groups=" << groups.size() << " time=" << seconds
                                << " s");
    } else {
        stages_meta["mbeam_time"] = 0.0;
    }
    const auto pipeline_t1 = std::chrono::steady_clock::now();
    Structure structure = sbi::make_structure(groups, bits, ctx, name());
    structure.meta["stages"] = {
            {"grow", use_grow},
            {"crystallize", use_crystallize},
            {"mbeam", run_mbeam},
            {"greedy_tail", run_greedy_tail},
            {"chain_tail", run_chain_tail},
    };
    structure.meta["tail_refine_stage"] =
            run_chain_tail ? "chain_tail"
                           : (run_greedy_tail ? "greedy_tail"
                                              : (run_mbeam ? "mbeam" : "none"));
    structure.meta["stage_times"] = std::move(stages_meta);
    structure.meta["pipeline_time"] =
            std::chrono::duration<double>(pipeline_t1 - pipeline_t0).count();
    structure.meta["proxy_train_rows"] = proxy.xt_train.rows();
    structure.meta["proxy_eval_rows"] = proxy.xt_eval.rows();
    structure.meta["proxy_cache"] = {
            {"cache_slices", proxy.cache_slices},
            {"d_hits", proxy.cache_stats.d_hits},
            {"d_misses", proxy.cache_stats.d_misses},
            {"d_size", proxy.d_cache.size()},
            {"d_capacity", proxy.d_cache.max_size},
            {"d_fast_hits", proxy.cache_stats.d_fast_hits},
            {"d_fast_misses", proxy.cache_stats.d_fast_misses},
            {"d_fast_size", proxy.d_fast_cache.size()},
            {"d_fast_capacity", proxy.d_fast_cache.max_size},
            {"xtr_hits", proxy.cache_stats.xtr_hits},
            {"xtr_misses", proxy.cache_stats.xtr_misses},
            {"xtr_size", proxy.xtr_cache.size()},
            {"xtr_bytes", proxy.xtr_cache.weight()},
            {"xtr_capacity_bytes", proxy.xtr_cache.max_weight},
            {"xev_hits", proxy.cache_stats.xev_hits},
            {"xev_misses", proxy.cache_stats.xev_misses},
            {"xev_size", proxy.xev_cache.size()},
            {"xev_bytes", proxy.xev_cache.weight()},
            {"xev_capacity_bytes", proxy.xev_cache.max_weight},
            {"pca_top_dims", proxy.pca_top_dims},
            {"pca_hits", proxy.cache_stats.pca_hits},
            {"pca_misses", proxy.cache_stats.pca_misses},
            {"pca_size", proxy.pca_cache.size()},
            {"pca_capacity", proxy.pca_cache.max_size},
            {"pca_fast_top_dims", proxy.fast_pca_top_dims},
            {"pca_fast_hits", proxy.cache_stats.pca_fast_hits},
            {"pca_fast_misses", proxy.cache_stats.pca_fast_misses},
            {"pca_fast_size", proxy.pca_fast_cache.size()},
            {"pca_fast_capacity", proxy.pca_fast_cache.max_size},
    };
    structure.meta["proxy_work"] = {
            {"d_calls", proxy.work_stats.d_calls},
            {"d_empty_calls", proxy.work_stats.d_empty_calls},
            {"d_fast_calls", proxy.work_stats.d_fast_calls},
            {"d_fast_empty_calls", proxy.work_stats.d_fast_empty_calls},
            {"kmeans_calls", proxy.work_stats.kmeans_calls},
            {"kmeans_k_total", proxy.work_stats.kmeans_k_total},
            {"kmeans_dims_total", proxy.work_stats.kmeans_dims_total},
            {"kmeans_train_rows_total", proxy.work_stats.kmeans_train_rows_total},
            {"kmeans_eval_rows_total", proxy.work_stats.kmeans_eval_rows_total},
            {"kmeans_fast_calls", proxy.work_stats.kmeans_fast_calls},
            {"kmeans_fast_k_total", proxy.work_stats.kmeans_fast_k_total},
            {"kmeans_fast_dims_total", proxy.work_stats.kmeans_fast_dims_total},
            {"kmeans_fast_train_rows_total", proxy.work_stats.kmeans_fast_train_rows_total},
            {"kmeans_fast_eval_rows_total", proxy.work_stats.kmeans_fast_eval_rows_total},
            {"pca_approx_calls", proxy.work_stats.pca_approx_calls},
            {"pca_fits", proxy.work_stats.pca_fits},
            {"pca_full_dims_total", proxy.work_stats.pca_full_dims_total},
            {"pca_proj_dims_total", proxy.work_stats.pca_proj_dims_total},
            {"pca_tail_dims_total", proxy.work_stats.pca_tail_dims_total},
            {"pca_fast_approx_calls", proxy.work_stats.pca_fast_approx_calls},
            {"pca_fast_fits", proxy.work_stats.pca_fast_fits},
            {"pca_fast_full_dims_total", proxy.work_stats.pca_fast_full_dims_total},
            {"pca_fast_proj_dims_total", proxy.work_stats.pca_fast_proj_dims_total},
            {"pca_fast_tail_dims_total", proxy.work_stats.pca_fast_tail_dims_total},
            {"solve_bits_calls", proxy.work_stats.solve_bits_calls},
            {"solve_bits_groups_total", proxy.work_stats.solve_bits_groups_total},
            {"solve_bits_cost_evals", proxy.work_stats.solve_bits_cost_evals},
            {"solve_bits_dp_states", proxy.work_stats.solve_bits_dp_states},
            {"solve_bits_dp_transitions", proxy.work_stats.solve_bits_dp_transitions},
    };
    if (proxy.chain_tail_profile.used) {
        structure.meta["chain_tail_profile"] = {
                {"iterations", proxy.chain_tail_profile.iterations},
                {"iters_with_candidates", proxy.chain_tail_profile.iters_with_candidates},
                {"seeds_raw_total", proxy.chain_tail_profile.seeds_raw_total},
                {"seeds_kept_total", proxy.chain_tail_profile.seeds_kept_total},
                {"candidates_total", proxy.chain_tail_profile.candidates_total},
                {"exact_local_reranked_total",
                 proxy.chain_tail_profile.exact_local_reranked_total},
                {"exact_local_kept_total", proxy.chain_tail_profile.exact_local_kept_total},
                {"local_gate_pruned_total",
                 proxy.chain_tail_profile.local_gate_pruned_total},
                {"donor_small_stops_total",
                 proxy.chain_tail_profile.donor_small_stops_total},
                {"no_step_stops_total", proxy.chain_tail_profile.no_step_stops_total},
                {"prefix_cut_stops_total",
                 proxy.chain_tail_profile.prefix_cut_stops_total},
                {"total_steps", proxy.chain_tail_profile.total_steps},
                {"max_steps", proxy.chain_tail_profile.max_steps},
                {"exact_attempted", proxy.chain_tail_profile.exact_attempted},
                {"exact_children", proxy.chain_tail_profile.exact_children},
                {"exact_dup_pruned", proxy.chain_tail_profile.exact_dup_pruned},
                {"exact_seen_pruned", proxy.chain_tail_profile.exact_seen_pruned},
                {"improved_iters", proxy.chain_tail_profile.improved_iters},
        };
    }
    EPQ_STRUCTURE_DEBUG_LOG(
            1,
            "pipeline groups=" << structure.groups.size()
                               << " time=" << structure.meta["pipeline_time"].get<double>()
                               << " s");
    if (sbi::group_stats_env_enabled()) {
        sbi::print_group_proxy_stats(
                std::cout,
                name(),
                "main",
                groups,
                bits,
                ctx,
                proxy);
    }
    return structure;
}

std::unique_ptr<StructureBuilder> RefinedStructureBuilder::clone() const {
    return std::make_unique<RefinedStructureBuilder>(*this);
}

std::string RefinedStructureBuilder::name() const {
    return "RefinedStructureBuilder";
}

}  // namespace epq
