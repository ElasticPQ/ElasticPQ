#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct RecordHeader {
    int dim = 0;
    size_t count = 0;
};

struct Args {
    std::filesystem::path input = "data/sift1M/sift_learn.fvecs";
    int rows = 5000;
    int train_rows = 4000;
    int eval_rows = 1000;
    int repeats = 3;
    int niter = 8;
    int nredo = 1;
    int seed = 123;
    int min_points_per_centroid = 4;
    int blas_threshold = -1;
};

struct CaseSpec {
    const char* name;
    int dim = 0;
    int k = 0;
};

struct TimingSummary {
    double train_s = 0.0;
    double search_s = 0.0;
    double total_s = 0.0;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

RecordHeader inspect_fvecs(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail("failed to open " + path.string());
    }
    int dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(int));
    if (!in || dim <= 0) {
        fail("invalid fvecs header in " + path.string());
    }
    const auto bytes = std::filesystem::file_size(path);
    const size_t record_size =
            static_cast<size_t>(dim + 1) * sizeof(float);
    if (bytes % record_size != 0) {
        fail("unexpected fvecs size for " + path.string());
    }
    return RecordHeader{
            .dim = dim,
            .count = static_cast<size_t>(bytes / record_size),
    };
}

std::vector<float> load_fvecs_prefix(
        const std::filesystem::path& path,
        size_t rows,
        int* out_dim) {
    const RecordHeader header = inspect_fvecs(path);
    if (rows > header.count) {
        fail("requested rows exceed file rows");
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail("failed to open " + path.string());
    }
    std::vector<float> out(rows * static_cast<size_t>(header.dim));
    for (size_t i = 0; i < rows; ++i) {
        int dim = 0;
        in.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (!in || dim != header.dim) {
            fail("fvecs row dimension mismatch");
        }
        in.read(
                reinterpret_cast<char*>(out.data() +
                                        i * static_cast<size_t>(header.dim)),
                sizeof(float) * static_cast<size_t>(header.dim));
        if (!in) {
            fail("failed while reading row from " + path.string());
        }
    }
    *out_dim = header.dim;
    return out;
}

std::vector<float> slice_prefix_dims(
        const std::vector<float>& x,
        int rows,
        int src_dim,
        int dst_dim,
        int row_offset = 0) {
    std::vector<float> out(static_cast<size_t>(rows) * static_cast<size_t>(dst_dim));
    for (int i = 0; i < rows; ++i) {
        const float* src =
                x.data() + static_cast<size_t>(row_offset + i) * static_cast<size_t>(src_dim);
        float* dst = out.data() + static_cast<size_t>(i) * static_cast<size_t>(dst_dim);
        std::copy(src, src + dst_dim, dst);
    }
    return out;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if ((values.size() & 1U) != 0U) {
        return values[mid];
    }
    return 0.5 * (values[mid - 1] + values[mid]);
}

TimingSummary run_case(
        const std::vector<float>& xt,
        int train_rows,
        int eval_rows,
        int src_dim,
        const CaseSpec& cs,
        const Args& args) {
    std::vector<double> train_times;
    std::vector<double> search_times;
    std::vector<double> total_times;
    train_times.reserve(static_cast<size_t>(args.repeats));
    search_times.reserve(static_cast<size_t>(args.repeats));
    total_times.reserve(static_cast<size_t>(args.repeats));

    const auto x_train = slice_prefix_dims(xt, train_rows, src_dim, cs.dim, 0);
    const auto x_eval = slice_prefix_dims(xt, eval_rows, src_dim, cs.dim, train_rows);
    const int effective_k = std::min(cs.k, train_rows);

    for (int rep = 0; rep < args.repeats; ++rep) {
        faiss::ClusteringParameters cp;
        cp.niter = args.niter;
        cp.nredo = args.nredo;
        cp.seed = args.seed + 10007 * rep;
        cp.verbose = false;
        cp.min_points_per_centroid = args.min_points_per_centroid;

        const auto t0 = std::chrono::steady_clock::now();
        faiss::Clustering clustering(cs.dim, effective_k, cp);
        faiss::IndexFlatL2 assign_index(cs.dim);
        clustering.train(train_rows, x_train.data(), assign_index);
        const auto t1 = std::chrono::steady_clock::now();

        std::vector<float> centroids(
                static_cast<size_t>(cs.k) * static_cast<size_t>(cs.dim));
        const float* trained = clustering.centroids.data();
        for (int i = 0; i < effective_k; ++i) {
            std::copy(
                    trained + static_cast<size_t>(i) * static_cast<size_t>(cs.dim),
                    trained + static_cast<size_t>(i + 1) * static_cast<size_t>(cs.dim),
                    centroids.data() +
                            static_cast<size_t>(i) * static_cast<size_t>(cs.dim));
        }
        for (int i = effective_k; i < cs.k; ++i) {
            const int src = (effective_k - 1 + i) % effective_k;
            std::copy(
                    trained + static_cast<size_t>(src) * static_cast<size_t>(cs.dim),
                    trained + static_cast<size_t>(src + 1) * static_cast<size_t>(cs.dim),
                    centroids.data() +
                            static_cast<size_t>(i) * static_cast<size_t>(cs.dim));
        }

        faiss::IndexFlatL2 index(cs.dim);
        index.add(cs.k, centroids.data());
        std::vector<float> distances(static_cast<size_t>(eval_rows));
        std::vector<faiss::idx_t> labels(static_cast<size_t>(eval_rows));
        index.search(
                eval_rows,
                x_eval.data(),
                1,
                distances.data(),
                labels.data());
        const auto t2 = std::chrono::steady_clock::now();

        train_times.push_back(
                std::chrono::duration<double>(t1 - t0).count());
        search_times.push_back(
                std::chrono::duration<double>(t2 - t1).count());
        total_times.push_back(
                std::chrono::duration<double>(t2 - t0).count());
    }

    return TimingSummary{
            .train_s = median(std::move(train_times)),
            .search_s = median(std::move(search_times)),
            .total_s = median(std::move(total_times)),
    };
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        auto parse_int = [&](std::string_view prefix, int* out) -> bool {
            if (!arg.starts_with(prefix)) {
                return false;
            }
            *out = std::stoi(std::string(arg.substr(prefix.size())));
            return true;
        };
        if (parse_int("--rows=", &args.rows) ||
            parse_int("--train=", &args.train_rows) ||
            parse_int("--eval=", &args.eval_rows) ||
            parse_int("--repeats=", &args.repeats) ||
            parse_int("--niter=", &args.niter) ||
            parse_int("--nredo=", &args.nredo) ||
            parse_int("--seed=", &args.seed) ||
            parse_int("--min-points=", &args.min_points_per_centroid) ||
            parse_int("--blas-threshold=", &args.blas_threshold)) {
            continue;
        }
        if (arg.starts_with("--input=")) {
            args.input = std::filesystem::path(std::string(arg.substr(8)));
            continue;
        }
        fail("unknown argument: " + std::string(arg));
    }
    if (args.rows <= 0 || args.train_rows <= 0 || args.eval_rows <= 0 ||
        args.rows < args.train_rows + args.eval_rows || args.repeats <= 0) {
        fail("invalid rows/train/eval/repeats combination");
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (args.blas_threshold >= 0) {
            faiss::distance_compute_blas_threshold = args.blas_threshold;
        }
        int src_dim = 0;
        const auto xt =
                load_fvecs_prefix(args.input, static_cast<size_t>(args.rows), &src_dim);

        const std::vector<CaseSpec> cases = {
                {"d6_k128", 6, 128},
                {"d8_k512", 8, 512},
                {"d10_k1024", 10, 1024},
                {"d11_k2048", 11, 2048},
                {"d13_k4096", 13, 4096},
        };

        std::cout << "impl,case,dim,k_requested,k_effective,train_rows,eval_rows,repeats,niter,nredo,train_s,search_s,total_s\n";
        std::cout << std::fixed << std::setprecision(6);
        for (const auto& cs : cases) {
            const auto summary =
                    run_case(xt, args.train_rows, args.eval_rows, src_dim, cs, args);
            std::cout << "cpp"
                      << ',' << cs.name
                      << ',' << cs.dim
                      << ',' << cs.k
                      << ',' << std::min(cs.k, args.train_rows)
                      << ',' << args.train_rows
                      << ',' << args.eval_rows
                      << ',' << args.repeats
                      << ',' << args.niter
                      << ',' << args.nredo
                      << ',' << summary.train_s
                      << ',' << summary.search_s
                      << ',' << summary.total_s
                      << '\n';
        }
    } catch (const std::exception& ex) {
        std::cerr << "kmeans_microbench: " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
