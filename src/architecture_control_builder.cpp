#include "structure_builder_internal.h"

#include "epq/structure.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace sbi = epq::structure_builder_internal;

namespace {

struct Args {
    fs::path input;
    fs::path learned_structure;
    fs::path output_dir;
    int total_bits = 128;
    int min_bits = 0;
    int max_bits = 12;
    std::vector<int> grid_groups = {11, 12, 13, 14, 15, 16};
    int proxy_max_train_rows = 16384;
    int proxy_max_eval_rows = 4096;
    float proxy_eval_frac = 0.2f;
    int proxy_kmeans_niter = 8;
    int proxy_kmeans_nredo = 1;
    int proxy_min_points_per_centroid = 4;
    int seed = 123;
};

struct FvecsData {
    int d = 0;
    size_t rows = 0;
    std::vector<float> values;
};

struct PairSpec {
    int dims = 0;
    int bits = 0;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

std::vector<int> parse_ints(std::string_view text) {
    std::vector<int> values;
    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t end = text.find(',', begin);
        const auto token = text.substr(
                begin,
                end == std::string_view::npos ? text.size() - begin : end - begin);
        if (!token.empty()) {
            values.push_back(std::stoi(std::string(token)));
        }
        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1;
    }
    return values;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg.starts_with("--input=")) {
            args.input = std::string(arg.substr(8));
        } else if (arg.starts_with("--learned-structure=")) {
            args.learned_structure = std::string(arg.substr(20));
        } else if (arg.starts_with("--output-dir=")) {
            args.output_dir = std::string(arg.substr(13));
        } else if (arg.starts_with("--bits=")) {
            args.total_bits = std::stoi(std::string(arg.substr(7)));
        } else if (arg.starts_with("--min-bits=")) {
            args.min_bits = std::stoi(std::string(arg.substr(11)));
        } else if (arg.starts_with("--max-bits=")) {
            args.max_bits = std::stoi(std::string(arg.substr(11)));
        } else if (arg.starts_with("--grid-groups=")) {
            args.grid_groups = parse_ints(arg.substr(14));
        } else if (arg.starts_with("--proxy-train=")) {
            args.proxy_max_train_rows = std::stoi(std::string(arg.substr(14)));
        } else if (arg.starts_with("--proxy-eval=")) {
            args.proxy_max_eval_rows = std::stoi(std::string(arg.substr(13)));
        } else if (arg.starts_with("--proxy-eval-frac=")) {
            args.proxy_eval_frac = std::stof(std::string(arg.substr(18)));
        } else if (arg.starts_with("--proxy-kmeans-niter=")) {
            args.proxy_kmeans_niter = std::stoi(std::string(arg.substr(21)));
        } else if (arg.starts_with("--proxy-kmeans-nredo=")) {
            args.proxy_kmeans_nredo = std::stoi(std::string(arg.substr(21)));
        } else if (arg.starts_with("--proxy-min-points=")) {
            args.proxy_min_points_per_centroid =
                    std::stoi(std::string(arg.substr(19)));
        } else if (arg.starts_with("--seed=")) {
            args.seed = std::stoi(std::string(arg.substr(7)));
        } else {
            fail("unknown argument: " + std::string(arg));
        }
    }
    if (args.input.empty() || args.learned_structure.empty() ||
        args.output_dir.empty()) {
        fail("usage: architecture_control_builder --input=FILE "
             "--learned-structure=FILE --output-dir=DIR [--bits=N] "
             "[--grid-groups=M1,M2,...]");
    }
    if (args.grid_groups.empty()) {
        fail("--grid-groups must not be empty");
    }
    return args;
}

FvecsData load_fvecs(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail("failed to open " + path.string());
    }
    int d = 0;
    in.read(reinterpret_cast<char*>(&d), sizeof(int));
    if (!in || d <= 0) {
        fail("invalid fvecs header in " + path.string());
    }
    const size_t record_bytes = static_cast<size_t>(d + 1) * sizeof(float);
    const size_t file_bytes = fs::file_size(path);
    if (file_bytes % record_bytes != 0) {
        fail("invalid fvecs file size for " + path.string());
    }
    const size_t rows = file_bytes / record_bytes;
    in.clear();
    in.seekg(0);
    std::vector<float> values(rows * static_cast<size_t>(d));
    for (size_t row = 0; row < rows; ++row) {
        int row_d = 0;
        in.read(reinterpret_cast<char*>(&row_d), sizeof(int));
        if (!in || row_d != d) {
            fail("inconsistent fvecs dimension in " + path.string());
        }
        in.read(
                reinterpret_cast<char*>(
                        values.data() + row * static_cast<size_t>(d)),
                static_cast<std::streamsize>(sizeof(float) * static_cast<size_t>(d)));
        if (!in) {
            fail("short fvecs read from " + path.string());
        }
    }
    return FvecsData{.d = d, .rows = rows, .values = std::move(values)};
}

std::vector<PairSpec> canonical_pairs(
        const sbi::Groups& groups,
        const sbi::Bits& bits) {
    if (groups.size() != bits.size()) {
        fail("group/bit size mismatch");
    }
    std::vector<PairSpec> pairs;
    pairs.reserve(groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
        pairs.push_back(PairSpec{
                .dims = static_cast<int>(groups[i].size()),
                .bits = bits[i],
        });
    }
    std::sort(
            pairs.begin(),
            pairs.end(),
            [](const PairSpec& lhs, const PairSpec& rhs) {
                return std::tie(lhs.dims, lhs.bits) >
                        std::tie(rhs.dims, rhs.bits);
            });
    return pairs;
}

epq::Structure structure_from_pairs(
        const std::vector<PairSpec>& pairs,
        const epq::BuildContext& ctx,
        const std::string& variant,
        double proxy_j) {
    sbi::Groups groups;
    sbi::Bits bits;
    groups.reserve(pairs.size());
    bits.reserve(pairs.size());
    int next_dim = 0;
    for (const auto& pair : pairs) {
        std::vector<int> dims(static_cast<size_t>(pair.dims));
        std::iota(dims.begin(), dims.end(), next_dim);
        next_dim += pair.dims;
        groups.push_back(std::move(dims));
        bits.push_back(pair.bits);
    }
    if (next_dim != ctx.d) {
        fail("pair dimensions do not cover the input dimension");
    }
    auto structure = sbi::make_structure(
            groups,
            bits,
            ctx,
            "ArchitectureControlBuilder");
    structure.meta["variant"] = variant;
    structure.meta["canonical_pair_order"] = "descending_dims_then_bits";
    structure.meta["design_proxy_j"] = proxy_j;
    return structure;
}

double score_fixed(
        sbi::ProxyContext& proxy,
        const sbi::Groups& groups,
        const sbi::Bits& bits) {
    if (groups.size() != bits.size()) {
        fail("cannot score mismatched groups and bits");
    }
    double score = 0.0;
    for (size_t i = 0; i < groups.size(); ++i) {
        score += proxy.D(groups[i], bits[i]);
    }
    return score;
}

void save_variant(
        const fs::path& output_dir,
        const std::string& name,
        const std::vector<PairSpec>& pairs,
        const epq::BuildContext& ctx,
        double proxy_j) {
    auto structure = structure_from_pairs(pairs, ctx, name, proxy_j);
    const fs::path output = output_dir / (name + ".json");
    structure.save_json(output.string());
    std::cout << "variant=" << name
              << " M=" << structure.group_count()
              << " Jstar=" << proxy_j
              << " output=" << output << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const FvecsData data = load_fvecs(args.input);
        const epq::Structure learned =
                epq::Structure::load_json(args.learned_structure.string());
        if (learned.d != data.d || learned.total_bits != args.total_bits) {
            fail("learned structure does not match input dimension/bit budget");
        }
        const epq::BuildContext ctx{
                .d = data.d,
                .total_bits = args.total_bits,
                .min_bits = args.min_bits,
                .max_bits = args.max_bits,
        };
        learned.validate(ctx.min_bits, ctx.max_bits);

        const Eigen::Map<const sbi::RowMatrixXf> x(
                data.values.data(),
                static_cast<Eigen::Index>(data.rows),
                data.d);
        const auto split = sbi::split_train_eval_rows(
                x,
                args.proxy_max_train_rows,
                args.proxy_max_eval_rows,
                args.proxy_eval_frac,
                args.seed);
        sbi::ProxyContext proxy{
                .build_ctx = ctx,
                .xt_train = split.train,
                .xt_eval = split.eval,
                .km_niter = args.proxy_kmeans_niter,
                .km_nredo = args.proxy_kmeans_nredo,
                .min_points_per_centroid = args.proxy_min_points_per_centroid,
                .seed = args.seed,
        };
        proxy.cache_slices = true;
        proxy.d_cache.max_size = 400000;
        proxy.xtr_cache.max_weight = 32ULL << 30;
        proxy.xev_cache.max_weight = 32ULL << 30;

        fs::create_directories(args.output_dir);
        sbi::Groups learned_groups;
        sbi::Bits learned_bits;
        for (const auto& group : learned.groups) {
            learned_groups.push_back(group.dims);
            learned_bits.push_back(group.nbits);
        }
        const double learned_j = score_fixed(proxy, learned_groups, learned_bits);
        save_variant(
                args.output_dir,
                "learned_architecture",
                canonical_pairs(learned_groups, learned_bits),
                ctx,
                learned_j);

        const int learned_m = static_cast<int>(learned_groups.size());
        auto learned_balanced_bits = sbi::distribute_bits_evenly(
                args.total_bits,
                learned_m);
        auto learned_size_groups = learned_groups;
        std::sort(
                learned_size_groups.begin(),
                learned_size_groups.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.size() > rhs.size();
                });
        std::sort(learned_balanced_bits.begin(), learned_balanced_bits.end(), std::greater<>());
        const double learned_balanced_j =
                score_fixed(proxy, learned_size_groups, learned_balanced_bits);
        save_variant(
                args.output_dir,
                "learned_dims_balanced_bits",
                canonical_pairs(learned_size_groups, learned_balanced_bits),
                ctx,
                learned_balanced_j);

        const auto matched_groups = sbi::balanced_groups(data.d, learned_m);
        const auto matched_alloc = proxy.solve_bits(matched_groups);
        save_variant(
                args.output_dir,
                "balanced_dims_dp_bits_matched_m",
                canonical_pairs(matched_groups, matched_alloc.bits),
                ctx,
                matched_alloc.J);
        const auto matched_balanced_bits = sbi::distribute_bits_evenly(
                args.total_bits,
                learned_m);
        const double matched_balanced_j =
                score_fixed(proxy, matched_groups, matched_balanced_bits);
        save_variant(
                args.output_dir,
                "balanced_dims_balanced_bits_matched_m",
                canonical_pairs(matched_groups, matched_balanced_bits),
                ctx,
                matched_balanced_j);

        double grid_best_j = std::numeric_limits<double>::infinity();
        std::vector<PairSpec> grid_best_pairs;
        int grid_best_m = -1;
        for (int groups_count : args.grid_groups) {
            const auto groups = sbi::balanced_groups(data.d, groups_count);
            const auto alloc = proxy.solve_bits(groups);
            const std::string name =
                    "grid_m" + std::to_string(groups_count) + "_dp_bits";
            const auto pairs = canonical_pairs(groups, alloc.bits);
            save_variant(args.output_dir, name, pairs, ctx, alloc.J);
            const auto equal_bits = sbi::distribute_bits_evenly(
                    args.total_bits, groups_count);
            const std::string equal_name =
                    "grid_m" + std::to_string(groups_count) + "_equal_bits";
            save_variant(
                    args.output_dir,
                    equal_name,
                    canonical_pairs(groups, equal_bits),
                    ctx,
                    score_fixed(proxy, groups, equal_bits));
            if (alloc.J < grid_best_j) {
                grid_best_j = alloc.J;
                grid_best_pairs = pairs;
                grid_best_m = groups_count;
            }
        }
        save_variant(
                args.output_dir,
                "grid_best_dp_bits",
                grid_best_pairs,
                ctx,
                grid_best_j);
        std::cout << "grid_best_m=" << grid_best_m
                  << " grid_best_Jstar=" << grid_best_j << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "architecture_control_builder: " << e.what() << '\n';
        return 1;
    }
}
