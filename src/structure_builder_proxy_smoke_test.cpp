#include "structure_builder_internal.h"

#include "epq/structure_builder.h"
#include "epq/training_config.h"

#include <cmath>
#include <iostream>
#include <memory>

namespace sbi = epq::structure_builder_internal;

namespace {

sbi::RowMatrixXf make_matrix(int rows, float offset) {
    sbi::RowMatrixXf x(rows, 6);
    for (int i = 0; i < rows; ++i) {
        const float t = static_cast<float>(i) + offset;
        const float c = 4.0f * static_cast<float>(i % 4) + 0.15f * static_cast<float>(i / 4);
        x(i, 0) = c;
        x(i, 1) = 0.6f * c + 0.2f * static_cast<float>(i % 2);
        x(i, 2) = 0.05f * std::sin(t);
        x(i, 3) = 0.04f * std::cos(0.5f * t);
        x(i, 4) = 0.03f * static_cast<float>((i % 3) - 1);
        x(i, 5) = 0.02f * static_cast<float>((i % 5) - 2);
    }
    return x;
}

}  // namespace

int main() {
    const nlohmann::json config = {
            {"builder",
             {
                     {"type", "refined"},
                     {"refined",
                      {
                              {"proxy_pca_top_dims", 2},
                              {"proxy_max_pca_cache", 8},
                      }},
             }},
    };
    const auto builder_base = epq::make_structure_builder_from_config(config);
    const auto builder =
            std::dynamic_pointer_cast<epq::RefinedStructureBuilder>(builder_base);
    if (!builder) {
        std::cerr << "failed to parse refined builder config\n";
        return 1;
    }
    if (builder->proxy_pca_top_dims != 2 || builder->proxy_max_pca_cache != 8) {
        std::cerr << "proxy PCA config did not round-trip through JSON parsing\n";
        return 1;
    }

    const epq::BuildContext ctx{
            .d = 6,
            .total_bits = 8,
            .min_bits = 0,
            .max_bits = 4,
    };
    const sbi::RowMatrixXf xt_train = make_matrix(16, 0.0f);
    const sbi::RowMatrixXf xt_eval = make_matrix(8, 0.35f);
    const std::vector<int> dims = {0, 1, 2, 3, 4, 5};

    sbi::ProxyContext exact{
            .build_ctx = ctx,
            .xt_train = xt_train,
            .xt_eval = xt_eval,
            .km_niter = 5,
            .km_nredo = 1,
            .min_points_per_centroid = 1,
            .seed = 123,
    };
    const double d_exact = exact.D(dims, 2);

    sbi::ProxyContext approx{
            .build_ctx = ctx,
            .xt_train = xt_train,
            .xt_eval = xt_eval,
            .km_niter = 5,
            .km_nredo = 1,
            .min_points_per_centroid = 1,
            .pca_top_dims = 2,
            .seed = 123,
    };
    approx.pca_cache.max_size = 8;

    const double d_approx_b1 = approx.D(dims, 1);
    const double d_approx_b2 = approx.D(dims, 2);

    if (!std::isfinite(d_exact) || !std::isfinite(d_approx_b1) ||
        !std::isfinite(d_approx_b2)) {
        std::cerr << "proxy D returned non-finite value\n";
        return 1;
    }
    if (approx.cache_stats.pca_misses != 1 || approx.cache_stats.pca_hits < 1) {
        std::cerr << "proxy PCA cache stats unexpected: hits="
                  << approx.cache_stats.pca_hits
                  << " misses=" << approx.cache_stats.pca_misses << '\n';
        return 1;
    }
    if (approx.work_stats.pca_approx_calls != 2 || approx.work_stats.pca_fits != 1) {
        std::cerr << "proxy PCA work stats unexpected: calls="
                  << approx.work_stats.pca_approx_calls
                  << " fits=" << approx.work_stats.pca_fits << '\n';
        return 1;
    }
    if (approx.work_stats.kmeans_dims_total >= exact.work_stats.kmeans_dims_total * 2) {
        std::cerr << "proxy PCA did not reduce effective k-means dims\n";
        return 1;
    }

    std::cout << "structure_builder_proxy_smoke_test ok"
              << " exact=" << d_exact
              << " approx_b2=" << d_approx_b2
              << " pca_hits=" << approx.cache_stats.pca_hits << '\n';
    return 0;
}
