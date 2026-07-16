#include "epq/index_epq.h"
#include "epq/index_ivf_codec.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <faiss/IndexFlat.h>

#include "epq/structure_builder.h"

namespace {

std::vector<float> make_random_matrix(
        std::mt19937& rng,
        faiss::idx_t rows,
        int d) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> out(static_cast<size_t>(rows) * static_cast<size_t>(d));
    for (float& value : out) {
        value = dist(rng);
    }
    return out;
}

double exact_query_match_at_k(
        const std::vector<faiss::idx_t>& lhs,
        const std::vector<faiss::idx_t>& rhs,
        int nq,
        int topk,
        int k) {
    int matched = 0;
    for (int qi = 0; qi < nq; ++qi) {
        bool same = true;
        for (int j = 0; j < k; ++j) {
            const size_t offset =
                    static_cast<size_t>(qi) * static_cast<size_t>(topk) +
                    static_cast<size_t>(j);
            if (lhs[offset] != rhs[offset]) {
                same = false;
                break;
            }
        }
        matched += same ? 1 : 0;
    }
    return static_cast<double>(matched) / static_cast<double>(nq);
}

}  // namespace

int main() {
    try {
        constexpr int d = 15;
        constexpr faiss::idx_t ntrain = 16384;
        constexpr faiss::idx_t nb = 20000;
        constexpr faiss::idx_t nq = 128;
        constexpr faiss::idx_t nlist = 256;
        constexpr faiss::idx_t nprobe = 8;
        constexpr faiss::idx_t topk = 100;

        epq::Structure fixed_structure;
        fixed_structure.d = d;
        fixed_structure.total_bits = 128;
        fixed_structure.groups = {
                {{0}, 6},
                {{1}, 6},
                {{2}, 7},
                {{3}, 8},
                {{4}, 8},
                {{5}, 8},
                {{6}, 8},
                {{7}, 9},
                {{8}, 9},
                {{9}, 9},
                {{10}, 9},
                {{11}, 9},
                {{12}, 10},
                {{13}, 10},
                {{14}, 12},
        };

        auto builder = std::make_shared<epq::FixedStructureBuilder>(fixed_structure);
        auto codec = std::make_unique<epq::IndexEPQ>(d, fixed_structure.total_bits, builder);
        codec->use_uneven_transform = false;
        codec->kmeans_niter = 12;
        codec->kmeans_nredo = 1;

        auto* quantizer = new faiss::IndexFlatL2(d);
        epq::IndexIVFCodec<epq::IndexEPQ> index(
                std::move(codec), quantizer, nlist, "IVF+EPQ");
        index.cp.niter = 20;
        index.cp.nredo = 1;
        index.nprobe = nprobe;

        std::mt19937 rng(123);
        const std::vector<float> xt = make_random_matrix(rng, ntrain, d);
        const std::vector<float> xb = make_random_matrix(rng, nb, d);
        const std::vector<float> xq = make_random_matrix(rng, nq, d);

        index.train(ntrain, xt.data());
        index.add(nb, xb.data());

        std::vector<float> scalar_dist(static_cast<size_t>(nq) * static_cast<size_t>(topk));
        std::vector<float> exact_dist(static_cast<size_t>(nq) * static_cast<size_t>(topk));
        std::vector<float> fallback_dist(static_cast<size_t>(nq) * static_cast<size_t>(topk));
        std::vector<faiss::idx_t> scalar_lab(static_cast<size_t>(nq) * static_cast<size_t>(topk));
        std::vector<faiss::idx_t> exact_lab(static_cast<size_t>(nq) * static_cast<size_t>(topk));
        std::vector<faiss::idx_t> fallback_lab(static_cast<size_t>(nq) * static_cast<size_t>(topk));

        epq::set_epq_ivf_search_mode(epq::EpqIvfSearchMode::kScalarLut);
        index.search(nq, xq.data(), topk, scalar_dist.data(), scalar_lab.data());
        epq::set_epq_ivf_search_mode(epq::EpqIvfSearchMode::kExactDecode);
        index.search(nq, xq.data(), topk, exact_dist.data(), exact_lab.data());
        epq::set_epq_ivf_search_mode(epq::EpqIvfSearchMode::kFallbackScanner);
        index.search(nq, xq.data(), topk, fallback_dist.data(), fallback_lab.data());
        epq::set_epq_ivf_search_mode(epq::EpqIvfSearchMode::kScalarLut);

        const double scalar_exact_top10 =
                exact_query_match_at_k(scalar_lab, exact_lab, nq, topk, 10);
        const double scalar_exact_top100 =
                exact_query_match_at_k(scalar_lab, exact_lab, nq, topk, 100);
        const double fallback_exact_top10 =
                exact_query_match_at_k(fallback_lab, exact_lab, nq, topk, 10);
        const double fallback_exact_top100 =
                exact_query_match_at_k(fallback_lab, exact_lab, nq, topk, 100);

        if (scalar_exact_top100 < 0.999 || fallback_exact_top100 < 0.999) {
            std::cerr << "epq_ivf_lut_smoke_test mismatch"
                      << " scalar_exact_top10=" << scalar_exact_top10
                      << " scalar_exact_top100=" << scalar_exact_top100
                      << " fallback_exact_top10=" << fallback_exact_top10
                      << " fallback_exact_top100=" << fallback_exact_top100
                      << '\n';
            return 1;
        }

        std::cout << "epq_ivf_lut_smoke_test ok"
                  << " scalar_exact_top100=" << scalar_exact_top100
                  << " fallback_exact_top100=" << fallback_exact_top100
                  << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "epq_ivf_lut_smoke_test failed: " << e.what() << '\n';
        return 1;
    }
}
