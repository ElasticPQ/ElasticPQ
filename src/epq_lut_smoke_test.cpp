#include "epq/index_epq.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

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

float squared_l2(const float* lhs, const float* rhs, int d) {
    float acc = 0.0f;
    for (int i = 0; i < d; ++i) {
        const float diff = lhs[static_cast<size_t>(i)] - rhs[static_cast<size_t>(i)];
        acc += diff * diff;
    }
    return acc;
}

}  // namespace

int main() {
    try {
        constexpr int d = 15;
        constexpr faiss::idx_t ntrain = 8192;
        constexpr faiss::idx_t nb = 512;
        constexpr faiss::idx_t nq = 32;

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
        epq::IndexEPQ index(d, fixed_structure.total_bits, builder);
        index.use_uneven_transform = false;
        index.kmeans_niter = 12;
        index.kmeans_nredo = 1;

        std::mt19937 rng(123);
        const std::vector<float> xt = make_random_matrix(rng, ntrain, d);
        const std::vector<float> xb = make_random_matrix(rng, nb, d);
        const std::vector<float> xq = make_random_matrix(rng, nq, d);

        index.train(ntrain, xt.data());

        const size_t code_size = index.sa_code_size();
        std::vector<uint8_t> codes(static_cast<size_t>(nb) * code_size, 0);
        index.sa_encode(nb, xb.data(), codes.data());

        std::vector<float> lut(index.adc_lut_size(), 0.0f);
        std::vector<float> decoded(static_cast<size_t>(d), 0.0f);
        std::vector<float> transformed_query(static_cast<size_t>(d), 0.0f);

        float worst_abs_err = 0.0f;
        size_t worst_q = 0;
        size_t worst_i = 0;
        float worst_adc = 0.0f;
        float worst_exact = 0.0f;

        for (faiss::idx_t qi = 0; qi < nq; ++qi) {
            const float* query =
                    xq.data() + static_cast<size_t>(qi) * static_cast<size_t>(d);
            index.transform_vector(query, transformed_query.data());
            index.compute_adc_lut_from_transformed(transformed_query.data(), lut.data());
            for (faiss::idx_t i = 0; i < nb; ++i) {
                const uint8_t* code =
                        codes.data() + static_cast<size_t>(i) * code_size;
                const float adc = index.adc_distance_from_packed_code(code, lut.data());
                index.sa_decode(1, code, decoded.data());
                const float exact = squared_l2(decoded.data(), query, d);
                const float abs_err = std::fabs(adc - exact);
                if (abs_err > worst_abs_err) {
                    worst_abs_err = abs_err;
                    worst_q = static_cast<size_t>(qi);
                    worst_i = static_cast<size_t>(i);
                    worst_adc = adc;
                    worst_exact = exact;
                }
            }
        }

        if (worst_abs_err > 1e-3f) {
            std::cerr << "epq_lut_smoke_test mismatch"
                      << " q=" << worst_q
                      << " i=" << worst_i
                      << " adc=" << worst_adc
                      << " exact=" << worst_exact
                      << " abs_err=" << worst_abs_err << '\n';
            return 1;
        }

        std::cout << "epq_lut_smoke_test ok"
                  << " worst_abs_err=" << worst_abs_err << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "epq_lut_smoke_test failed: " << e.what() << '\n';
        return 1;
    }
}
