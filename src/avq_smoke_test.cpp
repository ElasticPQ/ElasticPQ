#include "epq/index_avq.h"

#include <faiss/Index.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        constexpr int d = 8;
        constexpr faiss::idx_t nb = 16;
        constexpr faiss::idx_t nq = 2;
        constexpr faiss::idx_t topk = 3;

        std::vector<float> xb(static_cast<size_t>(nb) * d);
        std::vector<float> xq(static_cast<size_t>(nq) * d);
        for (faiss::idx_t i = 0; i < nb; ++i) {
            for (int j = 0; j < d; ++j) {
                xb[static_cast<size_t>(i) * d + static_cast<size_t>(j)] =
                        0.01f * static_cast<float>(i * d + j);
            }
        }
        for (faiss::idx_t i = 0; i < nq; ++i) {
            for (int j = 0; j < d; ++j) {
                xq[static_cast<size_t>(i) * d + static_cast<size_t>(j)] =
                        xb[static_cast<size_t>(i) * d + static_cast<size_t>(j)];
            }
        }

        epq::IndexAVQ index(d, 32);
        index.default_num_neighbors = static_cast<int>(topk);
        index.training_threads = 1;
        index.search_threads = 1;
        index.train(nb, xb.data());
        index.add(nb, xb.data());

        std::vector<float> distances(static_cast<size_t>(nq) * topk);
        std::vector<faiss::idx_t> labels(static_cast<size_t>(nq) * topk, -1);
        index.search(
                nq,
                xq.data(),
                topk,
                distances.data(),
                labels.data());

        for (faiss::idx_t i = 0; i < nq; ++i) {
            if (labels[static_cast<size_t>(i) * topk] < 0) {
                throw std::runtime_error("AVQ smoke test returned an invalid top-1 id");
            }
            if (!std::isfinite(distances[static_cast<size_t>(i) * topk])) {
                throw std::runtime_error("AVQ smoke test returned a non-finite distance");
            }
        }

        std::cout << "avq_smoke_test passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "avq_smoke_test failed: " << e.what() << '\n';
        return 1;
    }
}
