#include <faiss/IndexFlat.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

bool nearly_equal(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

}  // namespace

int main() {
    constexpr faiss::idx_t d = 4;
    constexpr faiss::idx_t nb = 5;
    constexpr faiss::idx_t nq = 1;
    constexpr faiss::idx_t k = 3;

    const std::vector<float> xb = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
            1.0f, 1.0f, 0.0f, 0.0f,
    };
    const std::vector<float> xq = {
            0.0f, 1.0f, 0.0f, 0.0f,
    };

    faiss::IndexFlatL2 index(d);
    index.add(nb, xb.data());

    std::vector<faiss::idx_t> labels(k);
    std::vector<float> distances(k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    if (labels[0] != 1 || !nearly_equal(distances[0], 0.0f)) {
        std::cerr << "faiss smoke test failed: expected top-1=(1, 0.0), got ("
                  << labels[0] << ", " << distances[0] << ")\n";
        return EXIT_FAILURE;
    }

    std::cout << "faiss smoke test passed\n";
    for (faiss::idx_t i = 0; i < k; ++i) {
        std::cout << "rank " << i << ": id=" << labels[i]
                  << " dist=" << distances[i] << '\n';
    }
    return EXIT_SUCCESS;
}
