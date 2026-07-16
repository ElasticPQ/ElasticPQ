#include <faiss/IndexFlat.h>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
    using json = nlohmann::json;

    const Eigen::Matrix3f x = (Eigen::Matrix3f() << 1.0f, 2.0f, 3.0f,
                                                   4.0f, 5.0f, 6.0f,
                                                   7.0f, 8.0f, 10.0f)
                                      .finished();
    const Eigen::JacobiSVD<Eigen::Matrix3f> svd(x, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::Vector3f singular_values = svd.singularValues();

    constexpr faiss::idx_t d = 3;
    constexpr faiss::idx_t nb = 3;
    constexpr faiss::idx_t nq = 1;
    constexpr faiss::idx_t k = 2;

    const std::vector<float> xb = {
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.5f, 0.5f, 0.0f,
    };
    const std::vector<float> xq = {
            0.0f, 1.0f, 0.0f,
    };

    faiss::IndexFlatL2 index(d);
    index.add(nb, xb.data());

    std::vector<faiss::idx_t> labels(k);
    std::vector<float> distances(k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    if (labels[0] != 1) {
        std::cerr << "epq stack smoke test failed: unexpected faiss top-1 id=" << labels[0] << '\n';
        return EXIT_FAILURE;
    }

    json report;
    report["eigen"] = {
            {"rows", x.rows()},
            {"cols", x.cols()},
            {"singular_values", {singular_values(0), singular_values(1), singular_values(2)}},
    };
    report["faiss"] = {
            {"topk_ids", {labels[0], labels[1]}},
            {"topk_distances", {distances[0], distances[1]}},
    };
    report["status"] = "ok";

    std::cout << report.dump(2) << '\n';
    return EXIT_SUCCESS;
}
