#include "epq/index_bapq.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace {

using RowMatrixXf =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

float l2sqr(const float* a, const float* b, int d) {
    float acc = 0.0f;
    for (int i = 0; i < d; ++i) {
        const float diff = a[i] - b[i];
        acc += diff * diff;
    }
    return acc;
}

}  // namespace

int main() {
    constexpr int d = 32;
    constexpr int ntrain = 512;
    constexpr int nbase = 256;
    constexpr int nquery = 12;
    constexpr int bits = 16;
    constexpr int topk = 5;

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    RowMatrixXf xt(ntrain, d);
    RowMatrixXf xb(nbase, d);
    RowMatrixXf xq(nquery, d);
    for (int i = 0; i < xt.size(); ++i) {
        xt.data()[i] = dist(rng);
    }
    for (int i = 0; i < xb.size(); ++i) {
        xb.data()[i] = dist(rng);
    }
    for (int i = 0; i < xq.size(); ++i) {
        xq.data()[i] = dist(rng);
    }

    epq::IndexBAPQ index(d, bits, 4);
    index.max_train_rows = ntrain;
    index.pca_max_train_rows = ntrain;
    index.kmeans_niter = 12;
    index.kmeans_nredo = 1;
    index.query_batch = 4;
    index.db_chunk = 128;
    index.train(xt.rows(), xt.data());
    index.add(xb.rows(), xb.data());

    std::vector<float> distances(static_cast<size_t>(nquery) * topk);
    std::vector<faiss::idx_t> labels(static_cast<size_t>(nquery) * topk);
    index.search(
            xq.rows(),
            xq.data(),
            topk,
            distances.data(),
            labels.data());

    std::vector<faiss::idx_t> all_ids(static_cast<size_t>(nbase));
    for (int i = 0; i < nbase; ++i) {
        all_ids[static_cast<size_t>(i)] = i;
    }
    RowMatrixXf xb_recons;
    index.reconstruct_rows(all_ids, xb_recons);

    for (int qi = 0; qi < nquery; ++qi) {
        std::vector<std::pair<float, int>> brute;
        brute.reserve(nbase);
        for (int bi = 0; bi < nbase; ++bi) {
            brute.emplace_back(
                    l2sqr(xq.row(qi).data(), xb_recons.row(bi).data(), d),
                    bi);
        }
        std::partial_sort(
                brute.begin(),
                brute.begin() + topk,
                brute.end(),
                [](const auto& lhs, const auto& rhs) {
                    if (lhs.first != rhs.first) {
                        return lhs.first < rhs.first;
                    }
                    return lhs.second < rhs.second;
                });
        for (int k = 0; k < topk; ++k) {
            const auto got_id =
                    labels[static_cast<size_t>(qi) * topk + k];
            const auto want_id = brute[static_cast<size_t>(k)].second;
            if (got_id != want_id) {
                std::cerr << "bapq_smoke_test mismatch q=" << qi
                          << " rank=" << k
                          << " got_id=" << got_id
                          << " want_id=" << want_id << "\n";
                return 1;
            }
        }
    }

    std::cout << "bapq_smoke_test ok\n";
    return 0;
}
