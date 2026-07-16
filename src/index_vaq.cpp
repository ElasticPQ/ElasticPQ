#include "epq/index_vaq.h"
#include "epq/variable_bit_packing.h"

#include "VAQ.hpp"

#include <faiss/impl/FaissException.h>

#include <Eigen/Core>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <utility>

namespace epq {
namespace {

using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

int choose_subspace_count(int d, int total_bits) {
    const int target = std::max(1, total_bits / 4);
    const int minimum = std::max(1, (total_bits + 7) / 8);
    for (int candidate = std::min(d, target); candidate >= minimum; --candidate) {
        if (d % candidate == 0) {
            return candidate;
        }
    }
    return 0;
}

float l2(const float* lhs, const float* rhs, int d) {
    float distance = 0.0f;
    for (int i = 0; i < d; ++i) {
        const float delta = lhs[i] - rhs[i];
        distance += delta * delta;
    }
    return distance;
}

class OmpThreadCountGuard {
   public:
    OmpThreadCountGuard() : count_(omp_get_max_threads()) {}
    ~OmpThreadCountGuard() {
        omp_set_num_threads(count_);
    }

   private:
    int count_;
};

}  // namespace

struct IndexVAQ::Impl {
    VAQ upstream;
    Matrix rotation;
    Matrix inverse_rotation;
    std::vector<int> bit_offsets;
    std::vector<size_t> lut_offsets;

    Matrix project(faiss::idx_t n, int d, const float* x) const {
        Eigen::Map<const Matrix> mapped(x, n, d);
        Matrix projected(n, d);
        projected.noalias() = mapped * rotation;
        return projected;
    }
};

IndexVAQ::IndexVAQ(
        int d,
        int total_bits,
        int subspaces_in,
        int min_bits_per_subspace_in,
        int max_bits_per_subspace_in,
        float variance_fraction_in)
        : faiss::Index(d, faiss::METRIC_L2),
          total_bits(total_bits),
          subspaces(
                  subspaces_in > 0 ? subspaces_in
                                   : choose_subspace_count(d, total_bits)),
          min_bits_per_subspace(min_bits_per_subspace_in),
          max_bits_per_subspace(max_bits_per_subspace_in),
          variance_fraction(variance_fraction_in),
          impl_(std::make_unique<Impl>()) {
    validate_config();
}

IndexVAQ::~IndexVAQ() = default;

void IndexVAQ::validate_config() const {
    FAISS_THROW_IF_NOT_MSG(d > 0, "VAQ requires a positive dimension");
    FAISS_THROW_IF_NOT_MSG(total_bits > 0, "VAQ requires a positive bit budget");
    FAISS_THROW_IF_NOT_MSG(total_bits % 8 == 0, "VAQ requires bits divisible by 8");
    FAISS_THROW_IF_NOT_MSG(subspaces > 0, "VAQ could not choose a valid subspace count");
    FAISS_THROW_IF_NOT_MSG(d % subspaces == 0, "VAQ requires d divisible by subspaces");
    FAISS_THROW_IF_NOT_MSG(
            total_bits >= subspaces * min_bits_per_subspace,
            "VAQ bit budget is below the configured per-subspace minimum");
    FAISS_THROW_IF_NOT_MSG(
            total_bits <= subspaces * max_bits_per_subspace,
            "VAQ bit budget exceeds the configured per-subspace maximum");
    FAISS_THROW_IF_NOT_MSG(
            max_bits_per_subspace <= 15,
            "VAQ wrapper supports at most 15 bits per subspace");
}

void IndexVAQ::train(faiss::idx_t n, const float* x) {
    validate_config();
    FAISS_THROW_IF_NOT_MSG(n > 0 && x != nullptr, "VAQ train requires data");
    const auto t0 = std::chrono::steady_clock::now();
    stats_ = {};
    codes_.clear();
    ntotal = 0;

    Eigen::Map<const Matrix> mapped(x, n, d);
    ::RowMatrixXf training(mapped);
    VAQ& vaq = impl_->upstream;
    vaq.mBitBudget = total_bits;
    vaq.mSubspaceNum = subspaces;
    // Upstream compares cumulative float sums against this value to decide
    // which subspaces receive the minimum-bit constraint. Positive infinity
    // represents full variance without losing the last subspace to rounding.
    vaq.mPercentVarExplained = variance_fraction >= 1.0f
            ? std::numeric_limits<float>::infinity()
            : variance_fraction;
    vaq.mMinBitsPerSubs = min_bits_per_subspace;
    vaq.mMaxBitsPerSubs = max_bits_per_subspace;
    vaq.mMethods = VAQ::NNMethod::Heap;
    vaq.mHierarchicalKmeans = false;
    vaq.mBinaryKmeans = false;
    vaq.mBitsAlloc.clear();
    vaq.mCentroidsNum.clear();
    vaq.mCentroidsPerSubs.clear();
    vaq.mCentroidsPerSubsCMajor.clear();

    const OmpThreadCountGuard thread_count_guard;
    vaq.train(training, false);

    FAISS_THROW_IF_NOT_MSG(
            static_cast<int>(vaq.mBitsAlloc.size()) == subspaces,
            "VAQ returned an unexpected bit allocation size");
    const int allocated =
            std::accumulate(vaq.mBitsAlloc.begin(), vaq.mBitsAlloc.end(), 0);
    FAISS_THROW_IF_NOT_MSG(allocated == total_bits, "VAQ did not honor the bit budget");
    for (const int width : vaq.mBitsAlloc) {
        FAISS_THROW_IF_NOT_MSG(
                width >= min_bits_per_subspace &&
                        width <= max_bits_per_subspace,
                "VAQ returned an unsupported subspace bit width");
    }

    impl_->rotation = vaq.mEigenVectors.real();
    impl_->inverse_rotation = impl_->rotation.transpose();
    impl_->bit_offsets.resize(static_cast<size_t>(subspaces + 1));
    impl_->lut_offsets.resize(static_cast<size_t>(subspaces + 1));
    impl_->bit_offsets[0] = 0;
    impl_->lut_offsets[0] = 0;
    for (int group = 0; group < subspaces; ++group) {
        impl_->bit_offsets[static_cast<size_t>(group + 1)] =
                impl_->bit_offsets[static_cast<size_t>(group)] +
                vaq.mBitsAlloc[static_cast<size_t>(group)];
        impl_->lut_offsets[static_cast<size_t>(group + 1)] =
                impl_->lut_offsets[static_cast<size_t>(group)] +
                static_cast<size_t>(vaq.mCentroidsNum[static_cast<size_t>(group)]);
    }

    stats_.codebook_time =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                    .count();
    stats_.total_time = stats_.codebook_time;
    is_trained = true;
}

void IndexVAQ::add(faiss::idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "VAQ index is not trained");
    FAISS_THROW_IF_NOT_MSG(n >= 0 && (n == 0 || x != nullptr), "invalid VAQ add input");
    const size_t old_size = codes_.size();
    codes_.resize(old_size + static_cast<size_t>(n) * sa_code_size());
    sa_encode(n, x, codes_.data() + old_size);
    ntotal += n;
}

void IndexVAQ::search(
        faiss::idx_t n,
        const float* x,
        faiss::idx_t k,
        float* distances,
        faiss::idx_t* labels,
        const faiss::SearchParameters*) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "VAQ index is not trained");
    FAISS_THROW_IF_NOT_MSG(n >= 0 && k > 0, "invalid VAQ search shape");
    FAISS_THROW_IF_NOT_MSG(x != nullptr && distances != nullptr && labels != nullptr,
                           "invalid VAQ search buffers");
    const size_t code_size = sa_code_size();
#pragma omp parallel
    {
        std::vector<float> transformed(static_cast<size_t>(d));
        std::vector<float> lut(adc_lut_size());
#pragma omp for schedule(static)
        for (faiss::idx_t query = 0; query < n; ++query) {
            transform_vector(x + static_cast<size_t>(query) * d, transformed.data());
            compute_adc_lut_from_transformed(transformed.data(), lut.data());
            using Candidate = std::pair<float, faiss::idx_t>;
            std::priority_queue<Candidate> heap;
            for (faiss::idx_t id = 0; id < ntotal; ++id) {
                const float distance = adc_distance_from_packed_code(
                        codes_.data() + static_cast<size_t>(id) * code_size,
                        lut.data());
                if (static_cast<faiss::idx_t>(heap.size()) < k) {
                    heap.emplace(distance, id);
                } else if (distance < heap.top().first) {
                    heap.pop();
                    heap.emplace(distance, id);
                }
            }
            const size_t base = static_cast<size_t>(query) * static_cast<size_t>(k);
            const faiss::idx_t found = static_cast<faiss::idx_t>(heap.size());
            for (faiss::idx_t rank = found; rank-- > 0;) {
                distances[base + static_cast<size_t>(rank)] = heap.top().first;
                labels[base + static_cast<size_t>(rank)] = heap.top().second;
                heap.pop();
            }
            for (faiss::idx_t rank = found; rank < k; ++rank) {
                distances[base + static_cast<size_t>(rank)] =
                        std::numeric_limits<float>::infinity();
                labels[base + static_cast<size_t>(rank)] = -1;
            }
        }
    }
}

void IndexVAQ::reset() {
    codes_.clear();
    ntotal = 0;
}

void IndexVAQ::reconstruct(faiss::idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(key >= 0 && key < ntotal, "VAQ reconstruct key out of range");
    sa_decode(
            1,
            codes_.data() + static_cast<size_t>(key) * sa_code_size(),
            recons);
}

size_t IndexVAQ::sa_code_size() const {
    return static_cast<size_t>((total_bits + 7) / 8);
}

void IndexVAQ::sa_encode(faiss::idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "VAQ codec is not trained");
    FAISS_THROW_IF_NOT_MSG(n >= 0 && (n == 0 || (x != nullptr && bytes != nullptr)),
                           "invalid VAQ encode input");
    constexpr faiss::idx_t batch_size = 65536;
    const int subspace_dim = d / subspaces;
    const size_t code_size = sa_code_size();
    for (faiss::idx_t begin = 0; begin < n; begin += batch_size) {
        const faiss::idx_t count = std::min(batch_size, n - begin);
        Matrix projected = impl_->project(
                count, d, x + static_cast<size_t>(begin) * static_cast<size_t>(d));
#pragma omp parallel for schedule(static)
        for (faiss::idx_t row = 0; row < count; ++row) {
            uint8_t* output = bytes +
                    static_cast<size_t>(begin + row) * code_size;
            std::memset(output, 0, code_size);
            for (int group = 0; group < subspaces; ++group) {
                const auto& centroids =
                        impl_->upstream.mCentroidsPerSubs[static_cast<size_t>(group)];
                const float* input = projected.row(row).data() + group * subspace_dim;
                uint16_t best = 0;
                float best_distance = std::numeric_limits<float>::infinity();
                for (int centroid = 0; centroid < centroids.rows(); ++centroid) {
                    const float distance =
                            l2(input, centroids.row(centroid).data(), subspace_dim);
                    if (distance < best_distance) {
                        best_distance = distance;
                        best = static_cast<uint16_t>(centroid);
                    }
                }
                detail::pack_variable_bits(
                        output,
                        impl_->bit_offsets[static_cast<size_t>(group)],
                        impl_->upstream.mBitsAlloc[static_cast<size_t>(group)],
                        best);
            }
        }
    }
}

void IndexVAQ::sa_decode(faiss::idx_t n, const uint8_t* bytes, float* x) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "VAQ codec is not trained");
    FAISS_THROW_IF_NOT_MSG(n >= 0 && (n == 0 || (bytes != nullptr && x != nullptr)),
                           "invalid VAQ decode input");
    const int subspace_dim = d / subspaces;
    const size_t code_size = sa_code_size();
    Matrix projected(n, d);
    for (faiss::idx_t row = 0; row < n; ++row) {
        const uint8_t* code = bytes + static_cast<size_t>(row) * code_size;
        for (int group = 0; group < subspaces; ++group) {
            const uint16_t centroid = detail::unpack_variable_bits(
                    code,
                    impl_->bit_offsets[static_cast<size_t>(group)],
                    impl_->upstream.mBitsAlloc[static_cast<size_t>(group)]);
            projected.block(row, group * subspace_dim, 1, subspace_dim) =
                    impl_->upstream.mCentroidsPerSubs[static_cast<size_t>(group)]
                            .row(centroid);
        }
    }
    Eigen::Map<Matrix> output(x, n, d);
    output.noalias() = projected * impl_->inverse_rotation;
}

int IndexVAQ::component_count() const noexcept {
    return subspaces;
}

const std::vector<int>& IndexVAQ::bit_allocation() const {
    return impl_->upstream.mBitsAlloc;
}

const VAQTrainingStats& IndexVAQ::training_stats() const noexcept {
    return stats_;
}

size_t IndexVAQ::serialized_payload_bytes() const {
    size_t bytes = codes_.size();
    bytes += static_cast<size_t>(impl_->rotation.size()) * sizeof(float);
    bytes += impl_->upstream.mBitsAlloc.size() * sizeof(int);
    for (const auto& centroids : impl_->upstream.mCentroidsPerSubs) {
        bytes += static_cast<size_t>(centroids.size()) * sizeof(float);
    }
    return bytes;
}

size_t IndexVAQ::adc_lut_size() const noexcept {
    return impl_->lut_offsets.empty() ? 0 : impl_->lut_offsets.back();
}

void IndexVAQ::transform_vector(const float* x, float* out) const {
    Eigen::Map<const Eigen::RowVectorXf> input(x, d);
    Eigen::Map<Eigen::RowVectorXf> output(out, d);
    output.noalias() = input * impl_->rotation;
}

void IndexVAQ::compute_adc_lut_from_transformed(
        const float* query_transformed,
        float* lut) const {
    const int subspace_dim = d / subspaces;
    for (int group = 0; group < subspaces; ++group) {
        const auto& centroids =
                impl_->upstream.mCentroidsPerSubs[static_cast<size_t>(group)];
        float* output = lut + impl_->lut_offsets[static_cast<size_t>(group)];
        for (int centroid = 0; centroid < centroids.rows(); ++centroid) {
            output[centroid] = l2(
                    query_transformed + group * subspace_dim,
                    centroids.row(centroid).data(),
                    subspace_dim);
        }
    }
}

float IndexVAQ::adc_distance_from_packed_code(
        const uint8_t* code,
        const float* lut) const {
    float distance = 0.0f;
    for (int group = 0; group < subspaces; ++group) {
        const uint16_t centroid = detail::unpack_variable_bits(
                code,
                impl_->bit_offsets[static_cast<size_t>(group)],
                impl_->upstream.mBitsAlloc[static_cast<size_t>(group)]);
        distance += lut[impl_->lut_offsets[static_cast<size_t>(group)] + centroid];
    }
    return distance;
}

nlohmann::json IndexVAQ::metadata() const {
    return {
            {"family", "vaq"},
            {"impl", "TheDatumOrg/VAQ"},
            {"upstream_commit", "0fbc56fec5475f8779f20ad4a7e932a064bdb354"},
            {"native_index", "VAQ(PCA+variance-aware-bit-allocation+per-subspace-kmeans)"},
            {"d", d},
            {"total_bits", total_bits},
            {"subspaces", subspaces},
            {"subspace_dim", subspaces > 0 ? d / subspaces : 0},
            {"min_bits_per_subspace", min_bits_per_subspace},
            {"max_bits_per_subspace", max_bits_per_subspace},
            {"variance_fraction", variance_fraction},
            {"bit_allocation", impl_->upstream.mBitsAlloc},
            {"code_size_bytes", sa_code_size()},
            {"search", "ADC"},
            {"integration_compatibility",
             nlohmann::json::array({
                     "initialize partial kmeans samples",
                     "use self-adjoint covariance eigensolver",
                     "swap row-major eigenvector columns elementwise",
                     "use upstream real BLAS projection path",
                     "isolate Eigen alignment from target-local AVX2 flags",
             })},
    };
}

}  // namespace epq
