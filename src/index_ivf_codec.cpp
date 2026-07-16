#include "epq/index_ivf_codec.h"

#include "epq/index_arepq.h"
#include "epq/index_bapq.h"
#include "epq/index_dpopq.h"
#include "epq/index_epq.h"
#include "epq/index_vaq.h"

#include <faiss/impl/FaissException.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/Heap.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

bool prefer_lut_scan(const epq::IndexBAPQ&, size_t) {
    return true;
}

bool prefer_lut_scan(const epq::IndexAREPQ&, size_t) {
    return true;
}

bool prefer_lut_scan(const epq::IndexDPOPQ&, size_t) {
    return true;
}

bool prefer_lut_scan(const epq::IndexVAQ&, size_t) {
    return true;
}

size_t epq_ivf_lut_min_list_size_override() {
    static constexpr size_t kNoOverride = std::numeric_limits<size_t>::max();
    static const size_t threshold = [] {
        const char* value = std::getenv("EPQ_IVF_LUT_MIN_LIST_SIZE");
        if (value == nullptr || *value == '\0') {
            return kNoOverride;
        }
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        if (end == value || *end != '\0') {
            return kNoOverride;
        }
        return static_cast<size_t>(parsed);
    }();
    return threshold;
}

bool prefer_lut_scan(const epq::IndexEPQ& codec, size_t list_size) {
    static constexpr size_t kNoOverride = std::numeric_limits<size_t>::max();
    const size_t threshold = epq_ivf_lut_min_list_size_override();
    if (threshold != kNoOverride) {
        return list_size >= threshold;
    }
    const size_t default_threshold = codec.ivf_default_lut_min_list_size();
    return default_threshold == 0 || list_size >= default_threshold;
}

std::atomic<int>& epq_ivf_search_mode_storage() {
    static std::atomic<int> mode{[] {
        const char* value = std::getenv("EPQ_FORCE_IVF_SCANNER");
        return (value != nullptr && std::string_view(value) == "1")
                ? static_cast<int>(epq::EpqIvfSearchMode::kFallbackScanner)
                : static_cast<int>(epq::EpqIvfSearchMode::kScalarLut);
    }()};
    return mode;
}

bool epq_ivf_mode_uses_exact_decode(epq::EpqIvfSearchMode mode) {
    return mode == epq::EpqIvfSearchMode::kExactDecode ||
            mode == epq::EpqIvfSearchMode::kFallbackScanner;
}

bool collect_epq_ivf_diagnostics() {
    static const bool enabled = [] {
        const char* value = std::getenv("EPQ_IVF_DIAGNOSTICS");
        return value != nullptr && std::string_view(value) == "1";
    }();
    return enabled;
}

nlohmann::json build_query_weighted_training_sample(
        const epq::IndexEPQ& codec,
        faiss::idx_t n,
        int d,
        const float* x,
        const faiss::idx_t* assign,
        size_t nlist,
        const std::vector<float>& query_exposure,
        std::vector<float>& sampled_x) {
    nlohmann::json diag = {
            {"enabled", true},
            {"rows_in", n},
            {"rows_out", n},
            {"base_mix", codec.ivf_query_weighted_sampling_base_mix},
            {"seed", codec.ivf_query_weighted_sampling_seed},
            {"rank_decay", codec.ivf_query_weighted_sampling_rank_decay},
            {"within_list_norm_alpha",
             codec.ivf_query_weighted_sampling_within_list_norm_alpha},
    };
    if (n <= 0 || x == nullptr || assign == nullptr ||
        query_exposure.size() != nlist) {
        diag["applied"] = false;
        diag["reason"] = "invalid_inputs";
        return diag;
    }

    std::vector<size_t> base_counts(nlist, 0);
    std::vector<std::vector<size_t>> list_rows(nlist);
    size_t valid_rows = 0;
    for (faiss::idx_t i = 0; i < n; ++i) {
        const faiss::idx_t list_no = assign[static_cast<size_t>(i)];
        if (list_no < 0 || static_cast<size_t>(list_no) >= nlist) {
            continue;
        }
        const size_t li = static_cast<size_t>(list_no);
        base_counts[li]++;
        list_rows[li].push_back(static_cast<size_t>(i));
        valid_rows++;
    }
    if (valid_rows == 0) {
        diag["applied"] = false;
        diag["reason"] = "no_valid_rows";
        return diag;
    }

    double exposure_sum = 0.0;
    size_t hot_lists = 0;
    for (size_t list_no = 0; list_no < nlist; ++list_no) {
        exposure_sum += std::max(0.0f, query_exposure[list_no]);
        hot_lists += query_exposure[list_no] > 0.0f ? 1 : 0;
    }
    diag["hot_lists"] = hot_lists;
    diag["hot_list_ratio"] =
            nlist ? double(hot_lists) / static_cast<double>(nlist) : 0.0;
    if (!(exposure_sum > 0.0)) {
        diag["applied"] = false;
        diag["reason"] = "zero_query_exposure";
        return diag;
    }

    const double base_mix = std::clamp(
            static_cast<double>(codec.ivf_query_weighted_sampling_base_mix),
            0.0,
            1.0);
    const double within_list_alpha = std::clamp(
            static_cast<double>(
                    codec.ivf_query_weighted_sampling_within_list_norm_alpha),
            0.0,
            1.0);
    std::vector<double> row_weights(static_cast<size_t>(n), 0.0);
    double weight_sum = 0.0;
    size_t active_rows = 0;
    for (faiss::idx_t i = 0; i < n; ++i) {
        const faiss::idx_t list_no = assign[static_cast<size_t>(i)];
        if (list_no < 0 || static_cast<size_t>(list_no) >= nlist) {
            continue;
        }
        const size_t li = static_cast<size_t>(list_no);
        const double base_mass =
                static_cast<double>(base_counts[li]) / static_cast<double>(valid_rows);
        if (!(base_mass > 0.0)) {
            continue;
        }
        const double exposure_mass =
                std::max(0.0f, query_exposure[li]) / exposure_sum;
        const double target_mass =
                base_mix * base_mass + (1.0 - base_mix) * exposure_mass;
        const double row_weight = target_mass / base_mass;
        row_weights[static_cast<size_t>(i)] = row_weight;
        weight_sum += row_weight;
        active_rows += row_weight > 0.0 ? 1 : 0;
    }
    size_t within_list_biased_rows = 0;
    size_t within_list_biased_lists = 0;
    if (within_list_alpha > 0.0) {
        std::vector<float> row_norm2(static_cast<size_t>(n), 0.0f);
        for (faiss::idx_t i = 0; i < n; ++i) {
            const size_t row = static_cast<size_t>(i);
            const float* xi = x + row * static_cast<size_t>(d);
            float norm2 = 0.0f;
            for (int j = 0; j < d; ++j) {
                const float v = xi[static_cast<size_t>(j)];
                norm2 += v * v;
            }
            row_norm2[row] = norm2;
        }
        for (size_t li = 0; li < nlist; ++li) {
            auto& rows = list_rows[li];
            if (rows.size() <= 1) {
                continue;
            }
            std::sort(
                    rows.begin(),
                    rows.end(),
                    [&](size_t lhs, size_t rhs) { return row_norm2[lhs] < row_norm2[rhs]; });
            const double denom = static_cast<double>(rows.size() - 1);
            for (size_t rank = 0; rank < rows.size(); ++rank) {
                const double pct = static_cast<double>(rank) / denom;
                const double bias = 1.0 + within_list_alpha * (1.0 - 2.0 * pct);
                row_weights[rows[rank]] *= bias;
            }
            within_list_biased_rows += rows.size();
            within_list_biased_lists++;
        }
        weight_sum = 0.0;
        active_rows = 0;
        for (const double weight : row_weights) {
            weight_sum += weight;
            active_rows += weight > 0.0 ? 1 : 0;
        }
    }
    diag["active_rows"] = active_rows;
    diag["within_list_biased_rows"] = within_list_biased_rows;
    diag["within_list_biased_lists"] = within_list_biased_lists;
    if (!(weight_sum > 0.0)) {
        diag["applied"] = false;
        diag["reason"] = "zero_row_weight";
        return diag;
    }

    sampled_x.resize(static_cast<size_t>(n) * static_cast<size_t>(d));
    std::mt19937 rng(static_cast<uint32_t>(codec.ivf_query_weighted_sampling_seed));
    std::discrete_distribution<size_t> dist(row_weights.begin(), row_weights.end());
    for (faiss::idx_t out = 0; out < n; ++out) {
        const size_t src = dist(rng);
        std::memcpy(
                sampled_x.data() + static_cast<size_t>(out) * static_cast<size_t>(d),
                x + src * static_cast<size_t>(d),
                sizeof(float) * static_cast<size_t>(d));
    }

    diag["applied"] = true;
    diag["valid_rows"] = valid_rows;
    diag["query_exposure_sum"] = exposure_sum;
    return diag;
}

struct EpqIvfFastPathDiagnostics {
    size_t query_count = 0;
    size_t list_count = 0;
    size_t lut_lists = 0;
    size_t decode_lists = 0;
    size_t total_codes = 0;
    size_t lut_codes = 0;
    size_t decode_codes = 0;
    double setup_time = 0.0;
    double lut_build_time = 0.0;
    double lut_scan_time = 0.0;
    double decode_scan_time = 0.0;

    nlohmann::json to_json() const {
        const double total_scan_time = lut_scan_time + decode_scan_time;
        return {
                {"query_count", query_count},
                {"list_count", list_count},
                {"lut_lists", lut_lists},
                {"decode_lists", decode_lists},
                {"lut_list_ratio", list_count ? double(lut_lists) / double(list_count)
                                              : 0.0},
                {"decode_list_ratio",
                 list_count ? double(decode_lists) / double(list_count) : 0.0},
                {"total_codes", total_codes},
                {"lut_codes", lut_codes},
                {"decode_codes", decode_codes},
                {"lut_code_ratio",
                 total_codes ? double(lut_codes) / double(total_codes) : 0.0},
                {"decode_code_ratio",
                 total_codes ? double(decode_codes) / double(total_codes) : 0.0},
                {"setup_time", setup_time},
                {"lut_build_time", lut_build_time},
                {"lut_scan_time", lut_scan_time},
                {"decode_scan_time", decode_scan_time},
                {"timed_total", setup_time + lut_build_time + total_scan_time},
                {"timed_scan_total", total_scan_time},
        };
    }
};

template <typename Codec>
class IVFCodecScanner final : public faiss::InvertedListScanner {
   public:
    IVFCodecScanner(
            const faiss::IndexIVF& parent,
            const Codec& codec,
            bool store_pairs,
            const faiss::IDSelector* sel)
            : faiss::InvertedListScanner(store_pairs, sel),
              parent_(parent),
              codec_(codec),
              query_(static_cast<size_t>(codec.d), 0.0f),
              residual_query_(static_cast<size_t>(codec.d), 0.0f),
              centroid_(static_cast<size_t>(codec.d), 0.0f),
              transformed_query_(static_cast<size_t>(codec.d), 0.0f),
              transformed_residual_(static_cast<size_t>(codec.d), 0.0f),
              lut_(codec.adc_lut_size(), 0.0f),
              decoded_(static_cast<size_t>(codec.d), 0.0f) {
        keep_max = false;
        code_size = codec.sa_code_size();
    }

    void set_query(const float* query_vector) override {
        std::copy(query_vector, query_vector + codec_.d, query_.begin());
        residual_query_ = query_;
        codec_.transform_vector(query_vector, transformed_query_.data());
        transformed_residual_ = transformed_query_;
    }

    void set_list(faiss::idx_t list_no, float coarse_dis) override {
        faiss::InvertedListScanner::set_list(list_no, coarse_dis);
        if (!parent_.by_residual) {
            residual_query_ = query_;
            transformed_residual_ = transformed_query_;
            lut_ready_ = false;
            return;
        }
        parent_.quantizer->reconstruct(list_no, centroid_.data());
        for (int i = 0; i < codec_.d; ++i) {
            residual_query_[static_cast<size_t>(i)] =
                    query_[static_cast<size_t>(i)] - centroid_[static_cast<size_t>(i)];
        }
        codec_.transform_vector(residual_query_.data(), transformed_residual_.data());
        lut_ready_ = false;
    }

    float distance_to_code(const uint8_t* code) const override {
        if constexpr (std::is_same_v<Codec, epq::IndexEPQ>) {
            const epq::EpqIvfSearchMode mode = epq::epq_ivf_search_mode();
            if (epq_ivf_mode_uses_exact_decode(mode)) {
                return codec_.exact_distance_from_packed_code_transformed(
                        code,
                        transformed_residual_.data());
            }
            prepare_lut();
            return codec_.adc_distance_from_packed_code(code, lut_.data());
        }
        prepare_lut();
        return codec_.adc_distance_from_packed_code(code, lut_.data());
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const faiss::idx_t* ids,
            faiss::ResultHandler& handler) const override {
        if constexpr (std::is_same_v<Codec, epq::IndexEPQ>) {
            const epq::EpqIvfSearchMode mode = epq::epq_ivf_search_mode();
            if (!epq_ivf_mode_uses_exact_decode(mode) &&
                prefer_lut_scan(codec_, list_size)) {
                prepare_lut();
                size_t nup = 0;
                float threshold = handler.threshold;
                for (size_t j = 0; j < list_size; ++j) {
                    const int64_t id =
                            store_pairs ? faiss::lo_build(list_no, j) : ids[j];
                    if (sel != nullptr && !sel->is_member(id)) {
                        codes += code_size;
                        continue;
                    }
                    handler.stats.scan_cnt++;
                    const float dist =
                            codec_.adc_distance_from_packed_code(codes, lut_.data());
                    if (dist < threshold && handler.add_result(dist, id)) {
                        handler.stats.nheap_updates++;
                        nup++;
                        threshold = handler.threshold;
                    }
                    codes += code_size;
                }
                return nup;
            }
        }

        if (prefer_lut_scan(codec_, list_size)) {
            prepare_lut();
            return faiss::InvertedListScanner::scan_codes(
                    list_size,
                    codes,
                    ids,
                    handler);
        }

        size_t nup = 0;
        float threshold = handler.threshold;
        for (size_t j = 0; j < list_size; ++j) {
            const int64_t id = store_pairs ? faiss::lo_build(list_no, j) : ids[j];
            if (sel != nullptr && !sel->is_member(id)) {
                codes += code_size;
                continue;
            }
            handler.stats.scan_cnt++;
            float dist = 0.0f;
            if constexpr (std::is_same_v<Codec, epq::IndexEPQ>) {
                dist = codec_.exact_distance_from_packed_code_transformed(
                        codes,
                        transformed_residual_.data());
            } else {
                codec_.sa_decode(1, codes, decoded_.data());
                for (int i = 0; i < codec_.d; ++i) {
                    const float diff =
                            decoded_[static_cast<size_t>(i)] -
                            residual_query_[static_cast<size_t>(i)];
                    dist += diff * diff;
                }
            }
            if (dist < threshold) {
                if (handler.add_result(dist, id)) {
                    handler.stats.nheap_updates++;
                    nup++;
                    threshold = handler.threshold;
                }
            }
            codes += code_size;
        }
        return nup;
    }

   private:
    void prepare_lut() const {
        if (lut_ready_) {
            return;
        }
        codec_.compute_adc_lut_from_transformed(
                transformed_residual_.data(),
                lut_.data());
        lut_ready_ = true;
    }

    const faiss::IndexIVF& parent_;
    const Codec& codec_;
    std::vector<float> query_;
    std::vector<float> residual_query_;
    std::vector<float> centroid_;
    std::vector<float> transformed_query_;
    std::vector<float> transformed_residual_;
    mutable std::vector<float> lut_;
    mutable std::vector<float> decoded_;
    mutable bool lut_ready_ = false;
};

}  // namespace

namespace epq {

EpqIvfSearchMode epq_ivf_search_mode() {
    return static_cast<EpqIvfSearchMode>(
            epq_ivf_search_mode_storage().load(std::memory_order_relaxed));
}

void set_epq_ivf_search_mode(EpqIvfSearchMode mode) {
    epq_ivf_search_mode_storage().store(
            static_cast<int>(mode),
            std::memory_order_relaxed);
}

const char* epq_ivf_mode_name(EpqIvfSearchMode mode) {
    switch (mode) {
        case EpqIvfSearchMode::kScalarLut:
            return "scalar_lut";
        case EpqIvfSearchMode::kFallbackScanner:
            return "fallback_scanner";
        case EpqIvfSearchMode::kExactDecode:
            return "exact_decode";
    }
    return "unknown";
}

template <typename Codec>
IndexIVFCodec<Codec>::IndexIVFCodec(
        std::unique_ptr<Codec> codec,
        faiss::Index* quantizer,
        size_t nlist,
        std::string label)
        : faiss::IndexIVF(
                  quantizer,
                  static_cast<size_t>(codec->d),
                  nlist,
                  codec->sa_code_size(),
                  faiss::METRIC_L2,
                  true),
          codec_(std::move(codec)),
          label_(std::move(label)) {
    own_fields = true;
    by_residual = true;
    if constexpr (std::is_same_v<Codec, epq::IndexEPQ>) {
        epq_assignment_cache_.resize(nlist);
        epq_assignment_cache_mutexes_ = std::make_unique<std::mutex[]>(nlist);
    }
}

template <typename Codec>
void IndexIVFCodec<Codec>::train_encoder(
        faiss::idx_t n,
        const float* x,
        const faiss::idx_t* assign) {
    if constexpr (std::is_same_v<Codec, epq::IndexEPQ>) {
        last_query_weighted_training_diagnostics_ = nlohmann::json::object();
        if (codec_->ivf_query_weighted_sampling &&
            !query_weighted_training_exposure_.empty() && assign != nullptr) {
            std::vector<float> sampled_x;
            auto diag = build_query_weighted_training_sample(
                    *codec_,
                    n,
                    d,
                    x,
                    assign,
                    nlist,
                    query_weighted_training_exposure_,
                    sampled_x);
            last_query_weighted_training_diagnostics_ = diag;
            if (diag.value("applied", false) && !sampled_x.empty()) {
                codec_->train(n, sampled_x.data());
                sync_code_size_from_codec();
                return;
            }
        }
    }
    codec_->train(n, x);
    sync_code_size_from_codec();
}

template <typename Codec>
void IndexIVFCodec<Codec>::encode_vectors(
        faiss::idx_t n,
        const float* x,
        const faiss::idx_t* list_nos,
        uint8_t* codes,
        bool include_listno) const {
    FAISS_THROW_IF_NOT_MSG(
            codec_->sa_code_size() == code_size,
            "IVF codec code_size is stale; sync after codec training");
    const size_t coarse_size = include_listno ? coarse_code_size() : 0;
    std::vector<float> residuals(static_cast<size_t>(n) * static_cast<size_t>(d));
    bool has_unassigned = false;
    for (faiss::idx_t i = 0; i < n; ++i) {
        if (list_nos[static_cast<size_t>(i)] < 0) {
            has_unassigned = true;
            break;
        }
    }
    if (!has_unassigned) {
        quantizer->compute_residual_n(n, x, residuals.data(), list_nos);
    } else {
        for (faiss::idx_t i = 0; i < n; ++i) {
            float* dst =
                    residuals.data() + static_cast<size_t>(i) * static_cast<size_t>(d);
            if (list_nos[static_cast<size_t>(i)] < 0) {
                std::fill(dst, dst + d, 0.0f);
                continue;
            }
            quantizer->compute_residual(
                    x + static_cast<size_t>(i) * static_cast<size_t>(d),
                    dst,
                    list_nos[static_cast<size_t>(i)]);
        }
    }
    if (!include_listno) {
        codec_->sa_encode(n, residuals.data(), codes);
        return;
    }
    std::vector<uint8_t> payload(static_cast<size_t>(n) * code_size);
    codec_->sa_encode(n, residuals.data(), payload.data());
    for (faiss::idx_t i = 0; i < n; ++i) {
        uint8_t* dst = codes + static_cast<size_t>(i) * (coarse_size + code_size);
        encode_listno(list_nos[static_cast<size_t>(i)], dst);
        std::memcpy(
                dst + coarse_size,
                payload.data() + static_cast<size_t>(i) * code_size,
                code_size);
    }
}

template <typename Codec>
void IndexIVFCodec<Codec>::decode_vectors(
        faiss::idx_t n,
        const uint8_t* codes,
        const faiss::idx_t* list_nos,
        float* x) const {
    codec_->sa_decode(n, codes, x);
    if (!by_residual) {
        return;
    }
    std::vector<float> centroid(static_cast<size_t>(d), 0.0f);
    for (faiss::idx_t i = 0; i < n; ++i) {
        const auto list_no = list_nos[static_cast<size_t>(i)];
        if (list_no < 0) {
            continue;
        }
        quantizer->reconstruct(list_no, centroid.data());
        float* row = x + static_cast<size_t>(i) * static_cast<size_t>(d);
        for (int j = 0; j < d; ++j) {
            row[j] += centroid[static_cast<size_t>(j)];
        }
    }
}

template <typename Codec>
void IndexIVFCodec<Codec>::reset() {
    faiss::IndexIVF::reset();
    invalidate_epq_assignment_cache();
}

template <typename Codec>
faiss::InvertedListScanner* IndexIVFCodec<Codec>::get_InvertedListScanner(
        bool store_pairs,
        const faiss::IDSelector* sel,
        const faiss::IVFSearchParameters*) const {
    return new IVFCodecScanner<Codec>(*this, *codec_, store_pairs, sel);
}

template <typename Codec>
void IndexIVFCodec<Codec>::search_preassigned(
        faiss::idx_t n,
        const float* x,
        faiss::idx_t k,
        const faiss::idx_t* assign,
        const float* centroid_dis,
        float* distances,
        faiss::idx_t* labels,
        bool store_pairs,
        const faiss::IVFSearchParameters* params,
        faiss::IndexIVFStats* stats) const {
    (void)centroid_dis;
    if constexpr (!std::is_same_v<Codec, epq::IndexEPQ>) {
        faiss::IndexIVF::search_preassigned(
                n,
                x,
                k,
                assign,
                centroid_dis,
                distances,
                labels,
                store_pairs,
                params,
                stats);
        return;
    } else {
        FAISS_THROW_IF_NOT(k > 0);
        FAISS_THROW_IF_NOT_MSG(is_trained, "IVF index is not trained");
        FAISS_THROW_IF_NOT_MSG(invlists, "IVF index has no inverted lists");

        const EpqIvfSearchMode search_mode = epq_ivf_search_mode();
        if (search_mode == EpqIvfSearchMode::kFallbackScanner) {
            faiss::IndexIVF::search_preassigned(
                    n,
                    x,
                    k,
                    assign,
                    centroid_dis,
                    distances,
                    labels,
                    store_pairs,
                    params,
                    stats);
            return;
        }

        const faiss::idx_t cur_nprobe = std::min<faiss::idx_t>(
                static_cast<faiss::idx_t>(nlist),
                params ? params->nprobe : this->nprobe);
        FAISS_THROW_IF_NOT(cur_nprobe > 0);

        const faiss::idx_t cur_max_codes =
                params ? params->max_codes : this->max_codes;
        const bool ensure_topk_full = params ? params->ensure_topk_full : false;
        const auto* sel = params ? params->sel : nullptr;
        const void* inverted_list_context =
                params ? params->inverted_list_context : nullptr;
        const int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
        const bool no_heap_init =
                (this->parallel_mode & PARALLEL_MODE_NO_HEAP_INIT) != 0;

        if (metric_type != faiss::METRIC_L2 || store_pairs || sel != nullptr ||
            cur_max_codes != 0 || ensure_topk_full || invlists->use_iterator ||
            inverted_list_context != nullptr || no_heap_init ||
            (pmode != 0 && pmode != 3)) {
            faiss::IndexIVF::search_preassigned(
                    n,
                    x,
                    k,
                    assign,
                    centroid_dis,
                    distances,
                    labels,
                    store_pairs,
                    params,
                    stats);
            return;
        }

        size_t nlistv = 0;
        size_t ndis = 0;
        size_t nheap = 0;
        const bool collect_diag = collect_epq_ivf_diagnostics();
        size_t diag_list_count = 0;
        size_t diag_lut_lists = 0;
        size_t diag_decode_lists = 0;
        size_t diag_total_codes = 0;
        size_t diag_lut_codes = 0;
        size_t diag_decode_codes = 0;
        double diag_setup_time = 0.0;
        double diag_lut_build_time = 0.0;
        double diag_lut_scan_time = 0.0;
        double diag_decode_scan_time = 0.0;
        const bool do_parallel = pmode == 3 && omp_get_max_threads() >= 2 && n > 1;
        using HeapForL2 = faiss::CMax<float, faiss::idx_t>;

#pragma omp parallel for if (do_parallel) reduction(+ : nlistv, ndis, nheap, diag_list_count, diag_lut_lists, diag_decode_lists, diag_total_codes, diag_lut_codes, diag_decode_codes, diag_setup_time, diag_lut_build_time, diag_lut_scan_time, diag_decode_scan_time)
        for (faiss::idx_t qi = 0; qi < n; ++qi) {
            float* simi =
                    distances + static_cast<size_t>(qi) * static_cast<size_t>(k);
            faiss::idx_t* idxi =
                    labels + static_cast<size_t>(qi) * static_cast<size_t>(k);
            faiss::heap_heapify<HeapForL2>(k, simi, idxi);
            float threshold = simi[0];

            std::vector<float> query(static_cast<size_t>(d), 0.0f);
            std::copy(
                    x + static_cast<size_t>(qi) * static_cast<size_t>(d),
                    x + static_cast<size_t>(qi + 1) * static_cast<size_t>(d),
                    query.begin());
            std::vector<float> residual_query(static_cast<size_t>(d), 0.0f);
            std::vector<float> centroid(static_cast<size_t>(d), 0.0f);
            std::vector<float> transformed_query(static_cast<size_t>(d), 0.0f);
            std::vector<float> transformed_residual(static_cast<size_t>(d), 0.0f);
            std::vector<float> lut(codec_->adc_lut_size(), 0.0f);
            std::vector<float> decoded(static_cast<size_t>(d), 0.0f);
            codec_->transform_vector(
                    x + static_cast<size_t>(qi) * static_cast<size_t>(d),
                    transformed_query.data());

            for (faiss::idx_t pi = 0; pi < cur_nprobe; ++pi) {
                const size_t qp = static_cast<size_t>(qi) *
                                static_cast<size_t>(cur_nprobe) +
                        static_cast<size_t>(pi);
                const faiss::idx_t key = assign[qp];
                if (key < 0) {
                    continue;
                }
                FAISS_THROW_IF_NOT_FMT(
                        key < static_cast<faiss::idx_t>(nlist),
                        "Invalid key=%" PRId64 " nlist=%zd\n",
                        key,
                        nlist);
                if (invlists->is_empty(static_cast<size_t>(key))) {
                    continue;
                }

                const size_t list_size = invlists->list_size(static_cast<size_t>(key));
                if (list_size == 0) {
                    continue;
                }
                nlistv++;
                ndis += list_size;
                if (collect_diag) {
                    diag_list_count++;
                    diag_total_codes += list_size;
                }

                const auto setup_t0 = collect_diag
                        ? std::chrono::steady_clock::now()
                        : std::chrono::steady_clock::time_point{};
                if (by_residual) {
                    quantizer->reconstruct(key, centroid.data());
                    for (int dim = 0; dim < d; ++dim) {
                        residual_query[static_cast<size_t>(dim)] =
                                query[static_cast<size_t>(dim)] -
                                centroid[static_cast<size_t>(dim)];
                    }
                    codec_->transform_vector(
                            residual_query.data(), transformed_residual.data());
                } else {
                    transformed_residual = transformed_query;
                    residual_query = query;
                }
                if (collect_diag) {
                    diag_setup_time += std::chrono::duration<double>(
                                               std::chrono::steady_clock::now() -
                                               setup_t0)
                                               .count();
                }

                const bool use_lut =
                        !epq_ivf_mode_uses_exact_decode(search_mode) &&
                        prefer_lut_scan(*codec_, list_size);
                if (use_lut) {
                    if (collect_diag) {
                        diag_lut_lists++;
                        diag_lut_codes += list_size;
                    }
                    const auto lut_build_t0 = collect_diag
                            ? std::chrono::steady_clock::now()
                            : std::chrono::steady_clock::time_point{};
                    codec_->compute_adc_lut_from_transformed(
                            transformed_residual.data(),
                            lut.data());
                    if (collect_diag) {
                        diag_lut_build_time += std::chrono::duration<double>(
                                                       std::chrono::steady_clock::now() -
                                                       lut_build_t0)
                                                       .count();
                    }
                } else if (collect_diag) {
                    diag_decode_lists++;
                    diag_decode_codes += list_size;
                }

                faiss::InvertedLists::ScopedCodes scodes(invlists, key);
                faiss::InvertedLists::ScopedIds sids(invlists, key);
                const uint8_t* codes = scodes.get();
                const faiss::idx_t* ids = sids.get();
                const uint16_t* cached_assignments =
                        get_epq_assignment_cache(key, codes, list_size);
                const size_t assignment_stride = codec_->structure().group_count();
                if (use_lut) {
                    const auto lut_scan_t0 = collect_diag
                            ? std::chrono::steady_clock::now()
                            : std::chrono::steady_clock::time_point{};
                    for (size_t j = 0; j < list_size; ++j) {
                        const float dist =
                                codec_->adc_distance_from_assignments(
                                        cached_assignments + j * assignment_stride,
                                        lut.data());
                        if (dist < threshold) {
                            faiss::heap_replace_top<HeapForL2>(
                                    k,
                                    simi,
                                    idxi,
                                    dist,
                                    ids[j]);
                            threshold = simi[0];
                            nheap++;
                        }
                    }
                    if (collect_diag) {
                        diag_lut_scan_time += std::chrono::duration<double>(
                                                      std::chrono::steady_clock::now() -
                                                      lut_scan_t0)
                                                      .count();
                    }
                    continue;
                }
                const auto decode_scan_t0 = collect_diag
                        ? std::chrono::steady_clock::now()
                        : std::chrono::steady_clock::time_point{};
                for (size_t j = 0; j < list_size; ++j) {
                    const float dist =
                            codec_->exact_distance_from_assignments_transformed(
                                    cached_assignments + j * assignment_stride,
                                    transformed_residual.data());
                    if (dist < threshold) {
                        faiss::heap_replace_top<HeapForL2>(
                                k,
                                simi,
                                idxi,
                                dist,
                                ids[j]);
                        threshold = simi[0];
                        nheap++;
                    }
                }
                if (collect_diag) {
                    diag_decode_scan_time += std::chrono::duration<double>(
                                                     std::chrono::steady_clock::now() -
                                                     decode_scan_t0)
                                                     .count();
                }
            }

            faiss::heap_reorder<HeapForL2>(k, simi, idxi);
        }

        if (stats == nullptr) {
            stats = &faiss::indexIVF_stats;
        }
        stats->nq += n;
        stats->nlist += nlistv;
        stats->ndis += ndis;
        stats->nheap_updates += nheap;
        if (collect_diag) {
            EpqIvfFastPathDiagnostics diag;
            diag.query_count = static_cast<size_t>(n);
            diag.list_count = diag_list_count;
            diag.lut_lists = diag_lut_lists;
            diag.decode_lists = diag_decode_lists;
            diag.total_codes = diag_total_codes;
            diag.lut_codes = diag_lut_codes;
            diag.decode_codes = diag_decode_codes;
            diag.setup_time = diag_setup_time;
            diag.lut_build_time = diag_lut_build_time;
            diag.lut_scan_time = diag_lut_scan_time;
            diag.decode_scan_time = diag_decode_scan_time;
            std::lock_guard<std::mutex> guard(epq_diag_mu_);
            last_epq_ivf_diagnostics_ = diag.to_json();
        }
    }
}

template <typename Codec>
void IndexIVFCodec<Codec>::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    const uint8_t* code =
            faiss::InvertedLists::ScopedCodes(invlists, list_no, offset).get();
    const faiss::idx_t list = static_cast<faiss::idx_t>(list_no);
    decode_vectors(1, code, &list, recons);
}

template <typename Codec>
const Codec& IndexIVFCodec<Codec>::codec() const {
    return *codec_;
}

template <typename Codec>
Codec& IndexIVFCodec<Codec>::codec_mutable() {
    return *codec_;
}

template <typename Codec>
const std::string& IndexIVFCodec<Codec>::label() const {
    return label_;
}

template <typename Codec>
nlohmann::json IndexIVFCodec<Codec>::last_epq_ivf_diagnostics() const {
    if constexpr (!std::is_same_v<Codec, epq::IndexEPQ>) {
        return nlohmann::json();
    } else {
        std::lock_guard<std::mutex> guard(epq_diag_mu_);
        return last_epq_ivf_diagnostics_;
    }
}

template <typename Codec>
void IndexIVFCodec<Codec>::set_query_weighted_training_exposure(
        std::vector<float> exposure) {
    if constexpr (std::is_same_v<Codec, epq::IndexEPQ>) {
        query_weighted_training_exposure_ = std::move(exposure);
    } else {
        (void)exposure;
    }
}

template <typename Codec>
void IndexIVFCodec<Codec>::clear_query_weighted_training_exposure() {
    if constexpr (std::is_same_v<Codec, epq::IndexEPQ>) {
        query_weighted_training_exposure_.clear();
        last_query_weighted_training_diagnostics_ = nlohmann::json::object();
    }
}

template <typename Codec>
nlohmann::json IndexIVFCodec<Codec>::last_query_weighted_training_diagnostics()
        const {
    if constexpr (!std::is_same_v<Codec, epq::IndexEPQ>) {
        return nlohmann::json();
    } else {
        return last_query_weighted_training_diagnostics_;
    }
}

template <typename Codec>
void IndexIVFCodec<Codec>::invalidate_epq_assignment_cache() {
    if constexpr (!std::is_same_v<Codec, epq::IndexEPQ>) {
        return;
    } else {
        for (auto& entry : epq_assignment_cache_) {
            entry.source_codes = nullptr;
            entry.list_size = 0;
            entry.assignments.clear();
        }
    }
}

template <typename Codec>
const uint16_t* IndexIVFCodec<Codec>::get_epq_assignment_cache(
        faiss::idx_t list_no,
        const uint8_t* codes,
        size_t list_size) const {
    if constexpr (!std::is_same_v<Codec, epq::IndexEPQ>) {
        (void)list_no;
        (void)codes;
        (void)list_size;
        return nullptr;
    } else {
        const size_t list_index = static_cast<size_t>(list_no);
        auto& entry = epq_assignment_cache_[list_index];
        std::lock_guard<std::mutex> guard(epq_assignment_cache_mutexes_[list_index]);
        const size_t assignment_stride = codec_->structure().group_count();
        const size_t assignment_count = list_size * assignment_stride;
        if (entry.source_codes == codes && entry.list_size == list_size &&
            entry.assignments.size() == assignment_count) {
            return entry.assignments.data();
        }
        entry.source_codes = codes;
        entry.list_size = list_size;
        entry.assignments.resize(assignment_count);
        for (size_t j = 0; j < list_size; ++j) {
            codec_->unpack_code_assignments(
                    codes + j * code_size,
                    entry.assignments.data() + j * assignment_stride);
        }
        return entry.assignments.data();
    }
}

template <typename Codec>
void IndexIVFCodec<Codec>::sync_code_size_from_codec() {
    invalidate_epq_assignment_cache();
    const size_t trained_code_size = codec_->sa_code_size();
    FAISS_THROW_IF_NOT_MSG(
            trained_code_size > 0,
            "trained codec must expose a positive code_size");
    if (trained_code_size == code_size && invlists != nullptr &&
        invlists->code_size == trained_code_size) {
        return;
    }
    // Keep this path limited to IVF code storage updates; callers own any
    // training-time diagnostics or auxiliary caches tied to codec training.
    FAISS_THROW_IF_NOT_MSG(
            ntotal == 0,
            "cannot change IVF code_size after vectors have been added");
    code_size = trained_code_size;
    replace_invlists(new faiss::ArrayInvertedLists(nlist, code_size), true);
}

template class IndexIVFCodec<IndexEPQ>;
template class IndexIVFCodec<IndexBAPQ>;
template class IndexIVFCodec<IndexAREPQ>;
template class IndexIVFCodec<IndexDPOPQ>;
template class IndexIVFCodec<IndexVAQ>;

}  // namespace epq
