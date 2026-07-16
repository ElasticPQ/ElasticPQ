#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <faiss/IndexIVF.h>
#include <nlohmann/json.hpp>

namespace epq {

class IndexBAPQ;
class IndexDPOPQ;
class IndexEPQ;
class IndexAREPQ;
class IndexVAQ;

enum class EpqIvfSearchMode : int {
    kScalarLut = 0,
    kFallbackScanner = 1,
    kExactDecode = 2,
};

EpqIvfSearchMode epq_ivf_search_mode();
void set_epq_ivf_search_mode(EpqIvfSearchMode mode);
const char* epq_ivf_mode_name(EpqIvfSearchMode mode);

template <typename Codec>
class IndexIVFCodec final : public faiss::IndexIVF {
   public:
    IndexIVFCodec(
            std::unique_ptr<Codec> codec,
            faiss::Index* quantizer,
            size_t nlist,
            std::string label);

    void train_encoder(faiss::idx_t n, const float* x, const faiss::idx_t*) override;
    void encode_vectors(
            faiss::idx_t n,
            const float* x,
            const faiss::idx_t* list_nos,
            uint8_t* codes,
            bool include_listno = false) const override;
    void decode_vectors(
            faiss::idx_t n,
            const uint8_t* codes,
            const faiss::idx_t* list_nos,
            float* x) const override;
    void reset() override;
    faiss::InvertedListScanner* get_InvertedListScanner(
            bool store_pairs = false,
            const faiss::IDSelector* sel = nullptr,
            const faiss::IVFSearchParameters* params = nullptr) const override;
    void search_preassigned(
            faiss::idx_t n,
            const float* x,
            faiss::idx_t k,
            const faiss::idx_t* assign,
            const float* centroid_dis,
            float* distances,
            faiss::idx_t* labels,
            bool store_pairs,
            const faiss::IVFSearchParameters* params = nullptr,
            faiss::IndexIVFStats* stats = nullptr) const override;
    void reconstruct_from_offset(
            int64_t list_no,
            int64_t offset,
            float* recons) const override;

    const Codec& codec() const;
    Codec& codec_mutable();
    const std::string& label() const;
    nlohmann::json last_epq_ivf_diagnostics() const;
    void set_query_weighted_training_exposure(std::vector<float> exposure);
    void clear_query_weighted_training_exposure();
    nlohmann::json last_query_weighted_training_diagnostics() const;

   private:
    struct EpqAssignmentListCache {
        const uint8_t* source_codes = nullptr;
        size_t list_size = 0;
        std::vector<uint16_t> assignments;
    };

    void invalidate_epq_assignment_cache();
    const uint16_t* get_epq_assignment_cache(
            faiss::idx_t list_no,
            const uint8_t* codes,
            size_t list_size) const;
    void sync_code_size_from_codec();

    std::unique_ptr<Codec> codec_;
    std::string label_;
    mutable std::mutex epq_diag_mu_;
    mutable nlohmann::json last_epq_ivf_diagnostics_ = nlohmann::json::object();
    std::vector<float> query_weighted_training_exposure_;
    nlohmann::json last_query_weighted_training_diagnostics_ =
            nlohmann::json::object();
    mutable std::vector<EpqAssignmentListCache> epq_assignment_cache_;
    mutable std::unique_ptr<std::mutex[]> epq_assignment_cache_mutexes_;
};

extern template class IndexIVFCodec<IndexEPQ>;
extern template class IndexIVFCodec<IndexBAPQ>;
extern template class IndexIVFCodec<IndexAREPQ>;
extern template class IndexIVFCodec<IndexDPOPQ>;
extern template class IndexIVFCodec<IndexVAQ>;

}  // namespace epq
