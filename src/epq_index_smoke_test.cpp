#include "epq/index_epq.h"
#include "epq/index_arepq.h"

#include <filesystem>
#include <iostream>
#include <vector>

#include <nlohmann/json.hpp>

#include "epq/structure.h"
#include "epq/structure_builder.h"

namespace {

std::vector<float> make_train_set() {
    return {
            0.0f, 0.1f, 0.0f, 0.1f, 4.0f, 4.1f, 4.0f, 4.1f,
            0.1f, 0.0f, 0.2f, 0.0f, 4.2f, 4.0f, 4.1f, 3.9f,
            -0.1f, 0.0f, 0.0f, 0.2f, 3.8f, 4.0f, 4.2f, 4.1f,
            8.0f, 8.1f, 8.0f, 8.1f, -2.0f, -2.1f, -2.0f, -2.1f,
            8.1f, 8.0f, 8.2f, 8.0f, -1.8f, -2.0f, -2.1f, -2.2f,
            7.9f, 8.0f, 8.1f, 8.2f, -2.2f, -2.1f, -1.9f, -2.0f,
    };
}

std::vector<float> make_base_set() {
    return {
            0.0f, 0.0f, 0.1f, 0.0f, 4.0f, 4.0f, 4.1f, 4.0f,
            0.2f, -0.1f, 0.0f, 0.1f, 4.1f, 4.2f, 4.0f, 3.9f,
            8.0f, 8.0f, 8.1f, 8.0f, -2.0f, -2.0f, -2.1f, -2.0f,
            8.2f, 7.9f, 8.0f, 8.1f, -1.9f, -2.2f, -2.0f, -2.1f,
    };
}

}  // namespace

int main() {
    using json = nlohmann::json;

    constexpr int d = 8;
    constexpr faiss::idx_t ntrain = 6;
    constexpr faiss::idx_t nb = 4;
    constexpr faiss::idx_t nq = 2;

    const auto xt = make_train_set();
    const auto xb = make_base_set();
    const std::vector<float> xq(xb.begin(), xb.begin() + nq * d);

    epq::Structure fixed_structure;
    fixed_structure.d = d;
    fixed_structure.total_bits = 16;
    fixed_structure.groups = {
            {{0, 1, 2, 3}, 8},
            {{4, 5, 6, 7}, 8},
    };
    fixed_structure.meta = {{"source", "smoke"}};

    const std::filesystem::path structure_path =
            std::filesystem::temp_directory_path() / "epq_cpp_smoke_structure.json";
    fixed_structure.save_json(structure_path.string());
    const epq::Structure reloaded =
            epq::Structure::load_json(structure_path.string());

    auto builder = std::make_shared<epq::FixedStructureBuilder>(reloaded);
    epq::IndexEPQ index(d, 16, builder);
    index.use_uneven_transform = true;
    index.transform_niter = 2;
    index.kmeans_niter = 10;
    index.kmeans_nredo = 1;
    index.train(ntrain, xt.data());
    index.add(nb, xb.data());

    std::vector<float> distances(static_cast<size_t>(nq) * 2);
    std::vector<faiss::idx_t> labels(static_cast<size_t>(nq) * 2);
    index.search(nq, xq.data(), 2, distances.data(), labels.data());

    if (labels[0] != 0 || labels[2] != 1) {
        std::cerr << "unexpected ADC top-1 labels: "
                  << labels[0] << ", " << labels[2] << '\n';
        return 1;
    }

    epq::SearchParametersEPQ sdc_params;
    sdc_params.mode = epq::SearchMode::kSDC;
    std::vector<float> sdc_distances(static_cast<size_t>(nq) * 2);
    std::vector<faiss::idx_t> sdc_labels(static_cast<size_t>(nq) * 2);
    index.search(
            nq,
            xq.data(),
            2,
            sdc_distances.data(),
            sdc_labels.data(),
            &sdc_params);

    if (sdc_labels[0] != 0 || sdc_labels[2] != 1) {
        std::cerr << "unexpected SDC top-1 labels: "
                  << sdc_labels[0] << ", " << sdc_labels[2] << '\n';
        return 1;
    }

    std::vector<uint8_t> encoded(static_cast<size_t>(nb) * index.sa_code_size());
    index.sa_encode(nb, xb.data(), encoded.data());
    std::vector<float> decoded(static_cast<size_t>(nb) * d);
    index.sa_decode(nb, encoded.data(), decoded.data());

    epq::Structure arepq_structure;
    arepq_structure.d = d;
    arepq_structure.total_bits = 4;
    arepq_structure.groups = {
            {{0, 1, 2, 3}, 2},
            {{4, 5, 6, 7}, 2},
    };
    auto arepq_builder =
            std::make_shared<epq::FixedStructureBuilder>(arepq_structure);
    epq::IndexAREPQ arepq(d, 6, 2, 1, std::move(arepq_builder));
    arepq.main_index().use_uneven_transform = false;
    arepq.main_index().kmeans_niter = 3;
    arepq.main_index().kmeans_nredo = 1;
    arepq.tail_kmeans_niter = 3;
    arepq.tail_kmeans_nredo = 1;
    arepq.tail_alt_iters = 0;
    arepq.icm_iters = 0;
    arepq.train(ntrain, xt.data());
    arepq.add(nb, xb.data());
    const auto tail_memory = arepq.tail_memory_stats();
    if (tail_memory.payload_code_bytes != 1 ||
        tail_memory.serialized_codebook_bytes != 128 ||
        tail_memory.reconstruction_codebook_bytes != 128 ||
        tail_memory.transform_copy_bytes != 256 ||
        tail_memory.norm_table_bytes != 16 ||
        tail_memory.product_tail_table_bytes != 128 ||
        tail_memory.tail_pair_table_bytes != 0 ||
        tail_memory.query_lut_bytes_per_query != 16 ||
        tail_memory.resident_flat_code_bytes != 8 ||
        tail_memory.resident_search_model_bytes() != 272 ||
        tail_memory.resident_auxiliary_table_bytes() != 144 ||
        tail_memory.resident_model_bytes() != 656) {
        std::cerr << "unexpected AREPQ tail-memory accounting\n";
        return 1;
    }

    json report;
    report["status"] = "ok";
    report["structure"] = reloaded.to_json();
    report["adc_top1"] = {labels[0], labels[2]};
    report["sdc_top1"] = {sdc_labels[0], sdc_labels[2]};
    report["decoded_first"] = std::vector<float>(decoded.begin(), decoded.begin() + d);
    report["arepq_tail_memory"] = {
            {"payload_code_bytes", tail_memory.payload_code_bytes},
            {"serialized_codebook_bytes", tail_memory.serialized_codebook_bytes},
            {"product_tail_table_bytes", tail_memory.product_tail_table_bytes},
            {"resident_flat_code_bytes", tail_memory.resident_flat_code_bytes},
            {"transform_copy_bytes", tail_memory.transform_copy_bytes},
            {"resident_auxiliary_table_bytes", tail_memory.resident_auxiliary_table_bytes()},
            {"resident_model_bytes", tail_memory.resident_model_bytes()},
    };
    std::cout << report.dump(2) << '\n';
    return 0;
}
