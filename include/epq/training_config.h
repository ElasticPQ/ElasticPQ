#pragma once

#include <filesystem>
#include <memory>

#include <nlohmann/json.hpp>

namespace epq {

class StructureBuilder;
class IndexEPQ;

nlohmann::json load_json_file(const std::filesystem::path& path);

bool should_auto_reuse_structure(
        const nlohmann::json& config,
        bool default_value = true);

bool apply_faiss_runtime_config(const nlohmann::json& config);

std::shared_ptr<StructureBuilder> make_structure_builder_from_config(
        const nlohmann::json& config,
        const std::filesystem::path& config_dir = {});

void apply_index_training_config(
        IndexEPQ& index,
        const nlohmann::json& config);

}  // namespace epq
