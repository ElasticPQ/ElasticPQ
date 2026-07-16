#include "epq/structure.h"

#include <fstream>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace epq {
namespace {

[[noreturn]] void throw_validation(const std::string& message) {
    throw std::invalid_argument("epq::Structure: " + message);
}

}  // namespace

size_t GroupSpec::ksub() const {
    if (nbits <= 0) {
        return 1;
    }
    return size_t{1} << static_cast<size_t>(nbits);
}

void Structure::validate(int min_bits, int max_bits) const {
    if (d <= 0) {
        throw_validation("d must be positive");
    }
    if (total_bits < 0) {
        throw_validation("total_bits must be non-negative");
    }
    if (groups.empty()) {
        throw_validation("groups must not be empty");
    }
    if (min_bits < 0 || max_bits < min_bits) {
        throw_validation("invalid bit bounds");
    }

    std::vector<int> seen;
    seen.reserve(static_cast<size_t>(d));
    int sum_bits = 0;
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto& group = groups[i];
        if (group.dims.empty()) {
            throw_validation("group " + std::to_string(i) + " is empty");
        }
        if (group.nbits < min_bits || group.nbits > max_bits) {
            throw_validation(
                    "group " + std::to_string(i) + " has nbits outside configured range");
        }
        sum_bits += group.nbits;
        for (int dim : group.dims) {
            if (dim < 0 || dim >= d) {
                throw_validation("group " + std::to_string(i) + " contains invalid dim");
            }
            seen.push_back(dim);
        }
    }

    if (sum_bits != total_bits) {
        throw_validation("sum(group.nbits) must equal total_bits");
    }
    if (seen.size() != static_cast<size_t>(d)) {
        throw_validation("groups must cover each dimension exactly once");
    }
    std::unordered_set<int> uniq(seen.begin(), seen.end());
    if (uniq.size() != seen.size()) {
        throw_validation("groups contain duplicated dimensions");
    }
}

size_t Structure::group_count() const noexcept {
    return groups.size();
}

size_t Structure::code_size() const noexcept {
    return static_cast<size_t>((total_bits + 7) / 8);
}

int Structure::max_nbits() const noexcept {
    int out = 0;
    for (const auto& group : groups) {
        out = std::max(out, group.nbits);
    }
    return out;
}

std::vector<int> Structure::flatten_dims() const {
    std::vector<int> perm;
    perm.reserve(static_cast<size_t>(d));
    for (const auto& group : groups) {
        perm.insert(perm.end(), group.dims.begin(), group.dims.end());
    }
    return perm;
}

std::vector<int> Structure::group_sizes() const {
    std::vector<int> sizes;
    sizes.reserve(groups.size());
    for (const auto& group : groups) {
        sizes.push_back(static_cast<int>(group.dims.size()));
    }
    return sizes;
}

std::vector<std::vector<int>> Structure::contiguous_groups() const {
    std::vector<std::vector<int>> blocks;
    blocks.reserve(groups.size());
    int offset = 0;
    for (const auto& group : groups) {
        std::vector<int> dims(group.dims.size());
        for (size_t j = 0; j < dims.size(); ++j) {
            dims[j] = offset + static_cast<int>(j);
        }
        offset += static_cast<int>(dims.size());
        blocks.push_back(std::move(dims));
    }
    return blocks;
}

nlohmann::json Structure::to_json() const {
    validate();
    nlohmann::json j;
    j["format_version"] = format_version;
    j["d"] = d;
    j["total_bits"] = total_bits;
    j["meta"] = meta;
    j["groups"] = nlohmann::json::array();
    for (const auto& group : groups) {
        j["groups"].push_back(
                {{"dims", group.dims}, {"nbits", group.nbits}});
    }
    return j;
}

Structure Structure::from_json(const nlohmann::json& j) {
    Structure structure;
    structure.format_version = j.value("format_version", 1);
    structure.d = j.at("d").get<int>();
    structure.total_bits = j.contains("total_bits") ? j.at("total_bits").get<int>()
                                                    : j.at("B").get<int>();
    structure.meta = j.value("meta", nlohmann::json(nullptr));

    if (j.contains("groups") && !j.at("groups").empty() &&
        j.at("groups").front().is_object()) {
        for (const auto& group_json : j.at("groups")) {
            GroupSpec group;
            group.dims = group_json.at("dims").get<std::vector<int>>();
            group.nbits = group_json.at("nbits").get<int>();
            structure.groups.push_back(std::move(group));
        }
    } else {
        const auto dims_groups = j.at("groups").get<std::vector<std::vector<int>>>();
        const auto bits = j.contains("nbits") ? j.at("nbits").get<std::vector<int>>()
                                              : j.at("bits").get<std::vector<int>>();
        if (dims_groups.size() != bits.size()) {
            throw_validation("legacy JSON has mismatched groups/nbits lengths");
        }
        for (size_t i = 0; i < dims_groups.size(); ++i) {
            structure.groups.push_back(GroupSpec{dims_groups[i], bits[i]});
        }
    }

    structure.validate();
    return structure;
}

Structure Structure::load_json(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open structure JSON: " + path);
    }
    nlohmann::json j;
    in >> j;
    return from_json(j);
}

void Structure::save_json(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write structure JSON: " + path);
    }
    out << to_json().dump(2) << '\n';
}

}  // namespace epq
