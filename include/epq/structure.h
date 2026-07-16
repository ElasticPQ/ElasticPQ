#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace epq {

struct GroupSpec {
    std::vector<int> dims;
    int nbits = 0;

    size_t ksub() const;
};

struct Structure {
    int d = 0;
    int total_bits = 0;
    int format_version = 1;
    std::vector<GroupSpec> groups;
    nlohmann::json meta = nullptr;

    void validate(int min_bits = 0, int max_bits = 16) const;

    size_t group_count() const noexcept;
    size_t code_size() const noexcept;
    int max_nbits() const noexcept;
    std::vector<int> flatten_dims() const;
    std::vector<int> group_sizes() const;
    std::vector<std::vector<int>> contiguous_groups() const;

    nlohmann::json to_json() const;

    static Structure from_json(const nlohmann::json& j);
    static Structure load_json(const std::string& path);
    void save_json(const std::string& path) const;
};

}  // namespace epq
