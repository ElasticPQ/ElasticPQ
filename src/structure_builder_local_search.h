#pragma once

#include "structure_builder_internal.h"

namespace epq::structure_builder_internal {

Groups apply_relocate(const Groups& groups, int A, int B, int v);
Groups apply_swap(const Groups& groups, int A, int B, int v, int u);

struct BeamState {
    Groups groups;
    Bits bits;
    double J = 0.0;
};

struct SearchConfig {
    int donor_topk = 0;
    int recv_topk = 0;
    int dims_sample_per_group = 0;
    float suspicious_alpha = 0.0f;
    int n_relocate = 0;
    int n_swap_pairs = 0;
    int relocate_pair_limit = 0;
    int swap_pair_limit = 0;
    int shortlist_k = 0;
    int shortlist_per_pair = 0;
    double max_local_score = 0.0;
    double shift_lambda = 1.0;
};

struct Move {
    enum class Kind {
        kRelocate,
        kSwap,
    };

    Kind kind = Kind::kRelocate;
    std::vector<int> dims;
    std::pair<int, int> groups{0, 0};
    double dJ_struct = 0.0;
    double gml = 0.0;
    double score = 0.0;

    std::pair<int, int> pair_key() const {
        if (kind == Kind::kRelocate) {
            return groups;
        }
        return {
                std::min(groups.first, groups.second),
                std::max(groups.first, groups.second),
        };
    }
};

struct MoveBuildStats {
    size_t raw = 0;
    size_t shortlisted = 0;
    size_t pair_pruned = 0;
    size_t score_pruned = 0;
};

SearchConfig make_greedy_tail_search_config(const RefinedStructureBuilder& cfg);
std::vector<Move> build_shortlisted_moves(
        ProxyContext& proxy,
        const Groups& cur_groups,
        const Bits& cur_bits,
        const SearchConfig& cfg,
        std::mt19937& rng,
        MoveBuildStats* stats_out);

struct SeenWindow {
    explicit SeenWindow(int width) : width_(std::max(1, width)) {
        rounds_.push_back({});
    }

    std::optional<double> get_best(const PartitionKey& key) const {
        auto it = best_.find(key);
        if (it == best_.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    void set_best(const PartitionKey& key, double value) {
        auto& cur = rounds_.back();
        auto it = cur.find(key);
        if (it == cur.end() || value < it->second) {
            cur[key] = value;
        }
        auto best_it = best_.find(key);
        if (best_it == best_.end() || value < best_it->second) {
            best_[key] = value;
        }
    }

    void next_round() {
        rounds_.push_back({});
        while (static_cast<int>(rounds_.size()) > width_) {
            auto old = std::move(rounds_.front());
            rounds_.pop_front();
            for (const auto& [key, value] : old) {
                auto best_it = best_.find(key);
                if (best_it == best_.end() || best_it->second != value) {
                    continue;
                }
                double new_best = std::numeric_limits<double>::infinity();
                bool found = false;
                for (const auto& round : rounds_) {
                    auto it = round.find(key);
                    if (it == round.end()) {
                        continue;
                    }
                    new_best = std::min(new_best, it->second);
                    found = true;
                }
                if (found) {
                    best_[key] = new_best;
                } else {
                    best_.erase(key);
                }
            }
        }
    }

   private:
    int width_ = 1;
    std::deque<std::unordered_map<PartitionKey, double, PartitionKeyHash>> rounds_;
    std::unordered_map<PartitionKey, double, PartitionKeyHash> best_;
};

}  // namespace epq::structure_builder_internal
