#include "structure_builder_internal.h"
#include "structure_builder_local_search.h"

namespace epq::structure_builder_internal {

double marginal_gain(
        ProxyContext& proxy,
        const std::vector<int>& dims,
        int bits) {
    if (bits >= proxy.build_ctx.max_bits) {
        return 0.0;
    }
    return proxy.D(dims, bits) - proxy.D(dims, bits + 1);
}

double marginal_loss(
        ProxyContext& proxy,
        const std::vector<int>& dims,
        int bits) {
    if (bits <= proxy.build_ctx.min_bits) {
        return std::numeric_limits<double>::infinity();
    }
    return proxy.D(dims, bits - 1) - proxy.D(dims, bits);
}

struct RemovedDimEval {
    int dim = -1;
    std::vector<int> removed_group;
    double removed_D = 0.0;
    double loss_after_remove = std::numeric_limits<double>::infinity();
    double harm = 0.0;
};

struct GroupMoveCache {
    std::vector<RemovedDimEval> removals;
    std::vector<double> sample_weights;
};

GroupMoveCache build_group_move_cache(
        ProxyContext& proxy,
        const std::vector<int>& group,
        int bits,
        double group_D0,
        std::mt19937& rng,
        int sample_k,
        float alpha) {
    GroupMoveCache cache;
    std::vector<int> candidate_dims = group;
    if (sample_k > 0 && static_cast<int>(candidate_dims.size()) > sample_k) {
        candidate_dims = sample_vector(candidate_dims, sample_k, rng);
    }
    cache.removals.reserve(candidate_dims.size());
    double min_harm = std::numeric_limits<double>::infinity();
    for (int dim : candidate_dims) {
        auto removed_group = remove_one(group, dim);
        const double removed_D = proxy.D(removed_group, bits);
        const double loss_after_remove = marginal_loss(proxy, removed_group, bits);
        const double harm = group_D0 - removed_D;
        min_harm = std::min(min_harm, harm);
        cache.removals.push_back(RemovedDimEval{
                .dim = dim,
                .removed_group = std::move(removed_group),
                .removed_D = removed_D,
                .loss_after_remove = loss_after_remove,
                .harm = harm,
        });
    }
    if (alpha > 0.0f) {
        cache.sample_weights.reserve(cache.removals.size());
        for (const auto& removal : cache.removals) {
            cache.sample_weights.push_back(std::pow(removal.harm - min_harm + 1e-12, alpha));
        }
    }
    return cache;
}

const RemovedDimEval* choose_suspicious_dim(
        const GroupMoveCache& cache,
        std::mt19937& rng,
        float alpha) {
    if (cache.removals.empty()) {
        return nullptr;
    }
    if (cache.removals.size() == 1) {
        return &cache.removals.front();
    }
    if (alpha <= 0.0f) {
        const int idx = std::uniform_int_distribution<int>(
                0,
                static_cast<int>(cache.removals.size()) - 1)(rng);
        return &cache.removals[static_cast<size_t>(idx)];
    }
    std::discrete_distribution<int> pick(
            cache.sample_weights.begin(),
            cache.sample_weights.end());
    return &cache.removals[static_cast<size_t>(pick(rng))];
}

Groups apply_relocate(const Groups& groups, int A, int B, int v) {
    Groups out = groups;
    out[static_cast<size_t>(A)] = remove_one(out[static_cast<size_t>(A)], v);
    if (out[static_cast<size_t>(A)].empty()) {
        throw std::runtime_error("empty donor group");
    }
    out[static_cast<size_t>(B)].push_back(v);
    return out;
}

Groups apply_swap(const Groups& groups, int A, int B, int v, int u) {
    Groups out = groups;
    out[static_cast<size_t>(A)] = remove_one(out[static_cast<size_t>(A)], v);
    out[static_cast<size_t>(A)].push_back(u);
    out[static_cast<size_t>(B)] = remove_one(out[static_cast<size_t>(B)], u);
    out[static_cast<size_t>(B)].push_back(v);
    return out;
}

struct MBeamRunConfig {
    const char* label = "mbeam";
    int iters = 0;
    int patience = 0;
    double eps_improve = 0.0;
    int beam_width = 1;
    int per_state_eval_topk = 1;
    int seen_window = 1;
    int min_novel_children = 0;
    SearchConfig search;
};

struct MoveKey {
    Move::Kind kind = Move::Kind::kRelocate;
    int group_a = 0;
    int group_b = 0;
    int dim_a = -1;
    int dim_b = -1;

    bool operator==(const MoveKey& other) const noexcept {
        return kind == other.kind && group_a == other.group_a &&
                group_b == other.group_b && dim_a == other.dim_a &&
                dim_b == other.dim_b;
    }
};

struct MoveKeyHash {
    std::size_t operator()(const MoveKey& key) const noexcept {
        std::size_t seed = 0;
        hash_combine(seed, static_cast<int>(key.kind));
        hash_combine(seed, key.group_a);
        hash_combine(seed, key.group_b);
        hash_combine(seed, key.dim_a);
        hash_combine(seed, key.dim_b);
        return seed;
    }
};

MoveKey make_relocate_move_key(int A, int B, int v) {
    return MoveKey{
            .kind = Move::Kind::kRelocate,
            .group_a = A,
            .group_b = B,
            .dim_a = v,
            .dim_b = -1,
    };
}

MoveKey make_swap_move_key(int A, int B, int v, int u) {
    if (A < B) {
        return MoveKey{
                .kind = Move::Kind::kSwap,
                .group_a = A,
                .group_b = B,
                .dim_a = v,
                .dim_b = u,
        };
    }
    return MoveKey{
            .kind = Move::Kind::kSwap,
            .group_a = B,
            .group_b = A,
            .dim_a = u,
            .dim_b = v,
    };
}

SearchConfig make_mbeam_search_config(const RefinedStructureBuilder& cfg) {
    return SearchConfig{
            .donor_topk = cfg.mbeam_donor_topk,
            .recv_topk = cfg.mbeam_recv_topk,
            .dims_sample_per_group = cfg.mbeam_dims_sample_per_group,
            .suspicious_alpha = cfg.mbeam_suspicious_alpha,
            .n_relocate = cfg.mbeam_n_relocate,
            .n_swap_pairs = cfg.mbeam_n_swap_pairs,
            .relocate_pair_limit = cfg.mbeam_relocate_pair_limit,
            .swap_pair_limit = cfg.mbeam_swap_pair_limit,
            .shortlist_k = cfg.mbeam_per_state_shortlist_k,
            .shortlist_per_pair = cfg.mbeam_shortlist_per_pair,
            .max_local_score = cfg.mbeam_max_local_score,
            .shift_lambda = cfg.mbeam_shift_lambda,
    };
}

SearchConfig make_greedy_tail_search_config(const RefinedStructureBuilder& cfg) {
    return SearchConfig{
            .donor_topk = cfg.greedy_tail_donor_topk,
            .recv_topk = cfg.greedy_tail_recv_topk,
            .dims_sample_per_group = cfg.greedy_tail_dims_sample_per_group,
            .suspicious_alpha = cfg.greedy_tail_suspicious_alpha,
            .n_relocate = cfg.greedy_tail_n_relocate,
            .n_swap_pairs = cfg.greedy_tail_n_swap_pairs,
            .relocate_pair_limit = cfg.greedy_tail_relocate_pair_limit,
            .swap_pair_limit = cfg.greedy_tail_swap_pair_limit,
            .shortlist_k = cfg.greedy_tail_shortlist_k,
            .shortlist_per_pair = cfg.greedy_tail_shortlist_per_pair,
            .max_local_score = cfg.greedy_tail_max_local_score,
            .shift_lambda = cfg.greedy_tail_shift_lambda,
    };
}

MBeamRunConfig make_mbeam_run_config(const RefinedStructureBuilder& cfg) {
    return MBeamRunConfig{
            .label = "mbeam",
            .iters = cfg.mbeam_iters,
            .patience = cfg.mbeam_patience,
            .eps_improve = cfg.mbeam_eps_improve,
            .beam_width = cfg.mbeam_beam_width,
            .per_state_eval_topk = cfg.mbeam_per_state_eval_topk,
            .seen_window = cfg.mbeam_seen_window,
            .min_novel_children = cfg.mbeam_min_novel_children,
            .search = make_mbeam_search_config(cfg),
    };
}

std::vector<std::pair<int, int>> build_relocate_pairs(
        const Groups& groups,
        const std::vector<int>& donor_pool,
        const std::vector<int>& recv_pool) {
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(donor_pool.size() * recv_pool.size());
    for (int A : donor_pool) {
        if (groups[static_cast<size_t>(A)].size() <= 1) {
            continue;
        }
        for (int B : recv_pool) {
            if (A == B) {
                continue;
            }
            pairs.emplace_back(A, B);
        }
    }
    return pairs;
}

std::vector<std::pair<int, int>> build_swap_pairs(
        const std::vector<int>& donor_pool,
        const std::vector<int>& recv_pool) {
    std::unordered_set<std::pair<int, int>, IntPairHash> uniq;
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(donor_pool.size() * recv_pool.size());
    for (int A : donor_pool) {
        for (int B : recv_pool) {
            if (A == B) {
                continue;
            }
            const std::pair<int, int> key{
                    std::min(A, B),
                    std::max(A, B),
            };
            if (uniq.insert(key).second) {
                pairs.push_back(key);
            }
        }
    }
    return pairs;
}

std::vector<Move> build_shortlisted_moves(
        ProxyContext& proxy,
        const Groups& cur_groups,
        const Bits& cur_bits,
        const SearchConfig& cfg,
        std::mt19937& rng,
        MoveBuildStats* stats_out = nullptr) {
    MoveBuildStats stats;
    const int M = static_cast<int>(cur_groups.size());
    if (M <= 1) {
        if (stats_out != nullptr) {
            *stats_out = stats;
        }
        return {};
    }

    std::vector<double> gain_now(static_cast<size_t>(M), 0.0);
    for (int i = 0; i < M; ++i) {
        gain_now[static_cast<size_t>(i)] = marginal_gain(
                proxy,
                cur_groups[static_cast<size_t>(i)],
                cur_bits[static_cast<size_t>(i)]);
    }
    std::vector<double> fat_score(static_cast<size_t>(M), 0.0);
    for (int i = 0; i < M; ++i) {
        fat_score[static_cast<size_t>(i)] =
                static_cast<double>(cur_bits[static_cast<size_t>(i)]) /
                std::max(1e-12, gain_now[static_cast<size_t>(i)] + 1e-12);
    }
    std::vector<int> donor_pool =
            top_indices(fat_score, std::min(cfg.donor_topk, M), true);
    std::vector<int> recv_pool =
            top_indices(gain_now, std::min(cfg.recv_topk, M), true);
    std::vector<double> D_before(static_cast<size_t>(M), 0.0);
    for (int i = 0; i < M; ++i) {
        D_before[static_cast<size_t>(i)] = proxy.D(
                cur_groups[static_cast<size_t>(i)],
                cur_bits[static_cast<size_t>(i)]);
    }
    std::vector<std::optional<GroupMoveCache>> group_move_caches(
            static_cast<size_t>(M));
    auto get_group_move_cache = [&](int gid) -> const GroupMoveCache& {
        auto& slot = group_move_caches[static_cast<size_t>(gid)];
        if (!slot.has_value()) {
            slot = build_group_move_cache(
                    proxy,
                    cur_groups[static_cast<size_t>(gid)],
                    cur_bits[static_cast<size_t>(gid)],
                    D_before[static_cast<size_t>(gid)],
                    rng,
                    cfg.dims_sample_per_group,
                    cfg.suspicious_alpha);
        }
        return *slot;
    };

    std::vector<Move> moves;
    moves.reserve(static_cast<size_t>(cfg.n_relocate + cfg.n_swap_pairs));
    std::unordered_set<MoveKey, MoveKeyHash> move_keys;

    auto relocate_pairs = build_relocate_pairs(cur_groups, donor_pool, recv_pool);
    for (int round = 0;
         round < std::max(1, cfg.relocate_pair_limit) &&
         static_cast<int>(moves.size()) < cfg.n_relocate;
         ++round) {
        shuffle_vector(relocate_pairs, rng);
        for (const auto& [A, B] : relocate_pairs) {
            if (static_cast<int>(moves.size()) >= cfg.n_relocate) {
                break;
            }
            const auto& donor_cache = get_group_move_cache(A);
            const RemovedDimEval* picked =
                    choose_suspicious_dim(donor_cache, rng, cfg.suspicious_alpha);
            if (picked == nullptr) {
                continue;
            }
            const int v = picked->dim;
            if (!move_keys.insert(make_relocate_move_key(A, B, v)).second) {
                continue;
            }
            const auto& gA2 = picked->removed_group;
            if (gA2.empty()) {
                continue;
            }
            auto gB2 = cur_groups[static_cast<size_t>(B)];
            gB2.push_back(v);
            const double dJ_struct =
                    picked->removed_D + proxy.D(gB2, cur_bits[static_cast<size_t>(B)]) -
                    D_before[static_cast<size_t>(A)] -
                    D_before[static_cast<size_t>(B)];
            const double gainB =
                    marginal_gain(proxy, gB2, cur_bits[static_cast<size_t>(B)]);
            const double gml = std::isfinite(picked->loss_after_remove)
                    ? std::max(0.0, gainB - picked->loss_after_remove)
                    : 0.0;
            moves.push_back(Move{
                    .kind = Move::Kind::kRelocate,
                    .dims = {v},
                    .groups = {A, B},
                    .dJ_struct = dJ_struct,
                    .gml = gml,
                    .score = dJ_struct - cfg.shift_lambda * gml,
            });
        }
    }

    const size_t relocate_count = moves.size();
    auto swap_pairs = build_swap_pairs(donor_pool, recv_pool);
    for (int round = 0;
         round < std::max(1, cfg.swap_pair_limit) &&
         static_cast<int>(moves.size() - relocate_count) < cfg.n_swap_pairs;
         ++round) {
        shuffle_vector(swap_pairs, rng);
        for (const auto& [A, B] : swap_pairs) {
            if (static_cast<int>(moves.size() - relocate_count) >= cfg.n_swap_pairs) {
                break;
            }
            const auto& cache_A = get_group_move_cache(A);
            const auto& cache_B = get_group_move_cache(B);
            const RemovedDimEval* picked_A =
                    choose_suspicious_dim(cache_A, rng, cfg.suspicious_alpha);
            const RemovedDimEval* picked_B =
                    choose_suspicious_dim(cache_B, rng, cfg.suspicious_alpha);
            if (picked_A == nullptr || picked_B == nullptr) {
                continue;
            }
            const int v = picked_A->dim;
            const int u = picked_B->dim;
            if (u == v) {
                continue;
            }
            if (!move_keys.insert(make_swap_move_key(A, B, v, u)).second) {
                continue;
            }
            auto gA2 = picked_A->removed_group;
            gA2.push_back(u);
            auto gB2 = picked_B->removed_group;
            gB2.push_back(v);
            const double dJ_struct =
                    proxy.D(gA2, cur_bits[static_cast<size_t>(A)]) +
                    proxy.D(gB2, cur_bits[static_cast<size_t>(B)]) -
                    D_before[static_cast<size_t>(A)] -
                    D_before[static_cast<size_t>(B)];
            moves.push_back(Move{
                    .kind = Move::Kind::kSwap,
                    .dims = {v, u},
                    .groups = {A, B},
                    .dJ_struct = dJ_struct,
                    .gml = 0.0,
                    .score = dJ_struct,
            });
        }
    }

    std::sort(moves.begin(), moves.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.score < rhs.score;
    });
    stats.raw = moves.size();
    if (cfg.shortlist_per_pair > 0) {
        std::vector<Move> diverse_moves;
        diverse_moves.reserve(
                std::min(static_cast<size_t>(cfg.shortlist_k), moves.size()));
        std::unordered_map<std::pair<int, int>, int, IntPairHash> pair_counts;
        for (auto& move : moves) {
            if (move.score > cfg.max_local_score) {
                ++stats.score_pruned;
                continue;
            }
            auto& used = pair_counts[move.pair_key()];
            if (used >= cfg.shortlist_per_pair) {
                ++stats.pair_pruned;
                continue;
            }
            ++used;
            diverse_moves.push_back(std::move(move));
            if (static_cast<int>(diverse_moves.size()) >= cfg.shortlist_k) {
                break;
            }
        }
        moves = std::move(diverse_moves);
    } else if (static_cast<int>(moves.size()) > cfg.shortlist_k) {
        std::vector<Move> filtered_moves;
        filtered_moves.reserve(
                std::min(static_cast<size_t>(cfg.shortlist_k), moves.size()));
        for (auto& move : moves) {
            if (move.score > cfg.max_local_score) {
                ++stats.score_pruned;
                continue;
            }
            filtered_moves.push_back(std::move(move));
            if (static_cast<int>(filtered_moves.size()) >= cfg.shortlist_k) {
                break;
            }
        }
        moves = std::move(filtered_moves);
    }
    stats.shortlisted = moves.size();
    if (stats_out != nullptr) {
        *stats_out = stats;
    }
    return moves;
}

std::pair<Groups, Bits> run_mbeam_stage_impl(
        const MBeamRunConfig& run_cfg,
        ProxyContext& proxy,
        const BuildContext& ctx,
        const Groups& groups,
        const Bits& bits) {
    (void)bits;
    std::mt19937 rng(proxy.seed);
    const SearchConfig& search_cfg = run_cfg.search;
    auto alloc0 = proxy.solve_bits(groups);
    std::vector<BeamState> beam{BeamState{groups, alloc0.bits, alloc0.J}};
    double global_best = alloc0.J;
    Groups global_best_groups = groups;
    Bits global_best_bits = alloc0.bits;
    SeenWindow seen(run_cfg.seen_window);
    seen.set_best(canonical_partition_key(groups), alloc0.J);
    int no_improve = 0;

    EPQ_STRUCTURE_DEBUG_LOG(
            2,
            run_cfg.label << " begin groups=" << groups.size()
                          << " objective=" << alloc0.J
                          << " beam_width=" << run_cfg.beam_width);

    for (int it = 0; it < run_cfg.iters; ++it) {
        if (run_cfg.patience > 0 && no_improve >= run_cfg.patience) {
            EPQ_STRUCTURE_DEBUG_LOG(
                    2,
                    run_cfg.label << " iter=" << it
                                  << " patience hit, stop");
            break;
        }
        const size_t beam_in = beam.size();
        const double iter_best_before = beam.front().J;
        int states_expanded = 0;
        size_t total_moves_raw = 0;
        size_t total_moves_shortlisted = 0;
        size_t total_seen_pruned = 0;
        size_t total_dup_pruned = 0;
        size_t total_pair_pruned = 0;
        size_t total_score_pruned = 0;
        size_t total_children = 0;
        seen.next_round();
        std::vector<BeamState> children;
        for (const auto& state : beam) {
            const Groups& cur_groups = state.groups;
            if (cur_groups.size() <= 1) {
                continue;
            }
            ++states_expanded;
            MoveBuildStats move_stats;
            auto moves = build_shortlisted_moves(
                    proxy,
                    cur_groups,
                    state.bits,
                    search_cfg,
                    rng,
                    &move_stats);
            total_moves_raw += move_stats.raw;
            total_moves_shortlisted += move_stats.shortlisted;
            total_pair_pruned += move_stats.pair_pruned;
            total_score_pruned += move_stats.score_pruned;

            int attempted = 0;
            std::unordered_set<PartitionKey, PartitionKeyHash> local_partitions;
            for (const auto& mv : moves) {
                if (attempted >= run_cfg.per_state_eval_topk) {
                    break;
                }
                ++attempted;
                Groups cand_groups;
                try {
                    if (mv.kind == Move::Kind::kRelocate) {
                        cand_groups = apply_relocate(
                                cur_groups,
                                mv.groups.first,
                                mv.groups.second,
                                mv.dims.front());
                    } else {
                        cand_groups = apply_swap(
                                cur_groups,
                                mv.groups.first,
                                mv.groups.second,
                                mv.dims[0],
                                mv.dims[1]);
                    }
                    validate_partition(cand_groups, ctx.d, true);
                } catch (const std::exception&) {
                    continue;
                }
                const PartitionKey key = canonical_partition_key(cand_groups);
                if (!local_partitions.insert(key).second) {
                    ++total_dup_pruned;
                    continue;
                }
                const auto prev = seen.get_best(key);
                if (prev.has_value()) {
                    ++total_seen_pruned;
                    continue;
                }
                auto alloc = proxy.solve_bits(cand_groups);
                seen.set_best(key, alloc.J);
                children.push_back(BeamState{
                        .groups = std::move(cand_groups),
                        .bits = std::move(alloc.bits),
                        .J = alloc.J,
                });
                ++total_children;
            }
        }

        if (run_cfg.min_novel_children > 0 &&
            total_children < static_cast<size_t>(run_cfg.min_novel_children)) {
            EPQ_STRUCTURE_DEBUG_LOG(
                    2,
                    run_cfg.label << " iter=" << it
                                  << " beam_in=" << beam_in
                                  << " states=" << states_expanded
                                  << " raw_moves=" << total_moves_raw
                                  << " shortlisted=" << total_moves_shortlisted
                                  << " pair_pruned=" << total_pair_pruned
                                  << " score_pruned=" << total_score_pruned
                                  << " dup_pruned=" << total_dup_pruned
                                  << " seen_pruned=" << total_seen_pruned
                                  << " children=" << total_children
                                  << " novel floor hit, stop");
            break;
        }

        std::vector<BeamState> candidates = beam;
        candidates.insert(
                candidates.end(),
                std::make_move_iterator(children.begin()),
                std::make_move_iterator(children.end()));
        std::unordered_map<PartitionKey, BeamState, PartitionKeyHash> uniq;
        for (auto& state : candidates) {
            const PartitionKey key = canonical_partition_key(state.groups);
            auto it = uniq.find(key);
            if (it == uniq.end() || state.J < it->second.J) {
                uniq[key] = std::move(state);
            }
        }
        beam.clear();
        for (auto& [_, state] : uniq) {
            beam.push_back(std::move(state));
        }
        std::sort(beam.begin(), beam.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.J < rhs.J;
        });
        if (beam.empty()) {
            EPQ_STRUCTURE_DEBUG_LOG(
                    2,
                    run_cfg.label << " iter=" << it
                                  << " empty beam, stop");
            break;
        }
        if (static_cast<int>(beam.size()) > run_cfg.beam_width) {
            beam.resize(static_cast<size_t>(run_cfg.beam_width));
        }
        if (beam.front().J < global_best - run_cfg.eps_improve) {
            const double delta_iter = iter_best_before - beam.front().J;
            const double delta_global = global_best - beam.front().J;
            global_best = beam.front().J;
            global_best_groups = beam.front().groups;
            global_best_bits = beam.front().bits;
            no_improve = 0;
            EPQ_STRUCTURE_DEBUG_LOG(
                    2,
                    run_cfg.label << " iter=" << it
                                  << " beam_in=" << beam_in
                                  << " states=" << states_expanded
                                  << " raw_moves=" << total_moves_raw
                                  << " shortlisted=" << total_moves_shortlisted
                                  << " pair_pruned=" << total_pair_pruned
                                  << " score_pruned=" << total_score_pruned
                                  << " dup_pruned=" << total_dup_pruned
                                  << " seen_pruned=" << total_seen_pruned
                                  << " children=" << total_children
                                  << " uniq=" << uniq.size()
                                  << " improved objective=" << global_best
                                  << " delta_iter=" << delta_iter
                                  << " delta_global=" << delta_global
                                  << " groups=" << global_best_groups.size()
                                  << " beam=" << beam.size()
                                  << " candidates=" << candidates.size());
        } else {
            ++no_improve;
            const double delta_iter = iter_best_before - beam.front().J;
            EPQ_STRUCTURE_DEBUG_LOG(
                    2,
                    run_cfg.label << " iter=" << it
                                  << " beam_in=" << beam_in
                                  << " states=" << states_expanded
                                  << " raw_moves=" << total_moves_raw
                                  << " shortlisted=" << total_moves_shortlisted
                                  << " pair_pruned=" << total_pair_pruned
                                  << " score_pruned=" << total_score_pruned
                                  << " dup_pruned=" << total_dup_pruned
                                  << " seen_pruned=" << total_seen_pruned
                                  << " children=" << total_children
                                  << " uniq=" << uniq.size()
                                  << " no_improve=" << no_improve
                                  << " beam_best=" << beam.front().J
                                  << " delta_iter=" << delta_iter
                                  << " global_best=" << global_best
                                  << " beam=" << beam.size()
                                  << " candidates=" << candidates.size());
        }
    }

    auto allocF = proxy.solve_bits(global_best_groups);
    EPQ_STRUCTURE_DEBUG_LOG(
            2,
            run_cfg.label << " end groups=" << global_best_groups.size()
                          << " objective=" << allocF.J);
    return {std::move(global_best_groups), std::move(allocF.bits)};
}

std::pair<Groups, Bits> run_mbeam_stage(
        const RefinedStructureBuilder& cfg,
        ProxyContext& proxy,
        const BuildContext& ctx,
        const Groups& groups,
        const Bits& bits) {
    return run_mbeam_stage_impl(
            make_mbeam_run_config(cfg), proxy, ctx, groups, bits);
}

}  // namespace epq::structure_builder_internal
