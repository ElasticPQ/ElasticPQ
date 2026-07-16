#include "structure_builder_local_search.h"

namespace epq::structure_builder_internal {

namespace {

double proxy_score(
        ProxyContext& proxy,
        const std::vector<int>& dims,
        int bits,
        bool use_fast_proxy) {
    return use_fast_proxy ? proxy.D_fast(dims, bits) : proxy.D(dims, bits);
}

double marginal_gain(
        ProxyContext& proxy,
        const std::vector<int>& dims,
        int bits,
        bool use_fast_proxy) {
    if (bits >= proxy.build_ctx.max_bits) {
        return 0.0;
    }
    return proxy_score(proxy, dims, bits, use_fast_proxy) -
            proxy_score(proxy, dims, bits + 1, use_fast_proxy);
}

double marginal_loss_with_value(
        ProxyContext& proxy,
        const std::vector<int>& dims,
        int bits,
        double value_at_bits,
        bool use_fast_proxy) {
    if (bits <= proxy.build_ctx.min_bits) {
        return std::numeric_limits<double>::infinity();
    }
    return proxy_score(proxy, dims, bits - 1, use_fast_proxy) - value_at_bits;
}

struct ChainTailConfig {
    int iters = 0;
    int patience = 0;
    double eps_improve = 0.0;
    int eval_topk = 0;
    int shortlist_k = 0;
    int donor_topk = 0;
    int recv_topk = 0;
    int dims_sample_per_group = 0;
    float suspicious_alpha = 0.0f;
    int n_seed_moves = 0;
    int receiver_topk_per_dim = 0;
    int max_depth = 0;
    double max_local_score = 0.0;
    double prefix_slack = 0.0;
    int seen_window = 1;
    bool use_fast_proxy = false;
    int fast_shortlist_mult = 1;
};

struct RemovedDimEval {
    int dim = -1;
    std::vector<int> removed_group;
    double removed_D = 0.0;
    double loss_after_remove = std::numeric_limits<double>::infinity();
    double harm = 0.0;
};

struct ReceiverEval {
    int gid = -1;
    double score = std::numeric_limits<double>::infinity();
};

struct ReceiverAppendEval {
    double recv_after_D = 0.0;
    double recv_gain_after_add = 0.0;
};

struct ChainStep {
    int from = -1;
    int to = -1;
    int dim = -1;
    double score = 0.0;
};

struct ChainCandidate {
    std::vector<ChainStep> steps;
    double score = std::numeric_limits<double>::infinity();
};

struct ChainBuildStats {
    size_t seeds_raw = 0;
    size_t seeds_kept = 0;
    size_t candidates_kept = 0;
    size_t local_gate_pruned = 0;
    size_t donor_small_stops = 0;
    size_t no_step_stops = 0;
    size_t prefix_cut_stops = 0;
    size_t total_steps = 0;
    size_t max_steps = 0;
    double score_sum = 0.0;
    double score_min = std::numeric_limits<double>::infinity();
    double score_max = -std::numeric_limits<double>::infinity();

    void observe_candidate(const ChainCandidate& candidate) {
        const size_t steps = candidate.steps.size();
        ++candidates_kept;
        total_steps += steps;
        max_steps = std::max(max_steps, steps);
        score_sum += candidate.score;
        score_min = std::min(score_min, candidate.score);
        score_max = std::max(score_max, candidate.score);
    }
};

struct ChainRunStats {
    size_t iterations = 0;
    size_t iters_with_candidates = 0;
    size_t seeds_raw_total = 0;
    size_t seeds_kept_total = 0;
    size_t candidates_total = 0;
    size_t local_gate_pruned_total = 0;
    size_t donor_small_stops_total = 0;
    size_t no_step_stops_total = 0;
    size_t prefix_cut_stops_total = 0;
    size_t total_steps = 0;
    size_t max_steps = 0;
    double score_sum = 0.0;
    double score_min = std::numeric_limits<double>::infinity();
    double score_max = -std::numeric_limits<double>::infinity();
    size_t exact_attempted = 0;
    size_t exact_children = 0;
    size_t exact_dup_pruned = 0;
    size_t exact_seen_pruned = 0;
    size_t exact_local_reranked_total = 0;
    size_t exact_local_kept_total = 0;
    size_t improved_iters = 0;
    double exact_delta_sum = 0.0;
    double exact_delta_min = std::numeric_limits<double>::infinity();
    double exact_delta_max = 0.0;

    void observe_build(const ChainBuildStats& stats) {
        seeds_raw_total += stats.seeds_raw;
        seeds_kept_total += stats.seeds_kept;
        candidates_total += stats.candidates_kept;
        local_gate_pruned_total += stats.local_gate_pruned;
        donor_small_stops_total += stats.donor_small_stops;
        no_step_stops_total += stats.no_step_stops;
        prefix_cut_stops_total += stats.prefix_cut_stops;
        total_steps += stats.total_steps;
        max_steps = std::max(max_steps, stats.max_steps);
        score_sum += stats.score_sum;
        score_min = std::min(score_min, stats.score_min);
        score_max = std::max(score_max, stats.score_max);
        if (stats.candidates_kept > 0) {
            ++iters_with_candidates;
        }
    }

    void observe_exact_improvement(double delta) {
        ++improved_iters;
        exact_delta_sum += delta;
        exact_delta_min = std::min(exact_delta_min, delta);
        exact_delta_max = std::max(exact_delta_max, delta);
    }
};

ChainTailConfig make_chain_tail_config(const RefinedStructureBuilder& cfg) {
    return ChainTailConfig{
            .iters = cfg.chain_tail_iters,
            .patience = cfg.chain_tail_patience,
            .eps_improve = cfg.chain_tail_eps_improve,
            .eval_topk = cfg.chain_tail_eval_topk,
            .shortlist_k = cfg.chain_tail_shortlist_k,
            .donor_topk = cfg.chain_tail_donor_topk,
            .recv_topk = cfg.chain_tail_recv_topk,
            .dims_sample_per_group = cfg.chain_tail_dims_sample_per_group,
            .suspicious_alpha = cfg.chain_tail_suspicious_alpha,
            .n_seed_moves = cfg.chain_tail_n_seed_moves,
            .receiver_topk_per_dim = cfg.chain_tail_receiver_topk_per_dim,
            .max_depth = cfg.chain_tail_max_depth,
            .max_local_score = cfg.chain_tail_max_local_score,
            .prefix_slack = cfg.chain_tail_prefix_slack,
            .seen_window = cfg.chain_tail_seen_window,
            .use_fast_proxy = cfg.chain_tail_fast_proxy_top_dims > 0,
            .fast_shortlist_mult = std::max(1, cfg.chain_tail_fast_shortlist_mult),
    };
}

std::vector<RemovedDimEval> build_removed_dim_evals(
        ProxyContext& proxy,
        const std::vector<int>& group,
        int bits,
        double group_D0,
        std::mt19937& rng,
        int sample_k,
        float suspicious_alpha,
        bool use_fast_proxy) {
    std::vector<int> candidate_dims = group;
    if (sample_k > 0 && static_cast<int>(candidate_dims.size()) > sample_k) {
        candidate_dims = sample_vector(candidate_dims, sample_k, rng);
    }
    std::vector<RemovedDimEval> out;
    out.reserve(candidate_dims.size());
    for (int dim : candidate_dims) {
        auto removed_group = remove_one(group, dim);
        const double removed_D = proxy_score(proxy, removed_group, bits, use_fast_proxy);
        const double loss_after_remove =
                marginal_loss_with_value(
                        proxy, removed_group, bits, removed_D, use_fast_proxy);
        out.push_back(RemovedDimEval{
                .dim = dim,
                .removed_group = std::move(removed_group),
                .removed_D = removed_D,
                .loss_after_remove = loss_after_remove,
                .harm = group_D0 - removed_D,
        });
    }
    if (suspicious_alpha > 0.0f) {
        std::sort(out.begin(), out.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.harm < rhs.harm;
        });
    } else {
        shuffle_vector(out, rng);
    }
    return out;
}

ReceiverAppendEval eval_receiver_after_add(
        ProxyContext& proxy,
        const std::vector<int>& recv_before,
        int recv_bits,
        int dim,
        bool use_fast_proxy) {
    auto recv_after = recv_before;
    recv_after.push_back(dim);
    const double recv_after_D =
            proxy_score(proxy, recv_after, recv_bits, use_fast_proxy);
    const double recv_gain_after_add = recv_bits >= proxy.build_ctx.max_bits
            ? 0.0
            : (recv_after_D -
               proxy_score(proxy, recv_after, recv_bits + 1, use_fast_proxy));
    return ReceiverAppendEval{
            .recv_after_D = recv_after_D,
            .recv_gain_after_add = recv_gain_after_add,
    };
}

double relocate_local_score(
        const std::vector<int>& donor_after,
        double donor_after_D,
        double donor_loss_after_remove,
        double recv_after_D,
        double recv_gain_after_add,
        double donor_before_D,
        double recv_before_D) {
    const double dJ_struct =
            donor_after_D + recv_after_D - donor_before_D - recv_before_D;
    const double gml = std::isfinite(donor_loss_after_remove)
            ? std::max(0.0, recv_gain_after_add - donor_loss_after_remove)
            : 0.0;
    return dJ_struct - gml;
}

Groups apply_chain_candidate(
        const Groups& groups,
        const ChainCandidate& candidate) {
    Groups out = groups;
    for (const auto& step : candidate.steps) {
        out = apply_relocate(out, step.from, step.to, step.dim);
    }
    return out;
}

struct PartialChainState {
    Groups working_groups;
    std::vector<ChainStep> steps;
    double cumulative_score = 0.0;
    double best_prefix_score = std::numeric_limits<double>::infinity();
    size_t best_prefix_len = 0;
    std::unordered_set<int> used_groups;
    std::unordered_set<int> moved_dims;
    int current_gid = -1;
    int incoming_dim = -1;
};

std::vector<ChainStep> rank_next_steps(
        ProxyContext& proxy,
        const Groups& working_groups,
        const Bits& cur_bits,
        const ChainTailConfig& cfg,
        std::mt19937& rng,
        int current_gid,
        int incoming_dim,
        const std::unordered_set<int>& used_groups,
        const std::unordered_set<int>& moved_dims,
        ChainBuildStats* stats_out = nullptr) {
    const int M = static_cast<int>(working_groups.size());
    if (current_gid < 0 || current_gid >= M) {
        return {};
    }
    const auto& donor_group = working_groups[static_cast<size_t>(current_gid)];
    if (donor_group.size() <= 1) {
        if (stats_out != nullptr) {
            ++stats_out->donor_small_stops;
        }
        return {};
    }

    std::vector<double> gain_step(static_cast<size_t>(M), 0.0);
    for (int gid = 0; gid < M; ++gid) {
        gain_step[static_cast<size_t>(gid)] = marginal_gain(
                proxy,
                working_groups[static_cast<size_t>(gid)],
                cur_bits[static_cast<size_t>(gid)],
                cfg.use_fast_proxy);
    }
    const std::vector<int> step_recv_pool =
            top_indices(gain_step, std::min(cfg.recv_topk, M), true);
    const double donor_before_D = proxy_score(
            proxy,
            working_groups[static_cast<size_t>(current_gid)],
            cur_bits[static_cast<size_t>(current_gid)],
            cfg.use_fast_proxy);
    const auto eject_candidates = build_removed_dim_evals(
            proxy,
            donor_group,
            cur_bits[static_cast<size_t>(current_gid)],
            donor_before_D,
            rng,
            cfg.dims_sample_per_group,
            cfg.suspicious_alpha,
            cfg.use_fast_proxy);

    std::vector<ChainStep> options;
    for (const auto& eject : eject_candidates) {
        if (eject.dim == incoming_dim || moved_dims.count(eject.dim) > 0) {
            continue;
        }
        if (eject.removed_group.empty()) {
            continue;
        }
        for (int recv_gid : step_recv_pool) {
            if (recv_gid == current_gid || used_groups.count(recv_gid) > 0) {
                continue;
            }
            const double recv_before_D = proxy_score(
                    proxy,
                    working_groups[static_cast<size_t>(recv_gid)],
                    cur_bits[static_cast<size_t>(recv_gid)],
                    cfg.use_fast_proxy);
            const auto recv_append = eval_receiver_after_add(
                    proxy,
                    working_groups[static_cast<size_t>(recv_gid)],
                    cur_bits[static_cast<size_t>(recv_gid)],
                    eject.dim,
                    cfg.use_fast_proxy);
            const double step_score = relocate_local_score(
                    eject.removed_group,
                    eject.removed_D,
                    eject.loss_after_remove,
                    recv_append.recv_after_D,
                    recv_append.recv_gain_after_add,
                    donor_before_D,
                    recv_before_D);
            options.push_back(ChainStep{
                    .from = current_gid,
                    .to = recv_gid,
                    .dim = eject.dim,
                    .score = step_score,
            });
        }
    }
    std::sort(options.begin(), options.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.score < rhs.score;
    });
    return options;
}

std::vector<ChainCandidate> build_chain_candidates(
        ProxyContext& proxy,
        const Groups& cur_groups,
        const Bits& cur_bits,
        const ChainTailConfig& cfg,
        std::mt19937& rng,
        ChainBuildStats* stats_out = nullptr) {
    const int M = static_cast<int>(cur_groups.size());
    if (M <= 1) {
        return {};
    }

    ChainBuildStats stats;
    std::vector<double> gain_now(static_cast<size_t>(M), 0.0);
    std::vector<double> fat_score(static_cast<size_t>(M), 0.0);
    std::vector<double> D_before(static_cast<size_t>(M), 0.0);
    for (int gid = 0; gid < M; ++gid) {
        const auto& group = cur_groups[static_cast<size_t>(gid)];
        const int bits = cur_bits[static_cast<size_t>(gid)];
        gain_now[static_cast<size_t>(gid)] =
                marginal_gain(proxy, group, bits, cfg.use_fast_proxy);
        fat_score[static_cast<size_t>(gid)] =
                static_cast<double>(bits) /
                std::max(1e-12, gain_now[static_cast<size_t>(gid)] + 1e-12);
        D_before[static_cast<size_t>(gid)] =
                proxy_score(proxy, group, bits, cfg.use_fast_proxy);
    }

    const std::vector<int> donor_pool =
            top_indices(fat_score, std::min(cfg.donor_topk, M), true);
    const std::vector<int> recv_pool =
            top_indices(gain_now, std::min(cfg.recv_topk, M), true);

    std::vector<ChainCandidate> seeds;
    for (int donor_gid : donor_pool) {
        auto removals = build_removed_dim_evals(
                proxy,
                cur_groups[static_cast<size_t>(donor_gid)],
                cur_bits[static_cast<size_t>(donor_gid)],
                D_before[static_cast<size_t>(donor_gid)],
                rng,
                cfg.dims_sample_per_group,
                cfg.suspicious_alpha,
                cfg.use_fast_proxy);
        for (const auto& removal : removals) {
            if (removal.removed_group.empty()) {
                continue;
            }
            std::vector<ReceiverEval> receivers;
            receivers.reserve(recv_pool.size());
            for (int recv_gid : recv_pool) {
                if (recv_gid == donor_gid) {
                    continue;
                }
                const auto recv_append = eval_receiver_after_add(
                        proxy,
                        cur_groups[static_cast<size_t>(recv_gid)],
                        cur_bits[static_cast<size_t>(recv_gid)],
                        removal.dim,
                        cfg.use_fast_proxy);
                receivers.push_back(ReceiverEval{
                        .gid = recv_gid,
                        .score = relocate_local_score(
                                removal.removed_group,
                                removal.removed_D,
                                removal.loss_after_remove,
                                recv_append.recv_after_D,
                                recv_append.recv_gain_after_add,
                                D_before[static_cast<size_t>(donor_gid)],
                                D_before[static_cast<size_t>(recv_gid)]),
                });
            }
            std::sort(receivers.begin(), receivers.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.score < rhs.score;
            });
            if (static_cast<int>(receivers.size()) > cfg.receiver_topk_per_dim) {
                receivers.resize(static_cast<size_t>(cfg.receiver_topk_per_dim));
            }
            for (const auto& receiver : receivers) {
                seeds.push_back(ChainCandidate{
                        .steps = {ChainStep{
                                .from = donor_gid,
                                .to = receiver.gid,
                                .dim = removal.dim,
                                .score = receiver.score,
                        }},
                        .score = receiver.score,
                });
            }
        }
    }

    std::sort(seeds.begin(), seeds.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.score < rhs.score;
    });
    stats.seeds_raw = seeds.size();
    if (static_cast<int>(seeds.size()) > cfg.n_seed_moves) {
        seeds.resize(static_cast<size_t>(cfg.n_seed_moves));
    }
    stats.seeds_kept = seeds.size();

    std::vector<ChainCandidate> out;
    out.reserve(seeds.size());
    for (const auto& seed : seeds) {
        PartialChainState partial{
                .working_groups = apply_chain_candidate(cur_groups, seed),
                .steps = seed.steps,
                .cumulative_score = seed.score,
                .best_prefix_score = seed.score,
                .best_prefix_len = seed.steps.size(),
                .used_groups = {seed.steps.front().from, seed.steps.front().to},
                .moved_dims = {seed.steps.front().dim},
                .current_gid = seed.steps.back().to,
                .incoming_dim = seed.steps.back().dim,
        };
        while (static_cast<int>(partial.steps.size()) < std::max(1, cfg.max_depth)) {
            auto options = rank_next_steps(
                    proxy,
                    partial.working_groups,
                    cur_bits,
                    cfg,
                    rng,
                    partial.current_gid,
                    partial.incoming_dim,
                    partial.used_groups,
                    partial.moved_dims,
                    &stats);
            if (options.empty()) {
                ++stats.no_step_stops;
                break;
            }
            const auto& best_step = options.front();
            const double next_cumulative_score =
                    partial.cumulative_score + best_step.score;
            if (next_cumulative_score > partial.best_prefix_score + cfg.prefix_slack) {
                ++stats.prefix_cut_stops;
                break;
            }

            partial.working_groups = apply_relocate(
                    partial.working_groups,
                    best_step.from,
                    best_step.to,
                    best_step.dim);
            partial.steps.push_back(best_step);
            partial.cumulative_score = next_cumulative_score;
            partial.current_gid = best_step.to;
            partial.incoming_dim = best_step.dim;
            partial.used_groups.insert(best_step.to);
            partial.moved_dims.insert(best_step.dim);
            if (partial.cumulative_score < partial.best_prefix_score) {
                partial.best_prefix_score = partial.cumulative_score;
                partial.best_prefix_len = partial.steps.size();
            }
        }

        if (partial.best_prefix_len == 0 || partial.best_prefix_score > cfg.max_local_score) {
            ++stats.local_gate_pruned;
            continue;
        }
        ChainCandidate candidate;
        candidate.steps.assign(
                partial.steps.begin(),
                partial.steps.begin() + partial.best_prefix_len);
        candidate.score = partial.best_prefix_score;
        stats.observe_candidate(candidate);
        out.push_back(std::move(candidate));
    }

    std::sort(out.begin(), out.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.score < rhs.score;
    });
    const int cheap_shortlist_k = cfg.use_fast_proxy
            ? std::max(cfg.shortlist_k, cfg.shortlist_k * cfg.fast_shortlist_mult)
            : cfg.shortlist_k;
    if (static_cast<int>(out.size()) > cheap_shortlist_k) {
        out.resize(static_cast<size_t>(cheap_shortlist_k));
    }
    if (stats_out != nullptr) {
        stats.candidates_kept = 0;
        stats.total_steps = 0;
        stats.max_steps = 0;
        stats.score_sum = 0.0;
        stats.score_min = std::numeric_limits<double>::infinity();
        stats.score_max = -std::numeric_limits<double>::infinity();
        for (const auto& candidate : out) {
            stats.observe_candidate(candidate);
        }
        *stats_out = stats;
    }
    return out;
}

double score_chain_candidate_exact(
        ProxyContext& proxy,
        const Groups& cur_groups,
        const Bits& cur_bits,
        const ChainCandidate& candidate) {
    Groups working_groups = cur_groups;
    double total_score = 0.0;
    for (const auto& step : candidate.steps) {
        if (step.from < 0 || step.to < 0 ||
            step.from >= static_cast<int>(working_groups.size()) ||
            step.to >= static_cast<int>(working_groups.size())) {
            return std::numeric_limits<double>::infinity();
        }
        const auto& donor_group = working_groups[static_cast<size_t>(step.from)];
        const auto& recv_group = working_groups[static_cast<size_t>(step.to)];
        const int donor_bits = cur_bits[static_cast<size_t>(step.from)];
        const int recv_bits = cur_bits[static_cast<size_t>(step.to)];
        const double donor_before_D = proxy.D(donor_group, donor_bits);
        const double recv_before_D = proxy.D(recv_group, recv_bits);
        auto donor_after = remove_one(donor_group, step.dim);
        if (donor_after.empty()) {
            return std::numeric_limits<double>::infinity();
        }
        const double donor_after_D = proxy.D(donor_after, donor_bits);
        const double donor_loss_after_remove =
                marginal_loss_with_value(
                        proxy, donor_after, donor_bits, donor_after_D, false);
        const auto recv_append = eval_receiver_after_add(
                proxy, recv_group, recv_bits, step.dim, false);
        total_score += relocate_local_score(
                donor_after,
                donor_after_D,
                donor_loss_after_remove,
                recv_append.recv_after_D,
                recv_append.recv_gain_after_add,
                donor_before_D,
                recv_before_D);
        working_groups = apply_relocate(working_groups, step.from, step.to, step.dim);
    }
    return total_score;
}

std::vector<ChainCandidate> exact_local_rerank_candidates(
        ProxyContext& proxy,
        const Groups& cur_groups,
        const Bits& cur_bits,
        const ChainTailConfig& cfg,
        std::vector<ChainCandidate> candidates,
        ChainRunStats* run_stats) {
    if (!cfg.use_fast_proxy || candidates.empty()) {
        return candidates;
    }
    if (run_stats != nullptr) {
        run_stats->exact_local_reranked_total += candidates.size();
    }
    for (auto& candidate : candidates) {
        candidate.score = score_chain_candidate_exact(
                proxy, cur_groups, cur_bits, candidate);
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.score < rhs.score;
    });
    if (static_cast<int>(candidates.size()) > cfg.shortlist_k) {
        candidates.resize(static_cast<size_t>(cfg.shortlist_k));
    }
    if (run_stats != nullptr) {
        run_stats->exact_local_kept_total += candidates.size();
    }
    return candidates;
}

}  // namespace

std::pair<Groups, Bits> run_chain_tail_stage(
        const RefinedStructureBuilder& cfg,
        ProxyContext& proxy,
        const BuildContext& ctx,
        const Groups& groups,
        const Bits& bits) {
    (void)bits;
    std::mt19937 rng(cfg.seed);
    const ChainTailConfig chain_cfg = make_chain_tail_config(cfg);
    ChainRunStats run_stats;
    auto alloc0 = proxy.solve_bits(groups);
    BeamState current{groups, alloc0.bits, alloc0.J};
    BeamState best = current;
    SeenWindow seen(chain_cfg.seen_window);
    seen.set_best(canonical_partition_key(groups), alloc0.J);
    int no_improve = 0;
#if EPQ_ENABLE_STRUCTURE_TRACE
    trace_structure_candidate(
            ctx,
            "chain_tail_root",
            0,
            current.groups,
            current.bits,
            current.J,
            "chain_tail");
#endif

    EPQ_STRUCTURE_DEBUG_LOG(
            2,
            "chain_tail begin groups=" << groups.size()
                                       << " objective=" << alloc0.J
                                       << " eval_topk=" << chain_cfg.eval_topk
                                       << " max_depth=" << chain_cfg.max_depth);

    for (int it = 0; it < chain_cfg.iters; ++it) {
        ++run_stats.iterations;
        if (chain_cfg.patience > 0 && no_improve >= chain_cfg.patience) {
            EPQ_STRUCTURE_DEBUG_LOG(
                    2,
                    "chain_tail iter=" << it
                                       << " patience hit, stop");
            break;
        }

        seen.next_round();
        ChainBuildStats build_stats;
        auto candidates = build_chain_candidates(
                proxy, current.groups, current.bits, chain_cfg, rng, &build_stats);
        run_stats.observe_build(build_stats);
        candidates = exact_local_rerank_candidates(
                proxy, current.groups, current.bits, chain_cfg, std::move(candidates), &run_stats);
        size_t dup_pruned = 0;
        size_t seen_pruned = 0;
        size_t children = 0;
        std::optional<BeamState> iter_best;

        if (candidates.empty()) {
            EPQ_STRUCTURE_DEBUG_LOG(
                    2,
                    "chain_tail iter=" << it
                                       << " no candidates, stop");
            break;
        }

        int attempted = 0;
        std::unordered_set<PartitionKey, PartitionKeyHash> local_partitions;
        for (const auto& candidate : candidates) {
            if (attempted >= chain_cfg.eval_topk) {
                break;
            }
            ++attempted;
            ++run_stats.exact_attempted;
            Groups cand_groups;
            try {
                cand_groups = apply_chain_candidate(current.groups, candidate);
                validate_partition(cand_groups, ctx.d, true);
            } catch (const std::exception&) {
                continue;
            }
            const PartitionKey key = canonical_partition_key(cand_groups);
            if (!local_partitions.insert(key).second) {
                ++dup_pruned;
                ++run_stats.exact_dup_pruned;
                continue;
            }
            if (seen.get_best(key).has_value()) {
                ++seen_pruned;
                ++run_stats.exact_seen_pruned;
                continue;
            }
            auto alloc = proxy.solve_bits(cand_groups);
            seen.set_best(key, alloc.J);
            ++children;
            ++run_stats.exact_children;
            if (!iter_best.has_value() || alloc.J < iter_best->J) {
                iter_best = BeamState{
                        .groups = std::move(cand_groups),
                        .bits = std::move(alloc.bits),
                        .J = alloc.J,
                };
            }
        }

        if (iter_best.has_value() && iter_best->J < current.J - chain_cfg.eps_improve) {
            const double delta_iter = current.J - iter_best->J;
            run_stats.observe_exact_improvement(delta_iter);
            current = std::move(*iter_best);
            if (current.J < best.J - chain_cfg.eps_improve) {
                best = current;
            }
#if EPQ_ENABLE_STRUCTURE_TRACE
            trace_structure_candidate(
                    ctx,
                    "chain_tail_improve",
                    it,
                    current.groups,
                    current.bits,
                    current.J,
                    "chain_tail");
#endif
            no_improve = 0;
            EPQ_STRUCTURE_DEBUG_LOG(
                    2,
                    "chain_tail iter=" << it
                                       << " candidates=" << candidates.size()
                                       << " dup_pruned=" << dup_pruned
                                       << " seen_pruned=" << seen_pruned
                                       << " children=" << children
                                       << " improved objective=" << current.J
                                       << " delta_iter=" << delta_iter
                                       << " groups=" << current.groups.size());
        } else {
            ++no_improve;
            EPQ_STRUCTURE_DEBUG_LOG(
                    2,
                    "chain_tail iter=" << it
                                       << " candidates=" << candidates.size()
                                       << " dup_pruned=" << dup_pruned
                                       << " seen_pruned=" << seen_pruned
                                       << " children=" << children
                                       << " no_improve=" << no_improve
                                       << " objective=" << current.J);
        }
    }

    proxy.chain_tail_profile = ChainTailProfile{
            .used = true,
            .iterations = run_stats.iterations,
            .iters_with_candidates = run_stats.iters_with_candidates,
            .seeds_raw_total = run_stats.seeds_raw_total,
            .seeds_kept_total = run_stats.seeds_kept_total,
            .candidates_total = run_stats.candidates_total,
            .exact_local_reranked_total = run_stats.exact_local_reranked_total,
            .exact_local_kept_total = run_stats.exact_local_kept_total,
            .local_gate_pruned_total = run_stats.local_gate_pruned_total,
            .donor_small_stops_total = run_stats.donor_small_stops_total,
            .no_step_stops_total = run_stats.no_step_stops_total,
            .prefix_cut_stops_total = run_stats.prefix_cut_stops_total,
            .total_steps = run_stats.total_steps,
            .max_steps = run_stats.max_steps,
            .exact_attempted = run_stats.exact_attempted,
            .exact_children = run_stats.exact_children,
            .exact_dup_pruned = run_stats.exact_dup_pruned,
            .exact_seen_pruned = run_stats.exact_seen_pruned,
            .improved_iters = run_stats.improved_iters,
    };
    EPQ_STRUCTURE_DEBUG_LOG(
            2,
            "chain_tail end groups=" << best.groups.size()
                                     << " objective=" << best.J);
    return {std::move(best.groups), std::move(best.bits)};
}

}  // namespace epq::structure_builder_internal
