#include "structure_builder_local_search.h"

namespace epq::structure_builder_internal {

std::pair<Groups, Bits> run_greedy_tail_stage(
        const RefinedStructureBuilder& cfg,
        ProxyContext& proxy,
        const BuildContext& ctx,
        const Groups& groups,
        const Bits& bits) {
    (void)bits;
    std::mt19937 rng(cfg.seed);
    const SearchConfig search_cfg = make_greedy_tail_search_config(cfg);
    auto alloc0 = proxy.solve_bits(groups);
    BeamState current{groups, alloc0.bits, alloc0.J};
    BeamState best = current;
    SeenWindow seen(cfg.greedy_tail_seen_window);
    seen.set_best(canonical_partition_key(groups), alloc0.J);
    int no_improve = 0;

    EPQ_STRUCTURE_DEBUG_LOG(
            2,
            "greedy_tail begin groups=" << groups.size()
                                        << " objective=" << alloc0.J
                                        << " eval_topk=" << cfg.greedy_tail_eval_topk);

    for (int it = 0; it < cfg.greedy_tail_iters; ++it) {
        if (cfg.greedy_tail_patience > 0 &&
            no_improve >= cfg.greedy_tail_patience) {
            EPQ_STRUCTURE_DEBUG_LOG(
                    2,
                    "greedy_tail iter=" << it
                                        << " patience hit, stop");
            break;
        }
        seen.next_round();
        MoveBuildStats move_stats;
        auto moves = build_shortlisted_moves(
                proxy, current.groups, current.bits, search_cfg, rng, &move_stats);
        size_t dup_pruned = 0;
        size_t seen_pruned = 0;
        size_t children = 0;
        std::optional<BeamState> iter_best;

        if (moves.empty()) {
            EPQ_STRUCTURE_DEBUG_LOG(
                    2,
                    "greedy_tail iter=" << it
                                        << " no moves, stop");
            break;
        }

        int attempted = 0;
        std::unordered_set<PartitionKey, PartitionKeyHash> local_partitions;
        for (const auto& mv : moves) {
            if (attempted >= cfg.greedy_tail_eval_topk) {
                break;
            }
            ++attempted;
            Groups cand_groups;
            try {
                if (mv.kind == Move::Kind::kRelocate) {
                    cand_groups = apply_relocate(
                            current.groups,
                            mv.groups.first,
                            mv.groups.second,
                            mv.dims.front());
                } else {
                    cand_groups = apply_swap(
                            current.groups,
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
                ++dup_pruned;
                continue;
            }
            if (seen.get_best(key).has_value()) {
                ++seen_pruned;
                continue;
            }
            auto alloc = proxy.solve_bits(cand_groups);
            seen.set_best(key, alloc.J);
            ++children;
            if (!iter_best.has_value() || alloc.J < iter_best->J) {
                iter_best = BeamState{
                        .groups = std::move(cand_groups),
                        .bits = std::move(alloc.bits),
                        .J = alloc.J,
                };
            }
        }

        if (iter_best.has_value() &&
            iter_best->J < current.J - cfg.greedy_tail_eps_improve) {
            const double delta_iter = current.J - iter_best->J;
            current = std::move(*iter_best);
            if (current.J < best.J - cfg.greedy_tail_eps_improve) {
                best = current;
            }
            no_improve = 0;
            EPQ_STRUCTURE_DEBUG_LOG(
                    2,
                    "greedy_tail iter=" << it
                                        << " raw_moves=" << move_stats.raw
                                        << " shortlisted=" << move_stats.shortlisted
                                        << " pair_pruned=" << move_stats.pair_pruned
                                        << " score_pruned=" << move_stats.score_pruned
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
                    "greedy_tail iter=" << it
                                        << " raw_moves=" << move_stats.raw
                                        << " shortlisted=" << move_stats.shortlisted
                                        << " pair_pruned=" << move_stats.pair_pruned
                                        << " score_pruned=" << move_stats.score_pruned
                                        << " dup_pruned=" << dup_pruned
                                        << " seen_pruned=" << seen_pruned
                                        << " children=" << children
                                        << " no_improve=" << no_improve
                                        << " objective=" << current.J);
        }
    }

    EPQ_STRUCTURE_DEBUG_LOG(
            2,
            "greedy_tail end groups=" << best.groups.size()
                                      << " objective=" << best.J);
    return {std::move(best.groups), std::move(best.bits)};
}

}  // namespace epq::structure_builder_internal
