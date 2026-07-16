#include "structure_builder_internal.h"

namespace epq::structure_builder_internal {

struct GrowGroupState {
    std::vector<int> dims;
    std::unordered_map<int, float> wsum;
    std::unordered_map<int, int> votes;
    double Dcur = 0.0;
    int b_ref = 0;
};

GrowGroupState init_group_state(
        const std::vector<int>& dims,
        const std::vector<char>& unassigned,
        const std::vector<std::vector<std::pair<int, float>>>& adj,
        ProxyContext& proxy,
        int b_ref) {
    GrowGroupState st;
    st.dims = dims;
    st.b_ref = b_ref;
    for (int v : dims) {
        for (const auto& [u, w] : adj[static_cast<size_t>(v)]) {
            if (!unassigned[static_cast<size_t>(u)]) {
                continue;
            }
            st.wsum[u] += w;
            st.votes[u] += 1;
        }
    }
    st.Dcur = proxy.D(st.dims, b_ref);
    return st;
}

void update_frontier_add_dim(
        GrowGroupState& st,
        int v,
        const std::vector<char>& unassigned,
        const std::vector<std::vector<std::pair<int, float>>>& adj) {
    st.wsum.erase(v);
    st.votes.erase(v);
    for (const auto& [u, w] : adj[static_cast<size_t>(v)]) {
        if (!unassigned[static_cast<size_t>(u)]) {
            continue;
        }
        st.wsum[u] += w;
        st.votes[u] += 1;
    }
}

std::pair<Groups, Bits> run_grow_stage(
        const RefinedStructureBuilder& cfg,
        ProxyContext& proxy,
        const BuildContext& ctx) {
    const int d = ctx.d;
    const int B = ctx.total_bits;
    const int proxy_bmax = ctx.max_bits;
    std::mt19937 rng(cfg.seed);

    int M0 = 0;
    if (cfg.grow_target_groups > 0) {
        M0 = cfg.grow_target_groups;
    } else {
        const int Mpq = B > 0 ? std::max(1, B / 8) : 1;
        M0 = std::max(
                cfg.grow_min_groups,
                static_cast<int>(std::lround(cfg.grow_alpha_groups * Mpq)));
    }
    if (cfg.grow_max_groups > 0) {
        M0 = std::min(M0, cfg.grow_max_groups);
    }
    M0 = std::clamp(M0, 1, d);
    const int dmax = std::max(1, cfg.grow_dmax);
    const int min_groups_by_dmax = (d + dmax - 1) / dmax;
    const int min_groups_by_bits = min_feasible_groups(ctx);
    M0 = std::max(M0, min_groups_by_dmax);
    M0 = std::max(M0, min_groups_by_bits);
    if (ctx.min_bits > 0) {
        M0 = std::min(M0, std::max(1, ctx.total_bits / ctx.min_bits));
        M0 = std::max(M0, min_groups_by_bits);
    }
    M0 = std::min(M0, d);

    EPQ_STRUCTURE_DEBUG_LOG(
            2,
            "grow begin d=" << d << " B=" << B << " seed_groups=" << M0
                            << " dmax=" << dmax);

    const auto neigh = build_dim_neighbors_by_corr_weighted(
            proxy.xt_train,
            cfg.grow_corr_adj_k,
            cfg.grow_corr_adj_abs,
            cfg.grow_corr_adj_rows,
            cfg.seed,
            cfg.grow_edge_tau);

    std::vector<std::vector<std::pair<int, float>>> adj(
            static_cast<size_t>(d));
    for (int i = 0; i < d; ++i) {
        for (const auto& [j, w] : neigh[static_cast<size_t>(i)]) {
            adj[static_cast<size_t>(i)].push_back({j, w});
            adj[static_cast<size_t>(j)].push_back({i, w});
        }
    }

    std::vector<char> unassigned(static_cast<size_t>(d), 1);
    const int b_seed = score_bits_for_group(
            d, B, proxy_bmax, cfg.grow_score_bits_fixed);

    Groups groups;
    groups.reserve(static_cast<size_t>(M0));
    for (int gi = 0; gi < M0 && std::any_of(unassigned.begin(), unassigned.end(), [](char v) {
             return v != 0;
         });
         ++gi) {
        std::vector<double> scores(
                static_cast<size_t>(d), -std::numeric_limits<double>::infinity());
        std::vector<int> active_dims;
        for (int dim = 0; dim < d; ++dim) {
            if (!unassigned[static_cast<size_t>(dim)]) {
                continue;
            }
            scores[static_cast<size_t>(dim)] = proxy.D({dim}, b_seed);
            active_dims.push_back(dim);
        }
        if (active_dims.empty()) {
            break;
        }
        std::sort(active_dims.begin(), active_dims.end(), [&](int lhs, int rhs) {
            return scores[static_cast<size_t>(lhs)] > scores[static_cast<size_t>(rhs)];
        });
        if (static_cast<int>(active_dims.size()) > cfg.grow_seed_topk) {
            active_dims.resize(static_cast<size_t>(cfg.grow_seed_topk));
        }
        const int seed =
                active_dims[static_cast<size_t>(std::uniform_int_distribution<int>(
                        0,
                        static_cast<int>(active_dims.size()) - 1)(rng))];
        unassigned[static_cast<size_t>(seed)] = 0;
        std::vector<int> group{seed};

        if (cfg.grow_seed_pair) {
            std::vector<int> candidates;
            for (const auto& [j, _] : adj[static_cast<size_t>(seed)]) {
                if (unassigned[static_cast<size_t>(j)]) {
                    candidates.push_back(j);
                }
            }
            shuffle_vector(candidates, rng);
            if (static_cast<int>(candidates.size()) >
                std::max(8, cfg.grow_rerank_L)) {
                candidates.resize(
                        static_cast<size_t>(std::max(8, cfg.grow_rerank_L)));
            }
            const double D0 = proxy.D({seed}, b_seed);
            int best_j = -1;
            double best_gain = 0.0;
            for (int j : candidates) {
                const double D1 = proxy.D({seed, j}, b_seed);
                const double gain = D0 - D1;
                if (gain > best_gain) {
                    best_gain = gain;
                    best_j = j;
                }
            }
            if (best_j >= 0 && best_gain > 0.0) {
                unassigned[static_cast<size_t>(best_j)] = 0;
                group.push_back(best_j);
            }
        }

        while (static_cast<int>(group.size()) < cfg.grow_min_group_size) {
            std::vector<int> extras;
            for (int dim = 0; dim < d; ++dim) {
                if (unassigned[static_cast<size_t>(dim)]) {
                    extras.push_back(dim);
                }
            }
            if (extras.empty()) {
                break;
            }
            const int take =
                    std::uniform_int_distribution<int>(0, static_cast<int>(extras.size()) - 1)(rng);
            const int dim = extras[static_cast<size_t>(take)];
            unassigned[static_cast<size_t>(dim)] = 0;
            group.push_back(dim);
        }
        groups.push_back(std::move(group));
    }

    while (static_cast<int>(groups.size()) < M0) {
        int dim = -1;
        for (int i = 0; i < d; ++i) {
            if (unassigned[static_cast<size_t>(i)]) {
                dim = i;
                break;
            }
        }
        if (dim < 0) {
            break;
        }
        unassigned[static_cast<size_t>(dim)] = 0;
        groups.push_back({dim});
    }

    std::vector<GrowGroupState> states;
    states.reserve(groups.size());
    for (const auto& group : groups) {
        const int b_ref =
                score_bits_for_group(d, B, proxy_bmax, cfg.grow_score_bits_fixed);
        states.push_back(init_group_state(group, unassigned, adj, proxy, b_ref));
    }

    auto has_unassigned = [&]() {
        return std::any_of(unassigned.begin(), unassigned.end(), [](char v) {
            return v != 0;
        });
    };

    while (has_unassigned()) {
        int best_group = -1;
        int best_u = -1;
        double best_D1 = 0.0;
        double best_gain = -std::numeric_limits<double>::infinity();

        for (size_t gi = 0; gi < states.size(); ++gi) {
            auto& st = states[gi];
            if (static_cast<int>(st.dims.size()) >= dmax || st.wsum.empty()) {
                continue;
            }
            std::vector<std::tuple<int, float, int>> items;
            for (const auto& [u, w] : st.wsum) {
                if (!unassigned[static_cast<size_t>(u)]) {
                    continue;
                }
                const int votes =
                        st.votes.count(u) ? st.votes[u] : 0;
                if (cfg.grow_min_votes > 0 && votes < cfg.grow_min_votes) {
                    continue;
                }
                items.push_back({u, w, votes});
            }
            if (items.empty() && cfg.grow_fill_when_stuck) {
                for (const auto& [u, w] : st.wsum) {
                    if (!unassigned[static_cast<size_t>(u)]) {
                        continue;
                    }
                    items.push_back({u, w, st.votes.count(u) ? st.votes[u] : 0});
                }
            }
            if (items.empty()) {
                continue;
            }
            std::sort(
                    items.begin(),
                    items.end(),
                    [](const auto& lhs, const auto& rhs) {
                        return std::get<1>(lhs) > std::get<1>(rhs);
                    });
            std::vector<int> cand;
            for (size_t i = 0;
                 i < items.size() && static_cast<int>(i) < cfg.grow_rerank_L;
                 ++i) {
                cand.push_back(std::get<0>(items[i]));
            }
            shuffle_vector(cand, rng);

            int local_best_u = -1;
            double local_best_D1 = 0.0;
            double local_best_gain = -std::numeric_limits<double>::infinity();
            for (int u : cand) {
                std::vector<int> dims_new = st.dims;
                dims_new.push_back(u);
                const double D1 = proxy.D(dims_new, st.b_ref);
                const double gain = st.Dcur - D1;
                if (cfg.grow_avg_gain_tau > 0.0f &&
                    !cfg.grow_fill_when_stuck &&
                    gain / std::max<int>(1, st.dims.size()) <
                            cfg.grow_avg_gain_tau) {
                    continue;
                }
                if (gain > local_best_gain) {
                    local_best_gain = gain;
                    local_best_u = u;
                    local_best_D1 = D1;
                }
            }
            if (local_best_u < 0 && cfg.grow_fill_when_stuck && !cand.empty()) {
                local_best_u = cand.front();
                std::vector<int> dims_new = st.dims;
                dims_new.push_back(local_best_u);
                local_best_D1 = proxy.D(dims_new, st.b_ref);
                local_best_gain = st.Dcur - local_best_D1;
            }
            if (local_best_u >= 0 && local_best_gain > best_gain) {
                best_gain = local_best_gain;
                best_group = static_cast<int>(gi);
                best_u = local_best_u;
                best_D1 = local_best_D1;
            }
        }

        if (best_group < 0 || best_u < 0) {
            for (int dim = 0; dim < d; ++dim) {
                if (!unassigned[static_cast<size_t>(dim)]) {
                    continue;
                }
                bool placed = false;
                for (auto& st : states) {
                    if (static_cast<int>(st.dims.size()) < dmax) {
                        unassigned[static_cast<size_t>(dim)] = 0;
                        st.dims.push_back(dim);
                        update_frontier_add_dim(st, dim, unassigned, adj);
                        st.Dcur = proxy.D(st.dims, st.b_ref);
                        placed = true;
                        break;
                    }
                }
                if (!placed) {
                    unassigned[static_cast<size_t>(dim)] = 0;
                    const int b_ref = score_bits_for_group(
                            d, B, proxy_bmax, cfg.grow_score_bits_fixed);
                    states.push_back(init_group_state({dim}, unassigned, adj, proxy, b_ref));
                }
            }
            break;
        }

        auto& st = states[static_cast<size_t>(best_group)];
        unassigned[static_cast<size_t>(best_u)] = 0;
        st.dims.push_back(best_u);
        update_frontier_add_dim(st, best_u, unassigned, adj);
        st.Dcur = best_D1;
    }

    Groups final_groups;
    final_groups.reserve(states.size());
    for (const auto& st : states) {
        final_groups.push_back(st.dims);
    }
    validate_partition(final_groups, d, true);
    auto alloc = proxy.solve_bits(final_groups);
#if EPQ_ENABLE_STRUCTURE_TRACE
    trace_structure_candidate(
            ctx,
            "grow_final",
            0,
            final_groups,
            alloc.bits,
            alloc.J,
            "grow");
#endif
    EPQ_STRUCTURE_DEBUG_LOG(
            2,
            "grow end groups=" << final_groups.size()
                               << " objective=" << alloc.J);
    return {std::move(final_groups), std::move(alloc.bits)};
}

}  // namespace epq::structure_builder_internal
