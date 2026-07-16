#include "structure_builder_internal.h"

namespace epq::structure_builder_internal {

using GidDimsMap = std::unordered_map<int, DimsKey>;
using GidAdjMap = std::unordered_map<int, std::unordered_set<int>>;

GidAdjMap rebuild_gid_adj_from_dim_neigh(
        int d,
        const std::vector<int>& active,
        const GidDimsMap& gid_dims,
        const std::vector<std::vector<int>>& dim_neigh) {
    GidAdjMap gid_adj;
    std::vector<int> dim2gid(static_cast<size_t>(d), -1);
    for (int gid : active) {
        gid_adj[gid] = {};
        for (int dim : gid_dims.at(gid)) {
            dim2gid[static_cast<size_t>(dim)] = gid;
        }
    }
    for (int dim = 0; dim < d; ++dim) {
        const int gi = dim2gid[static_cast<size_t>(dim)];
        for (int nb_dim : dim_neigh[static_cast<size_t>(dim)]) {
            const int gj = dim2gid[static_cast<size_t>(nb_dim)];
            if (gi >= 0 && gj >= 0 && gi != gj) {
                gid_adj[gi].insert(gj);
                gid_adj[gj].insert(gi);
            }
        }
    }
    return gid_adj;
}

std::vector<std::pair<int, int>> two_hop_edges(
        const std::vector<int>& active,
        const GidAdjMap& gid_adj,
        std::mt19937& rng,
        int per_gid) {
    if (per_gid <= 0) {
        return {};
    }
    std::unordered_set<std::pair<int, int>, IntPairHash> seen;
    std::vector<std::pair<int, int>> edges;
    const std::unordered_set<int> active_set(active.begin(), active.end());
    for (int g : active) {
        std::vector<int> ng(gid_adj.at(g).begin(), gid_adj.at(g).end());
        if (ng.size() < 2) {
            continue;
        }
        if (ng.size() > 24) {
            ng = sample_vector(ng, 24, rng);
        }
        std::unordered_map<int, int> cand;
        for (int u : ng) {
            for (int v : gid_adj.at(u)) {
                if (v == g || !active_set.count(v) || gid_adj.at(g).count(v)) {
                    continue;
                }
                cand[v] += 1;
            }
        }
        std::vector<std::pair<int, int>> ranked(cand.begin(), cand.end());
        std::sort(
                ranked.begin(),
                ranked.end(),
                [](const auto& lhs, const auto& rhs) {
                    if (lhs.second != rhs.second) {
                        return lhs.second > rhs.second;
                    }
                    return lhs.first < rhs.first;
                });
        for (size_t i = 0; i < ranked.size() && static_cast<int>(i) < per_gid; ++i) {
            int a = g;
            int b = ranked[i].first;
            if (a > b) {
                std::swap(a, b);
            }
            if (a != b && seen.insert({a, b}).second) {
                edges.push_back({a, b});
            }
        }
    }
    return edges;
}

std::vector<std::pair<int, int>> random_long_edges(
        const std::vector<int>& active,
        const GidDimsMap& gid_dims,
        std::mt19937& rng,
        int count,
        float power) {
    if (count <= 0 || active.size() <= 2) {
        return {};
    }
    std::vector<double> weights;
    weights.reserve(active.size());
    for (int gid : active) {
        weights.push_back(std::pow(
                std::max(1.0, static_cast<double>(gid_dims.at(gid).size())),
                std::max(0.0f, power)));
    }
    std::discrete_distribution<int> pick(weights.begin(), weights.end());
    std::unordered_set<std::pair<int, int>, IntPairHash> uniq;
    int rounds = 0;
    while (static_cast<int>(uniq.size()) < count && rounds < count * 64) {
        const int ia = pick(rng);
        const int ib = pick(rng);
        ++rounds;
        if (ia == ib) {
            continue;
        }
        int a = active[static_cast<size_t>(std::min(ia, ib))];
        int b = active[static_cast<size_t>(std::max(ia, ib))];
        if (a != b) {
            uniq.insert({a, b});
        }
    }
    return {uniq.begin(), uniq.end()};
}

std::vector<std::pair<int, int>> propose_edges(
        const RefinedStructureBuilder& cfg,
        const std::vector<int>& active,
        const GidAdjMap& gid_adj,
        const GidDimsMap& gid_dims,
        std::mt19937& rng,
        int max_pool) {
    const float w_corr = std::max(0.0f, cfg.crystallize_weight_corr);
    const float w_long = std::max(0.0f, cfg.crystallize_weight_long);
    if (w_corr + w_long <= 0.0f) {
        return {};
    }
    int q_corr = w_corr > 0.0f
            ? static_cast<int>(std::lround(max_pool * (w_corr / (w_corr + w_long))))
            : 0;
    int q_long = max_pool - q_corr;

    std::unordered_set<std::pair<int, int>, IntPairHash> edges;
    if (q_corr > 0) {
        const int q_two = static_cast<int>(
                std::lround(q_corr * std::clamp(cfg.crystallize_corr_two_hop_ratio, 0.0f, 1.0f)));
        const int q_adj = std::max(0, q_corr - q_two);
        std::vector<std::pair<int, int>> adj_pool;
        for (int g : active) {
            for (int h : gid_adj.at(g)) {
                int a = g;
                int b = h;
                if (a > b) {
                    std::swap(a, b);
                }
                if (a != b) {
                    adj_pool.push_back({a, b});
                }
            }
        }
        std::sort(adj_pool.begin(), adj_pool.end());
        adj_pool.erase(std::unique(adj_pool.begin(), adj_pool.end()), adj_pool.end());
        for (const auto& edge : sample_vector(adj_pool, q_adj, rng)) {
            edges.insert(edge);
        }
        const auto two_pool = two_hop_edges(
                active, gid_adj, rng, cfg.crystallize_corr_two_hop_per_gid);
        for (const auto& edge : sample_vector(two_pool, q_two, rng)) {
            edges.insert(edge);
        }
    }
    if (q_long > 0) {
        const int m_gen = static_cast<int>(
                std::ceil(cfg.crystallize_long_oversample * q_long));
        const auto long_pool = random_long_edges(
                active, gid_dims, rng, m_gen, cfg.crystallize_long_edge_power);
        for (const auto& edge : sample_vector(long_pool, q_long, rng)) {
            edges.insert(edge);
        }
    }
    std::vector<std::pair<int, int>> out(edges.begin(), edges.end());
    if (static_cast<int>(out.size()) > max_pool) {
        out = sample_vector(out, max_pool, rng);
    }
    return out;
}

void apply_merge_in_place(
        int a,
        int b,
        const DimsKey& merged_dims,
        GidDimsMap& gid_dims,
        GidAdjMap& gid_adj,
        std::vector<int>& active,
        int next_gid) {
    const int z = next_gid;
    gid_dims[z] = merged_dims;
    std::unordered_set<int> nz = gid_adj[a];
    nz.insert(gid_adj[b].begin(), gid_adj[b].end());
    nz.erase(a);
    nz.erase(b);

    active.erase(std::remove(active.begin(), active.end(), a), active.end());
    active.erase(std::remove(active.begin(), active.end(), b), active.end());
    for (int g : nz) {
        gid_adj[g].erase(a);
        gid_adj[g].erase(b);
    }
    gid_adj.erase(a);
    gid_adj.erase(b);
    gid_dims.erase(a);
    gid_dims.erase(b);

    active.push_back(z);
    gid_adj[z] = {};
    for (int g : nz) {
        gid_adj[z].insert(g);
        gid_adj[g].insert(z);
    }
}

struct CrystalState {
    std::vector<int> active;
    GidDimsMap gid_dims;
    GidAdjMap gid_adj;
    int next_gid = 0;
    double J = 0.0;
    Bits bits;
    std::vector<int> gids;
};

Groups groups_from_crystal_state(const CrystalState& state) {
    Groups groups;
    groups.reserve(state.gids.size());
    for (int gid : state.gids) {
        groups.push_back(state.gid_dims.at(gid));
    }
    return groups;
}

std::tuple<double, Bits, std::vector<int>> solve_crystal_state(
        ProxyContext& proxy,
        const std::vector<int>& active,
        const GidDimsMap& gid_dims) {
    std::vector<int> gids = active;
    std::sort(gids.begin(), gids.end(), [&](int lhs, int rhs) {
        const auto& gl = gid_dims.at(lhs);
        const auto& gr = gid_dims.at(rhs);
        if (gl.size() != gr.size()) {
            return gl.size() < gr.size();
        }
        if (gl.empty() || gr.empty()) {
            return lhs < rhs;
        }
        return gl.front() < gr.front();
    });
    Groups groups;
    groups.reserve(gids.size());
    for (int gid : gids) {
        groups.push_back(gid_dims.at(gid));
    }
    auto alloc = proxy.solve_bits(groups);
    return {alloc.J, alloc.bits, gids};
}

std::pair<Groups, Bits> run_crystallize_stage(
        const RefinedStructureBuilder& cfg,
        ProxyContext& proxy,
        const BuildContext& ctx,
        const Groups& groups,
        const Bits& bits_in) {
    std::mt19937 rng(cfg.seed);
    GidDimsMap gid_dims;
    std::vector<int> active;
    int next_gid = 0;
    for (const auto& group : groups) {
        DimsKey dims = group;
        std::sort(dims.begin(), dims.end());
        gid_dims[next_gid] = std::move(dims);
        active.push_back(next_gid++);
    }
    GidAdjMap gid_adj;
    if (cfg.crystallize_weight_corr > 0.0f) {
        const auto dim_neigh = build_dim_neighbors_by_corr(
                proxy.xt_train,
                cfg.crystallize_corr_adj_k,
                cfg.crystallize_corr_adj_abs,
                cfg.crystallize_corr_adj_rows,
                cfg.seed);
        gid_adj = rebuild_gid_adj_from_dim_neigh(
                ctx.d, active, gid_dims, dim_neigh);
    } else {
        for (int gid : active) {
            gid_adj[gid] = {};
        }
    }

    auto [J0, bits0, gids0] = solve_crystal_state(proxy, active, gid_dims);
    CrystalState root{
            .active = active,
            .gid_dims = gid_dims,
            .gid_adj = gid_adj,
            .next_gid = next_gid,
            .J = J0,
            .bits = bits0,
            .gids = gids0,
    };
    (void)bits_in;
    CrystalState best = root;
    std::vector<CrystalState> beam{root};
    const int min_groups = min_feasible_groups(ctx);
#if EPQ_ENABLE_STRUCTURE_TRACE
    trace_structure_candidate(
            ctx,
            "crystallize_root",
            0,
            groups_from_crystal_state(root),
            root.bits,
            root.J,
            "crystallize");
#endif

    EPQ_STRUCTURE_DEBUG_LOG(
            2,
            "crystallize begin groups=" << groups.size()
                                        << " objective=" << root.J
                                        << " min_groups=" << min_groups);
    const bool use_fast_proxy = proxy.fast_pca_top_dims > 0;

    for (int depth = 1; depth <= cfg.crystallize_beam_max_depth; ++depth) {
        if (static_cast<int>(best.active.size()) <= min_groups ||
            static_cast<int>(best.active.size()) <= 1) {
            break;
        }
        const size_t beam_in = beam.size();
        int states_expanded = 0;
        size_t total_edges = 0;
        size_t total_proxy_pairs = 0;
        size_t total_struct_candidates = 0;
        std::vector<CrystalState> children_all;
        for (const auto& state : beam) {
            if (static_cast<int>(state.active.size()) <= min_groups ||
                static_cast<int>(state.active.size()) <= 1) {
                continue;
            }
            ++states_expanded;
            std::unordered_map<int, int> gid2b;
            for (size_t i = 0; i < state.gids.size(); ++i) {
                gid2b[state.gids[i]] = state.bits[i];
            }
            std::unordered_map<int, double> D_b0;
            std::unordered_map<int, double> D_assigned;
            for (int gid : state.active) {
                D_b0[gid] = use_fast_proxy
                        ? proxy.D_fast(
                                  state.gid_dims.at(gid),
                                  std::clamp(cfg.crystallize_proxy_bits, 0, ctx.max_bits))
                        : proxy.D(
                                  state.gid_dims.at(gid),
                                  std::clamp(cfg.crystallize_proxy_bits, 0, ctx.max_bits));
                D_assigned[gid] = proxy.D(
                        state.gid_dims.at(gid),
                        gid2b.count(gid) ? gid2b[gid] : 0);
            }
            const int max_pool = std::max(
                    256,
                    cfg.crystallize_pool_mult * cfg.crystallize_candidates);
            auto edges = propose_edges(
                    cfg,
                    state.active,
                    state.gid_adj,
                    state.gid_dims,
                    rng,
                    max_pool);
            total_edges += edges.size();
            std::vector<std::tuple<double, int, int, DimsKey>> proxy_list;
            for (const auto& [a, b] : edges) {
                const auto& da = state.gid_dims.at(a);
                const auto& db = state.gid_dims.at(b);
                if (static_cast<int>(da.size() + db.size()) > cfg.crystallize_dmax) {
                    continue;
                }
                DimsKey merged = da;
                merged.insert(merged.end(), db.begin(), db.end());
                std::sort(merged.begin(), merged.end());
                const double Dz = use_fast_proxy
                        ? proxy.D_fast(
                                  merged,
                                  std::clamp(cfg.crystallize_proxy_bits, 0, ctx.max_bits))
                        : proxy.D(
                                  merged,
                                  std::clamp(cfg.crystallize_proxy_bits, 0, ctx.max_bits));
                proxy_list.push_back({Dz - D_b0[a] - D_b0[b], a, b, std::move(merged)});
            }
            total_proxy_pairs += proxy_list.size();
            std::sort(proxy_list.begin(), proxy_list.end(), [](const auto& lhs, const auto& rhs) {
                return std::get<0>(lhs) < std::get<0>(rhs);
            });
            std::unordered_map<int, int> used;
            std::vector<std::tuple<double, int, int, DimsKey, int, int>> struct_list;
            const int L =
                    std::max(1, cfg.crystallize_candidates * cfg.crystallize_shortlist_factor);
            for (size_t i = 0; i < proxy_list.size() && static_cast<int>(i) < L; ++i) {
                const auto& [_, a, b, merged] = proxy_list[i];
                if (used[a] >= cfg.crystallize_endpoint_quota ||
                    used[b] >= cfg.crystallize_endpoint_quota) {
                    continue;
                }
                used[a] += 1;
                used[b] += 1;
                const int bx = gid2b.count(a) ? gid2b[a] : 0;
                const int by = gid2b.count(b) ? gid2b[b] : 0;
                const int bz = std::min(ctx.max_bits, bx + by);
                const double Dz = proxy.D(merged, bz);
                const double dJ_struct = Dz - D_assigned[a] - D_assigned[b];
                if (dJ_struct <= cfg.crystallize_struct_tol) {
                    struct_list.push_back({dJ_struct, a, b, merged, bx, by});
                }
                if (static_cast<int>(struct_list.size()) >= cfg.crystallize_candidates) {
                    break;
                }
            }
            total_struct_candidates += struct_list.size();
            std::sort(struct_list.begin(), struct_list.end(), [](const auto& lhs, const auto& rhs) {
                return std::get<0>(lhs) < std::get<0>(rhs);
            });
            const int local_top = std::min<int>(cfg.crystallize_beam_topR, struct_list.size());
            for (int i = 0; i < local_top; ++i) {
                const auto& [_, a, b, merged, bx, by] = struct_list[static_cast<size_t>(i)];
                (void)bx;
                (void)by;
                CrystalState child = state;
                apply_merge_in_place(
                        a,
                        b,
                        merged,
                        child.gid_dims,
                        child.gid_adj,
                        child.active,
                        child.next_gid);
                child.next_gid += 1;
                auto [J_child, bits_child, gids_child] =
                        solve_crystal_state(proxy, child.active, child.gid_dims);
                child.J = J_child;
                child.bits = std::move(bits_child);
                child.gids = std::move(gids_child);
                children_all.push_back(std::move(child));
            }
        }
        if (children_all.empty()) {
            EPQ_STRUCTURE_DEBUG_LOG(
                    2,
                    "crystallize depth=" << depth
                                          << " no children, stop");
            break;
        }
        std::unordered_map<PartitionKey, CrystalState, PartitionKeyHash> best_by_key;
        for (auto& child : children_all) {
            Groups child_groups;
            child_groups.reserve(child.active.size());
            for (int gid : child.active) {
                child_groups.push_back(child.gid_dims.at(gid));
            }
            const PartitionKey key = canonical_partition_key(child_groups);
            auto it = best_by_key.find(key);
            if (it == best_by_key.end() || child.J < it->second.J) {
                best_by_key[key] = std::move(child);
            }
        }
        std::vector<CrystalState> uniq_children;
        for (auto& [_, child] : best_by_key) {
            uniq_children.push_back(std::move(child));
        }
        std::sort(uniq_children.begin(), uniq_children.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.J < rhs.J;
        });
        if (uniq_children.empty()) {
            EPQ_STRUCTURE_DEBUG_LOG(
                    2,
                    "crystallize depth=" << depth
                                          << " no unique children, stop");
            break;
        }
        if (uniq_children.front().J < best.J - 1e-12) {
            const double delta = best.J - uniq_children.front().J;
            best = uniq_children.front();
#if EPQ_ENABLE_STRUCTURE_TRACE
            trace_structure_candidate(
                    ctx,
                    "crystallize_best",
                    depth,
                    groups_from_crystal_state(best),
                    best.bits,
                    best.J,
                    "crystallize");
#endif
            EPQ_STRUCTURE_DEBUG_LOG(
                    2,
                    "crystallize depth=" << depth
                                          << " beam_in=" << beam_in
                                          << " states=" << states_expanded
                                          << " edges=" << total_edges
                                          << " proxy_pairs=" << total_proxy_pairs
                                          << " struct_candidates=" << total_struct_candidates
                                          << " children=" << children_all.size()
                                          << " best_groups=" << best.active.size()
                                          << " beam=" << uniq_children.size()
                                          << " delta=" << delta
                                          << " objective=" << best.J);
        } else {
            EPQ_STRUCTURE_DEBUG_LOG(
                    2,
                    "crystallize depth=" << depth
                                          << " beam_in=" << beam_in
                                          << " states=" << states_expanded
                                          << " edges=" << total_edges
                                          << " proxy_pairs=" << total_proxy_pairs
                                          << " struct_candidates=" << total_struct_candidates
                                          << " children=" << children_all.size()
                                          << " no improvement, stop");
            break;
        }
        beam = uniq_children;
        if (static_cast<int>(beam.size()) > cfg.crystallize_beam_width) {
            beam.resize(static_cast<size_t>(cfg.crystallize_beam_width));
        }
    }

    Groups final_groups;
    final_groups.reserve(best.gids.size());
    for (int gid : best.gids) {
        final_groups.push_back(best.gid_dims.at(gid));
    }
    validate_partition(final_groups, ctx.d, true);
    EPQ_STRUCTURE_DEBUG_LOG(
            2,
            "crystallize end groups=" << final_groups.size()
                                      << " objective=" << best.J);
    return {std::move(final_groups), best.bits};
}

}  // namespace epq::structure_builder_internal
