#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""elasticpq_crystallization_forwarder.py

CrystallizationForwarder (no-caps)
----------------------------------
Merge-only BEAM search forwarder for EPQ/ElasticPQ.

Design:
- No heat phase (no split / no melt)
- No caps / caps repair
- Bit allocation is delegated to ctx.solve_bits() (DP oracle)
- Per-group upper bound is ctx.bmax (the ONLY bound)
- Proposal generation: corr-adj (+ optional two-hop) + random long edges
- Candidate selection: proxy shortlist + endpoint quota
- Structural filter: evaluate dJ_struct at assigned bits (bx,by)->bz=min(bmax,bx+by)
- Beam dedup: canonical partition key

Notes:
- ctx.proxy must be attached (cached)
- ctx.solve_bits() must be available (DP allocator)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from grouper import EPQContext, ForwardingGrouper, Groups, Bits


# ============================================================
# Config
# ============================================================


@dataclass
class CrystallizationForwarderConfig:
    # constraints
    dmax: int = 1024

    # candidate sizing
    N_candidates: int = 128
    shortlist_factor: int = 4
    pool_mult: int = 16

    # proposals mixture
    proposal_weight_corr: float = 0.4
    proposal_weight_long: float = 0.6

    # corr-adj
    corr_adj_k: int = 16
    corr_adj_abs: bool = True
    corr_adj_rows: int = 4096

    # two-hop inside corr
    corr_two_hop_ratio: float = 0.25
    corr_two_hop_per_gid: int = 4

    # random long edges
    long_oversample: float = 2.0
    long_edge_power: float = 0.5

    # diversity quota (endpoint quota)
    endpoint_quota: int = 12

    # proxy bits for shortlist ranking
    proxy_b0: int = 4

    # structural filter (<= passes)
    dJ_struct_tol: float = 1e-6

    # beam
    beam_width: int = 8
    beam_topR: int = 8
    beam_max_depth: int = 1_000_000

    # logs
    verbose: bool = False


# ============================================================
# Utils
# ============================================================


def _tuple_sorted(dims: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(i) for i in dims))


def _canonical_partition_key(active: List[int], gid_dims: Dict[int, Tuple[int, ...]]) -> Tuple[Tuple[int, ...], ...]:
    parts = [gid_dims[g] for g in active]
    parts.sort()
    return tuple(parts)


def _subsample_rows(x: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or x.shape[0] <= max_rows:
        return x
    rs = np.random.RandomState(int(seed))
    idx = rs.choice(x.shape[0], size=int(max_rows), replace=False)
    return x[idx]


# ============================================================
# Corr-adj neighbors (dim-level)
# ============================================================


def _build_dim_neighbors_by_corr(
    xt: np.ndarray,
    *,
    knn: int,
    abs_corr: bool,
    max_rows: int,
    seed: int,
) -> List[List[int]]:
    x = xt
    if max_rows > 0 and x.shape[0] > max_rows:
        x = _subsample_rows(x, max_rows=int(max_rows), seed=int(seed))

    x = np.ascontiguousarray(x, dtype=np.float32)
    n, d = x.shape

    if n <= 1:
        neigh = [[] for _ in range(d)]
        half = max(1, int(knn) // 2)
        for i in range(d):
            for j in range(max(0, i - half), min(d, i + half + 1)):
                if j != i:
                    neigh[i].append(j)
        return neigh

    x = x - x.mean(axis=0, keepdims=True)
    std = np.maximum(x.std(axis=0, ddof=0, keepdims=True), 1e-6)
    x = x / std

    corr = (x.T @ x) / float(n)
    corr = np.clip(corr, -1.0, 1.0)
    score = np.abs(corr) if bool(abs_corr) else corr
    np.fill_diagonal(score, -np.inf)

    kk = min(int(knn), d - 1)
    neigh: List[List[int]] = [[] for _ in range(d)]
    for i in range(d):
        idx = np.argpartition(-score[i], kth=kk - 1)[:kk]
        idx = idx[np.argsort(-score[i, idx])]
        neigh[i] = [int(j) for j in idx]
    return neigh


def _rebuild_gid_adj_from_dim_neigh(
    *,
    d: int,
    active: List[int],
    gid_dims: Dict[int, Tuple[int, ...]],
    dim_neigh: List[List[int]],
) -> Dict[int, set]:
    gid_adj: Dict[int, set] = {gid: set() for gid in active}
    dim2gid = np.full(int(d), -1, dtype=np.int64)
    for gid in active:
        for dim in gid_dims[gid]:
            dim2gid[int(dim)] = int(gid)

    for dim in range(int(d)):
        gi = int(dim2gid[dim])
        for nb_dim in dim_neigh[dim]:
            gj = int(dim2gid[int(nb_dim)])
            if gi != gj:
                gid_adj[gi].add(gj)
                gid_adj[gj].add(gi)
    return gid_adj


# ============================================================
# Proposal augmentation
# ============================================================


def _two_hop_edges(
    *,
    active: List[int],
    gid_adj: Dict[int, set],
    rng: np.random.RandomState,
    per_gid: int,
) -> List[Tuple[int, int]]:
    if per_gid <= 0:
        return []

    edges: List[Tuple[int, int]] = []
    seen = set()
    active_set = set(active)

    for g in active:
        ng = list(gid_adj.get(g, set()))
        if len(ng) < 2:
            continue
        if len(ng) > 24:
            ng = list(rng.choice(ng, size=24, replace=False))

        cand: Dict[int, int] = {}
        g_neigh = gid_adj.get(g, set())
        for u in ng:
            for v in gid_adj.get(u, set()):
                if v == g or v not in active_set or v in g_neigh:
                    continue
                cand[v] = cand.get(v, 0) + 1

        if not cand:
            continue

        for v, _cnt in sorted(cand.items(), key=lambda kv: (-kv[1], kv[0]))[: int(per_gid)]:
            a, b = (g, v) if g < v else (v, g)
            if a != b and (a, b) not in seen:
                seen.add((a, b))
                edges.append((a, b))

    return edges


def _random_long_edges(
    *,
    active: List[int],
    gid_dims: Dict[int, Tuple[int, ...]],
    rng: np.random.RandomState,
    m: int,
    power: float,
) -> List[Tuple[int, int]]:
    G = int(len(active))
    m = int(m)
    if m <= 0 or G <= 2:
        return []

    imp = np.array([max(1.0, float(len(gid_dims[g]))) for g in active], dtype=np.float64)
    pw = float(max(0.0, power))
    w = imp**pw
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0.0:
        w = np.full(G, 1.0 / G, dtype=np.float64)
    else:
        w = w / s

    tri_size = (G * (G - 1)) // 2
    visited = np.zeros(tri_size, dtype=np.bool_)

    def tri_idx(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        i = i.astype(np.int64, copy=False)
        j = j.astype(np.int64, copy=False)
        return (i * (G - 1) - (i * (i + 1)) // 2 + (j - i - 1)).astype(np.int64, copy=False)

    edges: List[Tuple[int, int]] = []
    batch = int(max(1024, min(1 << 20, m * 32)))
    max_rounds = int(64)

    need = m
    for _round in range(max_rounds):
        if need <= 0:
            break

        k = int(min(batch, max(need * 64, need + 256)))
        i = rng.choice(G, size=k, replace=True, p=w).astype(np.int64, copy=False)

        j = rng.randint(0, G - 1, size=k).astype(np.int64, copy=False)
        j += (j >= i).astype(np.int64, copy=False)

        a = np.minimum(i, j)
        b = np.maximum(i, j)

        neq = (a != b)
        if not np.all(neq):
            a = a[neq]
            b = b[neq]
            if a.size == 0:
                continue

        idx = tri_idx(a, b)
        unseen = ~visited[idx]
        if not np.any(unseen):
            continue

        idx_u = idx[unseen]
        a_u = a[unseen]
        b_u = b[unseen]
        visited[idx_u] = True

        take = int(min(need, int(a_u.shape[0])))
        if take > 0:
            for t in range(take):
                ga = int(active[int(a_u[t])])
                gb = int(active[int(b_u[t])])
                edges.append((ga, gb) if ga < gb else (gb, ga))
            need -= take

    return edges


def _sample_edges(edges: List[Tuple[int, int]], rng: np.random.RandomState, m: int) -> List[Tuple[int, int]]:
    if m <= 0 or not edges:
        return []
    if len(edges) <= m:
        return edges
    idx = rng.choice(len(edges), size=int(m), replace=False)
    return [edges[int(i)] for i in idx]


def _propose_edges(
    *,
    active: List[int],
    gid_adj: Dict[int, set],
    gid_dims: Dict[int, Tuple[int, ...]],
    cfg: CrystallizationForwarderConfig,
    rng: np.random.RandomState,
    max_pool: int,
) -> List[Tuple[int, int]]:
    w_corr = float(max(0.0, cfg.proposal_weight_corr))
    w_long = float(max(0.0, cfg.proposal_weight_long))
    w_sum = w_corr + w_long
    if w_sum <= 0.0:
        return []

    max_pool = int(max(1, max_pool))

    q_corr = int(round(max_pool * (w_corr / w_sum))) if w_corr > 0 else 0
    q_long = int(round(max_pool * (w_long / w_sum))) if w_long > 0 else 0
    if q_corr + q_long != max_pool:
        # put drift to the larger side
        drift = max_pool - (q_corr + q_long)
        if w_corr >= w_long and w_corr > 0:
            q_corr = max(0, q_corr + drift)
        else:
            q_long = max(0, q_long + drift)

    edges_all: List[Tuple[int, int]] = []

    # corr: adj + two-hop
    if q_corr > 0:
        corr_ratio = float(cfg.corr_two_hop_ratio)
        corr_ratio = 0.0 if corr_ratio < 0.0 else (1.0 if corr_ratio > 1.0 else corr_ratio)
        q_two = int(round(q_corr * corr_ratio))
        q_adj = int(q_corr - q_two)
        q_adj = max(0, q_adj)
        q_two = max(0, q_two)

        adj_pool: List[Tuple[int, int]] = []
        for g in active:
            for h in gid_adj.get(g, set()):
                a, b = (g, h) if g < h else (h, g)
                if a != b:
                    adj_pool.append((a, b))
        if adj_pool:
            adj_pool = list(set(adj_pool))

        two_pool: List[Tuple[int, int]] = []
        if q_two > 0 and int(cfg.corr_two_hop_per_gid) > 0:
            two_pool = _two_hop_edges(active=active, gid_adj=gid_adj, rng=rng, per_gid=int(cfg.corr_two_hop_per_gid))
            if two_pool:
                two_pool = list(set(two_pool))

        edges_all += _sample_edges(adj_pool, rng, q_adj)
        edges_all += _sample_edges(two_pool, rng, q_two)

    # long
    if q_long > 0:
        m_gen = int(np.ceil(float(cfg.long_oversample) * float(q_long)))
        long_pool = _random_long_edges(active=active, gid_dims=gid_dims, rng=rng, m=m_gen, power=float(cfg.long_edge_power))
        edges_all += _sample_edges(long_pool, rng, q_long)

    # final unique + bound
    edges_all = list(set((a, b) if a < b else (b, a) for a, b in edges_all if a != b))
    if len(edges_all) > max_pool:
        idx = rng.choice(len(edges_all), size=max_pool, replace=False)
        edges_all = [edges_all[int(i)] for i in idx]
    return edges_all


# ============================================================
# Merge apply (incremental update)
# ============================================================


def _apply_merge_in_place(
    *,
    a: int,
    b: int,
    dz: Tuple[int, ...],
    gid_dims: Dict[int, Tuple[int, ...]],
    gid_adj: Dict[int, set],
    active: List[int],
    next_gid: int,
) -> int:
    z = int(next_gid)
    gid_dims[z] = dz

    na = gid_adj.get(a, set())
    nb = gid_adj.get(b, set())
    nz = set(na) | set(nb)
    nz.discard(a)
    nz.discard(b)

    if a in active:
        active.remove(a)
    if b in active:
        active.remove(b)

    for g in nz:
        if g in gid_adj:
            gid_adj[g].discard(a)
            gid_adj[g].discard(b)

    gid_adj.pop(a, None)
    gid_adj.pop(b, None)
    gid_dims.pop(a, None)
    gid_dims.pop(b, None)

    active.append(z)
    gid_adj[z] = set(g for g in nz if (g in gid_adj) or (g in active))
    for g in gid_adj[z]:
        gid_adj[g].add(z)

    return z


# ============================================================
# Forwarder
# ============================================================


class CrystallizationForwarder(ForwardingGrouper):
    """Crystallization forwarder: merge-only BEAM, no caps."""

    def __init__(self, cfg: CrystallizationForwarderConfig):
        self.cfg = cfg

    def forward_groups(self, ctx: EPQContext, *, groups: Groups, bits: Bits) -> Tuple[Groups, Bits]:
        cfg = self.cfg
        proxy = ctx.require_proxy()

        d = int(ctx.d)
        B = int(ctx.B)
        bmax = int(ctx.bmax)
        rng = np.random.RandomState(int(ctx.seed))

        if (float(cfg.proposal_weight_corr) + float(cfg.proposal_weight_long)) <= 0.0:
            raise RuntimeError("No proposal enabled: set proposal_weight_corr or proposal_weight_long > 0.")

        # --------------------------------------------------
        # Build initial gid state from input groups
        # --------------------------------------------------
        gid_dims: Dict[int, Tuple[int, ...]] = {}
        active: List[int] = []
        next_gid = 0
        for g in groups:
            gid_dims[next_gid] = _tuple_sorted(g)
            active.append(next_gid)
            next_gid += 1

        # --------------------------------------------------
        # Build initial adjacency (corr-adj)
        # --------------------------------------------------
        if float(cfg.proposal_weight_corr) > 0.0:
            dim_neigh = _build_dim_neighbors_by_corr(
                ctx.xt_train,
                knn=int(cfg.corr_adj_k),
                abs_corr=bool(cfg.corr_adj_abs),
                max_rows=int(cfg.corr_adj_rows),
                seed=int(ctx.seed),
            )
            gid_adj = _rebuild_gid_adj_from_dim_neigh(d=d, active=active, gid_dims=gid_dims, dim_neigh=dim_neigh)
        else:
            gid_adj = {gid: set() for gid in active}

        # --------------------------------------------------
        # Helper: solve bits for a state, returning (J, bits, gids_order)
        # IMPORTANT: bits are aligned with returned gids_order.
        # --------------------------------------------------
        def solve_state(active0: List[int], gid_dims0: Dict[int, Tuple[int, ...]]):
            gids0 = sorted(active0, key=lambda g: (len(gid_dims0[g]), gid_dims0[g][0]))
            groups0 = [list(gid_dims0[g]) for g in gids0]
            res = ctx.solve_bits(groups0)
            return float(res.J), list(res.bits), gids0

        # Initial exact evaluation (oracle)
        J0, bits0, gids0 = solve_state(active, gid_dims)

        root = {
            "active": list(active),
            "gid_dims": dict(gid_dims),
            "gid_adj": {k: set(v) for k, v in gid_adj.items()},
            "next_gid": int(next_gid),
            "J": float(J0),
            "bits": list(bits0),
            "gids": list(gids0),
        }

        best = root
        beam = [root]

        if cfg.verbose:
            print(
                f"[crystal] start: d={d} B={B} bmax={bmax} "
                f"M0={len(root['active'])} beam=(K={cfg.beam_width}, topR={cfg.beam_topR}) "
                f"proposal_w=(corr={cfg.proposal_weight_corr}, long={cfg.proposal_weight_long})"
            )

        # --------------------------------------------------
        # BEAM loop
        # --------------------------------------------------
        for depth in range(1, int(cfg.beam_max_depth) + 1):
            # Can't merge below ceil(B/bmax)
            min_groups = int((B + bmax - 1) // bmax)
            if len(best["active"]) <= min_groups or len(best["active"]) <= 1:
                break

            children_all: List[dict] = []

            for s in beam:
                active_s: List[int] = s["active"]
                if len(active_s) <= min_groups or len(active_s) <= 1:
                    continue

                gid_dims_s: Dict[int, Tuple[int, ...]] = s["gid_dims"]
                gid_adj_s: Dict[int, set] = s["gid_adj"]
                next_gid_s: int = s["next_gid"]
                bits_s: List[int] = s["bits"]
                gids_s: List[int] = s["gids"]

                gid2b = {g: int(bits_s[i]) for i, g in enumerate(gids_s)}

                # Precompute proxy D at b0 and at assigned bits (for structural filter)
                b0 = int(max(0, min(bmax, int(cfg.proxy_b0))))
                D_b0 = {g: float(proxy.D(list(gid_dims_s[g]), b0)) for g in active_s}
                D_assigned = {g: float(proxy.D(list(gid_dims_s[g]), int(gid2b.get(g, 0)))) for g in active_s}

                # Propose edges
                max_pool = int(max(256, int(cfg.pool_mult) * int(cfg.N_candidates)))
                edges = _propose_edges(
                    active=active_s,
                    gid_adj=gid_adj_s,
                    gid_dims=gid_dims_s,
                    cfg=cfg,
                    rng=rng,
                    max_pool=max_pool,
                )
                if not edges:
                    continue

                # dmax filter
                filtered: List[Tuple[int, int]] = []
                for a, b in edges:
                    if a in gid_dims_s and b in gid_dims_s and (len(gid_dims_s[a]) + len(gid_dims_s[b]) <= int(cfg.dmax)):
                        filtered.append((a, b))
                if not filtered:
                    continue

                # Proxy shortlist by ΔD(b0)
                proxies: List[Tuple[float, int, int, Tuple[int, ...]]] = []
                for a, b in filtered:
                    dz = _tuple_sorted(gid_dims_s[a] + gid_dims_s[b])
                    Dz = float(proxy.D(list(dz), b0))
                    proxies.append((float(Dz - D_b0[a] - D_b0[b]), a, b, dz))
                proxies.sort(key=lambda t: t[0])

                # Endpoint quota + shortlist size
                quota = int(max(1, cfg.endpoint_quota))
                used: Dict[int, int] = {g: 0 for g in active_s}

                L = int(max(1, int(cfg.N_candidates) * int(cfg.shortlist_factor)))
                cand_proxy: List[Tuple[float, int, int, Tuple[int, ...]]] = []
                for delta, a, b, dz in proxies:
                    if used.get(a, 0) >= quota or used.get(b, 0) >= quota:
                        continue
                    used[a] = used.get(a, 0) + 1
                    used[b] = used.get(b, 0) + 1
                    cand_proxy.append((delta, a, b, dz))
                    if len(cand_proxy) >= L:
                        break
                if not cand_proxy:
                    continue

                cand_proxy = cand_proxy[: min(len(cand_proxy), int(cfg.N_candidates))]

                # Structural filter at assigned bits: dJ_struct <= tol
                tol = float(cfg.dJ_struct_tol)
                struct_list: List[Tuple[float, int, int, Tuple[int, ...], int, int]] = []
                for _delta, a, b, dz in cand_proxy:
                    bx, by = int(gid2b.get(a, 0)), int(gid2b.get(b, 0))
                    Dx = float(D_assigned[a])
                    Dy = float(D_assigned[b])
                    bz = int(min(bmax, bx + by))
                    Dz = float(proxy.D(list(dz), bz))
                    dJ_struct = float(Dz - Dx - Dy)
                    if dJ_struct <= tol:
                        struct_list.append((dJ_struct, a, b, dz, bx, by))

                if not struct_list:
                    continue

                struct_list.sort(key=lambda t: t[0])
                local_top = struct_list[: min(len(struct_list), int(cfg.beam_topR))]

                for _dJ, a, b, dz, bx, by in local_top:
                    child_active = list(active_s)
                    child_gid_dims = dict(gid_dims_s)
                    child_gid_adj = {k: set(v) for k, v in gid_adj_s.items()}
                    child_next_gid = int(next_gid_s)

                    z = _apply_merge_in_place(
                        a=a,
                        b=b,
                        dz=dz,
                        gid_dims=child_gid_dims,
                        gid_adj=child_gid_adj,
                        active=child_active,
                        next_gid=child_next_gid,
                    )
                    child_next_gid += 1

                    J_child, bits_child, gids_child = solve_state(child_active, child_gid_dims)

                    children_all.append(
                        {
                            "active": child_active,
                            "gid_dims": child_gid_dims,
                            "gid_adj": child_gid_adj,
                            "next_gid": child_next_gid,
                            "J": float(J_child),
                            "bits": list(bits_child),
                            "gids": list(gids_child),
                        }
                    )

            if not children_all:
                # No candidate passed the structural filter at this step.
                # With the fixed filter, this should be rare unless no proposals were generated.
                if cfg.verbose:
                    print("[crystal] no children; stop.")
                break

            # Dedup by canonical partition key
            best_by_key: Dict[Tuple[Tuple[int, ...], ...], dict] = {}
            for c in children_all:
                k = _canonical_partition_key(c["active"], c["gid_dims"])
                prev = best_by_key.get(k)
                if prev is None or float(c["J"]) < float(prev["J"]):
                    best_by_key[k] = c
            uniq_children = list(best_by_key.values())

            uniq_children.sort(key=lambda ss: float(ss["J"]))
            beam = uniq_children[: int(max(1, cfg.beam_width))]

            # Stop if no improvement
            if float(beam[0]["J"]) < float(best["J"]) - 1e-12:
                best = beam[0]
            else:
                break

            if cfg.verbose:
                print(
                    f"[crystal] depth={depth} beam={len(beam)} "
                    f"bestJ={float(best['J']):.6f} groups={len(best['active'])}"
                )

        # Final output: stable ordering aligned with best["gids"]/best["bits"]
        final_gids = list(best["gids"])
        final_bits = list(best["bits"])
        final_groups = [sorted(list(best["gid_dims"][g])) for g in final_gids]

        # Sanity: ensure partition size matches d
        flat = [i for gg in final_groups for i in gg]
        if len(flat) != d or len(set(flat)) != d:
            raise RuntimeError("Final groups are not a valid partition.")

        return final_groups, final_bits
