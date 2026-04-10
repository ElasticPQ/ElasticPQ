from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Literal

import numpy as np

from grouper import Grouper, EPQContext


# ============================================================
# Config
# ============================================================

@dataclass
class ClusterGrowGrouperConfig:
    # --- how many groups ---
    alpha_M: float = 2.0          # M0 ≈ alpha * (B//8)
    min_M: int = 16               # auto-selected M0 should not fall below this floor
    target_M: int = 0             # if >0, overrides alpha rule
    max_M: int = 0                # if >0, clamp M0 <= max_M

    # --- corr graph (candidate generator / votes) ---
    corr_adj_k: int = 16
    corr_adj_abs: bool = True
    corr_adj_rows: int = 4096     # subsample rows to estimate corr
    edge_tau: float = 0.0         # keep edges with weight>=tau (0 disables)

    # --- growth control ---
    dmax: int = 1024              # hard cap per group size
    min_group_size: int = 1       # usually keep 1; can set 2 to avoid singleton seeds
    min_votes: int = 2            # anti-bridge: need >= votes from current cluster
    avg_gain_tau: float = 0.0     # anti-bridge: gain/|C| >= tau
    fill_when_stuck: bool = True  # if constraints block growth, fill by best-effort

    # --- "D-based" scoring policy ---
    score_bits_mode: Literal["fixed", "size_proportional"] = "fixed"
    score_bits_fixed: int = 4                 # used when score_bits_mode="fixed"
    score_bits_min: int = 0                   # clamp if size_proportional
    score_bits_max: int = 12                  # clamp if size_proportional (should match proxy.bmax)

    # --- rerank control (two-stage selection) ---
    # Use corr-derived frontier weights to shortlist, then rerank by true ΔD via proxy.D.
    rerank_L: int = 32                       # evaluate true ΔD for top-L candidates per group step

    # --- seed policy ---
    seed_mode: Literal["hardest", "easiest", "corr"] = "hardest"
    seed_topk: int = 16                       # random tie-break within top-k for diversity
    seed_pair: bool = True                    # try to form 2-dim seed using best ΔD


# ============================================================
# Corr-neighborhood builder (candidate generator)
# ============================================================

def _subsample_rows(x: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or x.shape[0] <= max_rows:
        return x
    rs = np.random.RandomState(int(seed))
    idx = rs.choice(x.shape[0], size=int(max_rows), replace=False)
    return x[idx]


def _build_dim_neighbors_by_corr(
    xt: np.ndarray,
    *,
    knn: int,
    abs_corr: bool,
    max_rows: int,
    seed: int,
    edge_tau: float,
) -> List[List[Tuple[int, float]]]:
    """
    neigh[i] = list of (j, w_ij) sorted by decreasing w_ij.
    w_ij = abs(corr) if abs_corr else corr
    edge_tau filters by w_ij >= tau (if tau>0).
    """
    x = xt
    if max_rows > 0 and x.shape[0] > max_rows:
        x = _subsample_rows(x, max_rows=int(max_rows), seed=int(seed))

    x = np.ascontiguousarray(x, dtype=np.float32)
    n, d = x.shape
    if d <= 1:
        return [[] for _ in range(d)]
    kk = min(int(knn), max(1, d - 1))

    # low-row fallback: near-index neighbors with weight 1.0
    if n <= 1:
        out: List[List[Tuple[int, float]]] = [[] for _ in range(d)]
        half = max(1, kk // 2)
        for i in range(d):
            for j in range(max(0, i - half), min(d, i + half + 1)):
                if j != i:
                    out[i].append((j, 1.0))
        return out

    # normalize columns
    x = x - x.mean(axis=0, keepdims=True)
    std = np.maximum(x.std(axis=0, ddof=0, keepdims=True), 1e-6)
    x = x / std

    corr = (x.T @ x) / float(n)
    corr = np.clip(corr, -1.0, 1.0)
    score = np.abs(corr) if bool(abs_corr) else corr
    np.fill_diagonal(score, -np.inf)

    out2: List[List[Tuple[int, float]]] = [[] for _ in range(d)]
    tau = float(max(0.0, edge_tau))

    for i in range(d):
        idx = np.argpartition(-score[i], kth=kk - 1)[:kk]
        idx = idx[np.argsort(-score[i, idx])]
        lst: List[Tuple[int, float]] = []
        for j in idx:
            w = float(score[i, j])
            if not np.isfinite(w):
                continue
            if tau > 0.0 and w < tau:
                continue
            lst.append((int(j), w))
        out2[i] = lst
    return out2


def _min_groups_for_feasible_bits(B: int, bmax: int) -> int:
    B = int(B)
    bmax = int(bmax)
    if B <= 0:
        return 1
    if bmax <= 0:
        raise ValueError(f"Infeasible bit budget: B={B} requires bmax>0, got bmax={bmax}")
    return int((B + bmax - 1) // bmax)


# ============================================================
# Grouper
# ============================================================

class ClusterGrowGrouper(Grouper):
    """
    Stage-1 coarse grouper: build groups by growing clusters guided by proxy D().

    High-level:
      1) Build corr-neighborhood graph (for candidate generation + votes).
      2) Seed M0 tiny groups (1-2 dims), using D(singleton) or corr degree.
      3) Globally assign remaining dims by maximizing true ΔD (reranked from corr frontier).
      4) Final bits: run ctx.solve_bits(groups) under ctx.bmax.
    """

    def __init__(self, cfg: ClusterGrowGrouperConfig):
        self.cfg = cfg

    # -------- scoring bits for D(·, b) --------

    def _score_bits_for_group(self, *, size: int, d: int, B: int, proxy_bmax: int) -> int:
        cfg = self.cfg
        if cfg.score_bits_mode == "fixed":
            b = int(cfg.score_bits_fixed)
        else:
            # size-proportional estimate
            if d <= 0:
                raw = float(B)
            else:
                raw = float(B) * (float(size) / float(d))
            b = int(round(raw))
            b = max(int(cfg.score_bits_min), b)
            b = min(int(cfg.score_bits_max), b)

        # clamp into proxy range
        b = max(0, min(int(proxy_bmax), int(b)))
        return int(b)

    # -------- seed selection --------

    def _compute_seed_scores(
        self,
        *,
        d: int,
        unassigned: np.ndarray,
        adj: List[List[Tuple[int, float]]],
        proxy,
        b_seed: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        cfg = self.cfg
        scores = np.full(d, -np.inf, dtype=np.float64)
        idx = np.where(unassigned)[0]
        if idx.size == 0:
            return scores

        if cfg.seed_mode == "corr":
            # total incident weight
            for i in idx:
                if adj[i]:
                    scores[i] = float(sum(w for _j, w in adj[i]))
                else:
                    scores[i] = 0.0
            return scores

        # D-based seed score: "hardest" => highest D([i], b_seed); "easiest" => lowest
        # We convert into "higher-is-better" score.
        for i in idx:
            Di = float(proxy.D([int(i)], int(b_seed)))
            if cfg.seed_mode == "hardest":
                scores[i] = Di
            else:
                scores[i] = -Di
        return scores

    def _pick_seed(
        self,
        *,
        scores: np.ndarray,
        unassigned: np.ndarray,
        rng: np.random.RandomState,
    ) -> int:
        cfg = self.cfg
        idx = np.where(unassigned)[0]
        if idx.size == 0:
            return -1
        s = scores[idx]
        # take top-k for diversity
        k = int(max(1, cfg.seed_topk))
        if idx.size <= k:
            cand = idx
        else:
            # partial select
            kth = min(k - 1, idx.size - 1)
            topk_pos = np.argpartition(-s, kth=kth)[:k]
            cand = idx[topk_pos]
        # choose best among cand, tie-break randomly
        best_val = float(scores[cand].max())
        best = cand[np.where(scores[cand] >= best_val - 1e-12)[0]]
        return int(rng.choice(best)) if best.size > 1 else int(best[0])

    def _try_seed_pair(
        self,
        *,
        seed: int,
        unassigned: np.ndarray,
        adj: List[List[Tuple[int, float]]],
        proxy,
        b_ref: int,
        rng: np.random.RandomState,
    ) -> Optional[int]:
        """Return a second dim to pair with seed if beneficial; else None."""
        cfg = self.cfg
        if not cfg.seed_pair:
            return None
        # candidates from corr neighbors first
        cand = [j for (j, _w) in adj[seed] if unassigned[j]]
        if not cand:
            return None

        # evaluate best ΔD from adding j
        D0 = float(proxy.D([seed], int(b_ref)))
        best_j = None
        best_gain = 0.0

        # small shuffle helps reduce deterministic traps when gains are close
        rng.shuffle(cand)
        for j in cand[: max(8, cfg.rerank_L)]:  # keep it bounded
            D1 = float(proxy.D([seed, int(j)], int(b_ref)))
            gain = D0 - D1
            if gain > best_gain:
                best_gain = gain
                best_j = int(j)

        # require positive gain (strictly better)
        return best_j if (best_j is not None and best_gain > 0.0) else None

    # -------- frontier / group state --------

    @dataclass
    class _GroupState:
        dims: List[int]
        # frontier candidate -> corr-weight sum + votes
        wsum: Dict[int, float]
        votes: Dict[int, int]
        # cached D(dims, b_ref) for current group
        Dcur: float
        b_ref: int

    def _init_group_state(
        self,
        *,
        dims: List[int],
        unassigned: np.ndarray,
        adj: List[List[Tuple[int, float]]],
        proxy,
        b_ref: int,
    ) -> "_GroupState":
        wsum: Dict[int, float] = {}
        votes: Dict[int, int] = {}
        for v in dims:
            for u, w in adj[v]:
                if not unassigned[u]:
                    continue
                wsum[u] = wsum.get(u, 0.0) + float(w)
                votes[u] = votes.get(u, 0) + 1
        Dcur = float(proxy.D(list(dims), int(b_ref)))
        return ClusterGrowGrouper._GroupState(
            dims=list(dims),
            wsum=wsum,
            votes=votes,
            Dcur=Dcur,
            b_ref=int(b_ref),
        )

    def _update_frontier_add_dim(
        self,
        st: "_GroupState",
        *,
        v: int,
        unassigned: np.ndarray,
        adj: List[List[Tuple[int, float]]],
    ) -> None:
        # remove v if it was a candidate
        st.wsum.pop(v, None)
        st.votes.pop(v, None)

        for u, w in adj[v]:
            if not unassigned[u]:
                continue
            st.wsum[u] = st.wsum.get(u, 0.0) + float(w)
            st.votes[u] = st.votes.get(u, 0) + 1

    def _pick_best_candidate_for_group(
        self,
        st: "_GroupState",
        *,
        unassigned: np.ndarray,
        proxy,
        d: int,
        B: int,
        proxy_bmax: int,
        dmax: int,
        rng: np.random.RandomState,
    ) -> Optional[int]:
        cfg = self.cfg
        if len(st.dims) >= dmax:
            return None
        if not st.wsum:
            return None

        # apply constraints
        min_votes = int(max(0, cfg.min_votes))
        avg_tau = float(max(0.0, cfg.avg_gain_tau))

        # Stage 1: shortlist by corr frontier weight (cheap)
        items = []
        for u, w in st.wsum.items():
            if not unassigned[u]:
                continue
            vcnt = st.votes.get(u, 0)
            if min_votes > 0 and vcnt < min_votes:
                continue
            items.append((int(u), float(w), int(vcnt)))

        # if stuck under constraints, relax (if allowed)
        if not items and cfg.fill_when_stuck:
            # relax avg_tau first (we haven't applied it yet), but keep votes
            for u, w in st.wsum.items():
                if not unassigned[u]:
                    continue
                vcnt = st.votes.get(u, 0)
                if min_votes > 0 and vcnt < min_votes:
                    continue
                items.append((int(u), float(w), int(vcnt)))

        if not items and cfg.fill_when_stuck:
            # relax votes too
            for u, w in st.wsum.items():
                if not unassigned[u]:
                    continue
                items.append((int(u), float(w), int(st.votes.get(u, 0))))

        if not items:
            return None

        # take top-L by wsum, with small randomness for diversity
        items.sort(key=lambda t: t[1], reverse=True)
        L = int(max(1, cfg.rerank_L))
        cand = [u for (u, _w, _vcnt) in items[: min(L, len(items))]]

        # Stage 2: rerank by true ΔD via proxy.D
        # Use a (possibly updated) b_ref policy; we keep group-local st.b_ref fixed to avoid thrash.
        b_ref = int(st.b_ref)
        D0 = float(st.Dcur)

        best_u = None
        best_gain = -1e100

        # random shuffle among top-L helps avoid deterministic dead-ends when gains are similar
        rng.shuffle(cand)

        for u in cand:
            if not unassigned[u]:
                continue
            dims_new = st.dims + [int(u)]
            D1 = float(proxy.D(dims_new, int(b_ref)))
            gain = D0 - D1  # positive is good

            # anti-bridge: average gain threshold (apply here, based on true gain)
            if avg_tau > 0.0:
                denom = float(max(1, len(st.dims)))
                if float(gain) / denom < avg_tau and cfg.fill_when_stuck is False:
                    continue
                # if fill_when_stuck, don't hard-block; just let it compete

            if gain > best_gain:
                best_gain = float(gain)
                best_u = int(u)

        if best_u is None and cfg.fill_when_stuck:
            # ultimate fallback: pick a valid unassigned candidate (max wsum)
            for u, w, _vcnt in items:
                if unassigned[u]:
                    best_u = int(u)
                    break

        return best_u

    # -------- main build_groups --------

    def build_groups(self, ctx: EPQContext) -> Tuple[List[List[int]], List[int]]:
        cfg = self.cfg
        d = int(ctx.d)
        B = int(ctx.B)
        if d <= 0:
            raise ValueError("d must be positive")
        if B < 0:
            raise ValueError("B must be non-negative")

        proxy = ctx.require_proxy()
        proxy_bmax = int(getattr(proxy, "bmax", 12))

        rng = np.random.RandomState(int(ctx.seed))

        # choose M0
        if int(cfg.target_M) > 0:
            M0 = int(cfg.target_M)
        else:
            Mpq = max(1, B // 8) if B > 0 else 1
            M0 = int(max(1, round(float(cfg.alpha_M) * float(Mpq))))
            M0 = max(M0, int(cfg.min_M))
        if int(cfg.max_M) > 0:
            M0 = min(M0, int(cfg.max_M))
        M0 = max(1, min(M0, d))

        dmax = int(max(1, cfg.dmax))
        min_groups_by_dmax = int((d + dmax - 1) // dmax)
        min_groups_by_bits = _min_groups_for_feasible_bits(B=B, bmax=proxy_bmax)
        if min_groups_by_bits > d:
            raise ValueError(
                f"Infeasible: need at least ceil(B/bmax)={min_groups_by_bits} groups "
                f"to allocate B={B} with bmax={proxy_bmax}, but d={d} allows at most {d} non-empty groups."
            )
        M0 = max(M0, min_groups_by_dmax)
        M0 = max(M0, min_groups_by_bits)
        M0 = min(M0, d)

        # Build corr neighbor graph from xt_train (more stable than full x)
        xt = np.ascontiguousarray(ctx.xt_train, dtype=np.float32)
        if xt.shape[1] != d:
            # fallback to ctx.x if needed
            xt = np.ascontiguousarray(ctx.x, dtype=np.float32)
            if xt.shape[1] != d:
                raise ValueError(f"dimension mismatch: d={d} xt.shape[1]={xt.shape[1]}")

        neigh = _build_dim_neighbors_by_corr(
            xt,
            knn=int(cfg.corr_adj_k),
            abs_corr=bool(cfg.corr_adj_abs),
            max_rows=int(cfg.corr_adj_rows),
            seed=int(ctx.seed),
            edge_tau=float(cfg.edge_tau),
        )

        # Symmetric adjacency
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(d)]
        for i in range(d):
            for j, w in neigh[i]:
                adj[i].append((int(j), float(w)))
                adj[int(j)].append((int(i), float(w)))

        unassigned = np.ones(d, dtype=np.bool_)

        # Seed bits for singleton scoring (seed stage)
        b_seed = self._score_bits_for_group(size=1, d=d, B=B, proxy_bmax=proxy_bmax)

        # Phase S: create M0 tiny seed groups
        groups: List[List[int]] = []
        for _ in range(M0):
            if not np.any(unassigned):
                break

            scores = self._compute_seed_scores(
                d=d,
                unassigned=unassigned,
                adj=adj,
                proxy=proxy,
                b_seed=int(b_seed),
                rng=rng,
            )
            seed = self._pick_seed(scores=scores, unassigned=unassigned, rng=rng)
            if seed < 0:
                break

            # claim seed
            unassigned[seed] = False
            g = [int(seed)]

            # optional pair
            # For seed-pair scoring, use b_ref for size=1 (or size=2); keep it simple at b_seed
            pair = self._try_seed_pair(
                seed=int(seed),
                unassigned=unassigned,
                adj=adj,
                proxy=proxy,
                b_ref=int(b_seed),
                rng=rng,
            )
            if pair is not None and unassigned[int(pair)]:
                unassigned[int(pair)] = False
                g.append(int(pair))

            # enforce min_group_size by stealing random dims (only during seeding)
            if int(cfg.min_group_size) > 1 and len(g) < int(cfg.min_group_size):
                need = int(cfg.min_group_size) - len(g)
                cand = np.where(unassigned)[0]
                if cand.size > 0:
                    take = min(need, int(cand.size))
                    extra = rng.choice(cand, size=take, replace=False).tolist()
                    for x in extra:
                        unassigned[int(x)] = False
                        g.append(int(x))

            groups.append(g)

        # If we somehow created fewer than M0 groups but still have unassigned, add singleton groups
        while len(groups) < M0 and np.any(unassigned):
            i = int(np.where(unassigned)[0][0])
            unassigned[i] = False
            groups.append([i])

        # Prepare group states
        states: List[ClusterGrowGrouper._GroupState] = []
        for g in groups:
            b_ref = self._score_bits_for_group(size=len(g), d=d, B=B, proxy_bmax=proxy_bmax)
            st = self._init_group_state(dims=g, unassigned=unassigned, adj=adj, proxy=proxy, b_ref=b_ref)
            states.append(st)

        # Phase G: global assignment until all dims assigned
        # Each step: pick the single best move among groups (by true ΔD rerank within each group).
        while np.any(unassigned):
            best_move = None  # (gain, group_idx, dim_u)
            best_gain = -1e100

            for gi, st in enumerate(states):
                u = self._pick_best_candidate_for_group(
                    st,
                    unassigned=unassigned,
                    proxy=proxy,
                    d=d,
                    B=B,
                    proxy_bmax=proxy_bmax,
                    dmax=dmax,
                    rng=rng,
                )
                if u is None:
                    continue

                # compute true gain for comparing across groups
                D0 = float(st.Dcur)
                D1 = float(proxy.D(st.dims + [int(u)], int(st.b_ref)))
                gain = D0 - D1

                if gain > best_gain:
                    best_gain = float(gain)
                    best_move = (gi, int(u), float(D1))

            if best_move is None:
                # No group has a frontier candidate. Fallback: dump remaining dims into not-full groups.
                dims_left = np.where(unassigned)[0].tolist()
                rng.shuffle(dims_left)

                # try to place into existing groups while respecting dmax
                ptr = 0
                for u in dims_left:
                    placed = False
                    for gi, st in enumerate(states):
                        if len(st.dims) < dmax:
                            # assign u to this group
                            unassigned[u] = False
                            st.dims.append(int(u))
                            # update frontier from u (it was unassigned before we flipped it)
                            self._update_frontier_add_dim(st, v=int(u), unassigned=unassigned, adj=adj)
                            # update Dcur
                            st.Dcur = float(proxy.D(st.dims, int(st.b_ref)))
                            placed = True
                            break
                    if not placed:
                        # need a new group due to dmax hard constraint
                        unassigned[u] = False
                        b_ref = self._score_bits_for_group(size=1, d=d, B=B, proxy_bmax=proxy_bmax)
                        st_new = self._init_group_state(dims=[int(u)], unassigned=unassigned, adj=adj, proxy=proxy, b_ref=b_ref)
                        states.append(st_new)
                        groups.append([int(u)])
                    ptr += 1
                break

            gi, u, D1 = best_move
            st = states[gi]

            if not unassigned[u]:
                # stale candidate; remove and continue
                st.wsum.pop(u, None)
                st.votes.pop(u, None)
                continue

            # assign u
            unassigned[u] = False
            st.dims.append(int(u))
            self._update_frontier_add_dim(st, v=int(u), unassigned=unassigned, adj=adj)

            # Keep b_ref stable for this group to avoid thrashing.
            # Recomputing b_ref here would also require refreshing Dcur and would
            # substantially increase proxy-call volume.
            st.Dcur = float(D1)

        # Materialize groups from states
        groups = [list(st.dims) for st in states]

        # sanity partition check
        flat = [x for gg in groups for x in gg]
        if len(flat) != d or len(set(flat)) != d or set(flat) != set(range(d)):
            raise RuntimeError("ClusterGrowGrouper produced invalid partition")

        res = ctx.solve_bits(groups)
        return groups, list(res.bits)
