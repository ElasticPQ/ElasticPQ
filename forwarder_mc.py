from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Deque

import hashlib
from collections import deque

import numpy as np

from grouper import EPQContext, Groups, Bits, ForwardingGrouper


# ============================================================
# Config
# ============================================================

@dataclass
class MarginalBeamForwarderConfig:
    # iterations / early stop
    iters: int = 1000
    patience: int = 30                 # stop if no globalBest improvement for this many iters
    eps_improve: float = 0.0           # consider improved if J < best - eps

    # beam
    beam_w: int = 4                    # number of states kept per iter
    per_state_eval_topk: int = 6       # DP-evaluate top-K moves per state (after score-shortlist)
    per_state_shortlist_k: int = 24    # keep best K moves by score per state before DP eval

    # group shortlist (used for candidate generation)
    donor_topk: int = 10               # "fat" groups shortlist
    recv_topk: int = 10                # "hungry" groups shortlist

    # choose dims inside a group (suspiciousness)
    dims_sample_per_group: int = 10
    suspicious_alpha: float = 1.0      # >0 biases sampling toward higher harm(v)=D(g,b)-D(g\\{v},b)

    # move proposal counts (PER STATE, PER ITER)
    n_relocate: int = 128              # relocate proposals
    n_swap_pairs: int = 48             # swap proposals

    # scoring
    # score = dJ_struct - lambda * gml
    dp_shift_lambda: float = 1.0

    # seen-best window: only remember last W iters (stabilizes across datasets)
    seen_window: int = 5               # keep only recent W rounds of seen keys

    # randomness
    seed: int = 123

    # verbosity
    verbose: int = 1


# ============================================================
# Helpers
# ============================================================

def _validate_partition(groups: Groups, d: int) -> None:
    if not groups:
        raise ValueError("partition is empty")
    if any(len(g) == 0 for g in groups):
        raise ValueError("partition contains empty group")
    flat = [int(x) for g in groups for x in g]
    if len(flat) != int(d):
        raise ValueError(f"partition must cover exactly d={int(d)} dims, got {len(flat)}")
    if len(set(flat)) != len(flat):
        raise ValueError("partition contains duplicated dims")
    want = set(range(int(d)))
    got = set(flat)
    if got != want:
        missing = sorted(want - got)
        extra = sorted(got - want)
        raise ValueError(f"partition has invalid dim ids: missing={missing} extra={extra}")


def _deepcopy_groups(groups: Groups) -> Groups:
    return [list(map(int, g)) for g in groups]


def _proxy_D(ctx: EPQContext, dims: List[int], b: int) -> float:
    proxy = ctx.require_proxy()
    return float(proxy.D(dims, int(b)))


def _marginal_gain(ctx: EPQContext, dims: List[int], b: int) -> float:
    # gain(b) = D(b) - D(b+1)
    b = int(b)
    if b >= int(ctx.bmax):
        return 0.0
    return _proxy_D(ctx, dims, b) - _proxy_D(ctx, dims, b + 1)


def _marginal_loss(ctx: EPQContext, dims: List[int], b: int) -> float:
    # loss(b) = D(b-1) - D(b)
    b = int(b)
    if b <= 0:
        return float("inf")
    return _proxy_D(ctx, dims, b - 1) - _proxy_D(ctx, dims, b)


def _remove_one(g: List[int], v: int) -> List[int]:
    v = int(v)
    out: List[int] = []
    removed = False
    for x in g:
        xx = int(x)
        if (not removed) and xx == v:
            removed = True
            continue
        out.append(xx)
    return out


def _choose_suspicious_dim(
    ctx: EPQContext,
    g: List[int],
    b: int,
    *,
    rng: np.random.RandomState,
    sample_k: int,
    alpha: float,
) -> int:
    """Pick a dim in g, biased toward higher harm(v)=D(g,b)-D(g\\{v},b)."""
    if len(g) == 1:
        return int(g[0])

    k = min(int(sample_k), len(g))
    idx = rng.choice(len(g), size=k, replace=False)
    cand = [int(g[int(i)]) for i in idx]

    D0 = _proxy_D(ctx, g, int(b))

    harms = np.zeros(len(cand), dtype=np.float64)
    for i, v in enumerate(cand):
        g_minus = _remove_one(g, v)
        harms[i] = float(D0 - _proxy_D(ctx, g_minus, int(b)))

    if float(alpha) <= 0.0:
        return int(cand[int(rng.randint(0, len(cand)))])

    mn = float(harms.min())
    hs = harms - mn + 1e-12
    hs = np.power(hs, float(alpha))
    ps = hs / float(hs.sum())
    pick = int(rng.choice(len(cand), p=ps))
    return int(cand[pick])


def _key_digest(ctx: EPQContext, groups: Groups) -> Tuple[int, int]:
    """
    Collision-resistant key compression.

    We hash the canonical partition_key(groups) with TWO independent digests:
      - blake2b (128-bit)
      - blake2s (64-bit)

    Key = (h128_as_int, h64_as_int). Collision probability is astronomically low.
    """
    canon = ctx.partition_key(groups)  # tuple-of-tuples, stable canonical form
    b = repr(canon).encode("utf-8", "strict")
    h128 = int.from_bytes(hashlib.blake2b(b, digest_size=16).digest(), "big", signed=False)
    h64 = int.from_bytes(hashlib.blake2s(b, digest_size=8).digest(), "big", signed=False)
    return (h128, h64)


# ============================================================
# Moves
# ============================================================

@dataclass
class _Move:
    kind: str  # "relocate" | "swap"
    dims: Tuple[int, ...]      # relocate:(v,) swap:(v,u)
    groups: Tuple[int, ...]    # relocate:(A,B) swap:(A,B) (indices in CURRENT partition)
    dJ_struct: float
    gml: float
    score: float


def _apply_relocate(groups: Groups, A: int, B: int, v: int) -> Groups:
    out = _deepcopy_groups(groups)
    gA = out[int(A)]
    gB = out[int(B)]
    gA2 = _remove_one(gA, int(v))
    if len(gA2) == 0:
        raise ValueError("relocate would delete donor group")
    out[int(A)] = gA2
    out[int(B)] = list(map(int, gB)) + [int(v)]
    return out


def _apply_swap(groups: Groups, A: int, B: int, v: int, u: int) -> Groups:
    out = _deepcopy_groups(groups)
    gA = out[int(A)]
    gB = out[int(B)]
    out[int(A)] = _remove_one(gA, int(v)) + [int(u)]
    out[int(B)] = _remove_one(gB, int(u)) + [int(v)]
    return out


# ============================================================
# Beam state
# ============================================================

@dataclass
class _State:
    groups: Groups
    bits: Bits
    J: float


# ============================================================
# Seen-window (last W rounds)
# ============================================================

class _SeenWindow:
    """
    Keep only last W rounds of {key -> bestJ_in_that_round}, and expose:
      - get_best(key): best J within window, or None
      - set_best(key, J): register in current round
      - next_round(): advance window, drop oldest, recompute minima only when needed

    Windowed memory is more stable across datasets than fixed-capacity LRU.
    """

    def __init__(self, W: int):
        self.W = max(1, int(W))
        self.rounds: Deque[Dict[Tuple[int, int], float]] = deque()
        self.best: Dict[Tuple[int, int], float] = {}   # min J within current window
        self.cur: Dict[Tuple[int, int], float] = {}
        self.rounds.append(self.cur)

    def get_best(self, key: Tuple[int, int]) -> Optional[float]:
        return self.best.get(key)

    def set_best(self, key: Tuple[int, int], J: float) -> None:
        # current round: keep best
        old = self.cur.get(key)
        if old is None or float(J) < float(old):
            self.cur[key] = float(J)

        # window best
        oldb = self.best.get(key)
        if oldb is None or float(J) < float(oldb):
            self.best[key] = float(J)

    def next_round(self) -> None:
        # push a new current round dict
        self.cur = {}
        self.rounds.append(self.cur)

        # pop if exceeds W
        while len(self.rounds) > self.W:
            old_round = self.rounds.popleft()
            if not old_round:
                continue

            # For each key in popped round: if it was responsible for the window-min,
            # recompute min across remaining rounds (W is small, O(W) scan is fine).
            for k, j_old in old_round.items():
                j_best = self.best.get(k)
                if j_best is None:
                    continue
                if float(j_old) != float(j_best):
                    continue

                # recompute min across remaining rounds
                new_best = None
                for rd in self.rounds:
                    jj = rd.get(k)
                    if jj is None:
                        continue
                    if new_best is None or float(jj) < float(new_best):
                        new_best = float(jj)
                if new_best is None:
                    self.best.pop(k, None)
                else:
                    self.best[k] = float(new_best)


# ============================================================
# Forwarder (beam)
# ============================================================

class MarginalBeamForwarder(ForwardingGrouper):
    """
    Beam version: relocate (main) + swap,
    IMPORTANT OPT-1:
      - DO NOT re-solve bits for beam states
      - solve_bits ONLY for newly generated children
    """

    def __init__(self, cfg: MarginalBeamForwarderConfig = MarginalBeamForwarderConfig()):
        self.cfg = cfg

    def forward_groups(self, ctx: EPQContext, *, groups: Groups, bits: Bits):
        d = int(ctx.d)
        rng = np.random.RandomState(int(self.cfg.seed))
        lam = float(self.cfg.dp_shift_lambda)

        # ---- init ----
        g0 = _deepcopy_groups(groups)
        _validate_partition(g0, d=d)

        alloc0 = ctx.solve_bits(g0)
        b0 = [int(b) for b in alloc0.bits]
        J0 = float(alloc0.J)

        beam: List[_State] = [_State(groups=g0, bits=b0, J=J0)]

        global_best = J0
        global_best_groups = _deepcopy_groups(g0)
        global_best_bits = list(b0)

        seen = _SeenWindow(int(self.cfg.seen_window))
        seen.set_best(_key_digest(ctx, g0), J0)

        if self.cfg.verbose:
            print(f"[mcb] init J={J0:.6f} M={len(g0)} sumB={sum(b0)} beam=1")

        no_improve = 0

        # ============================================================
        # main loop
        # ============================================================
        for it in range(int(self.cfg.iters)):
            if self.cfg.patience > 0 and no_improve >= self.cfg.patience:
                if self.cfg.verbose:
                    print(
                        f"[mcb] early_stop it={it:03d} "
                        f"no_improve={no_improve} bestJ={global_best:.6f}"
                    )
                break

            seen.next_round()

            children: List[_State] = []
            dp_evals = 0
            proposed = 0

            # ------------------------------------------------------------
            # expand each beam state
            # ------------------------------------------------------------
            for st in beam:
                # >>> OPT-1: reuse st.bits / st.J, no solve_bits here <<<
                cur_groups = st.groups
                cur_bits = st.bits
                cur_J = st.J

                M = len(cur_groups)
                if M <= 1:
                    continue

                # ---- compute marginal gains once ----
                gain_now = np.zeros(M, dtype=np.float64)
                for i in range(M):
                    gain_now[i] = _marginal_gain(ctx, cur_groups[i], cur_bits[i])

                fat_score = np.array(
                    [
                        float(cur_bits[i]) / max(1e-12, gain_now[i] + 1e-12)
                        for i in range(M)
                    ],
                    dtype=np.float64,
                )
                hungry_score = gain_now

                donor_pool = np.argsort(-fat_score)[: min(self.cfg.donor_topk, M)]
                recv_pool = np.argsort(-hungry_score)[: min(self.cfg.recv_topk, M)]

                # cache D(g, b_g)
                D_before = [
                    float(_proxy_D(ctx, cur_groups[i], cur_bits[i]))
                    for i in range(M)
                ]

                moves: List[_Move] = []

                # ---------------- relocate ----------------
                for _ in range(self.cfg.n_relocate):
                    A = int(donor_pool[rng.randint(len(donor_pool))])
                    B = int(recv_pool[rng.randint(len(recv_pool))])
                    if A == B:
                        continue
                    gA = cur_groups[A]
                    gB = cur_groups[B]
                    if len(gA) <= 1:
                        continue

                    v = _choose_suspicious_dim(
                        ctx, gA, cur_bits[A],
                        rng=rng,
                        sample_k=self.cfg.dims_sample_per_group,
                        alpha=self.cfg.suspicious_alpha,
                    )

                    gA2 = _remove_one(gA, v)
                    if not gA2:
                        continue
                    gB2 = gB + [v]

                    dJ_struct = (
                        _proxy_D(ctx, gA2, cur_bits[A])
                        + _proxy_D(ctx, gB2, cur_bits[B])
                        - D_before[A]
                        - D_before[B]
                    )

                    lossA = _marginal_loss(ctx, gA2, cur_bits[A])
                    gainB = _marginal_gain(ctx, gB2, cur_bits[B])
                    gml = 0.0 if not np.isfinite(lossA) else max(0.0, gainB - lossA)

                    moves.append(
                        _Move(
                            kind="relocate",
                            dims=(v,),
                            groups=(A, B),
                            dJ_struct=dJ_struct,
                            gml=gml,
                            score=dJ_struct - lam * gml,
                        )
                    )

                # ---------------- swap ----------------
                for _ in range(self.cfg.n_swap_pairs):
                    A = int(donor_pool[rng.randint(len(donor_pool))])
                    B = int(recv_pool[rng.randint(len(recv_pool))])
                    if A == B:
                        continue
                    gA = cur_groups[A]
                    gB = cur_groups[B]

                    v = _choose_suspicious_dim(
                        ctx, gA, cur_bits[A],
                        rng=rng,
                        sample_k=self.cfg.dims_sample_per_group,
                        alpha=self.cfg.suspicious_alpha,
                    )
                    u = _choose_suspicious_dim(
                        ctx, gB, cur_bits[B],
                        rng=rng,
                        sample_k=self.cfg.dims_sample_per_group,
                        alpha=self.cfg.suspicious_alpha,
                    )
                    if v == u:
                        continue

                    gA2 = _remove_one(gA, v) + [u]
                    gB2 = _remove_one(gB, u) + [v]

                    dJ_struct = (
                        _proxy_D(ctx, gA2, cur_bits[A])
                        + _proxy_D(ctx, gB2, cur_bits[B])
                        - D_before[A]
                        - D_before[B]
                    )

                    moves.append(
                        _Move(
                            kind="swap",
                            dims=(v, u),
                            groups=(A, B),
                            dJ_struct=dJ_struct,
                            gml=0.0,
                            score=dJ_struct,
                        )
                    )

                proposed += len(moves)
                if not moves:
                    continue

                # shortlist
                moves.sort(key=lambda m: m.score)
                moves = moves[: self.cfg.per_state_shortlist_k]

                evaluated = 0
                for mv in moves:
                    if evaluated >= self.cfg.per_state_eval_topk:
                        break

                    try:
                        if mv.kind == "relocate":
                            A, B = mv.groups
                            (v,) = mv.dims
                            cand_groups = _apply_relocate(cur_groups, A, B, v)
                        else:
                            A, B = mv.groups
                            v, u = mv.dims
                            cand_groups = _apply_swap(cur_groups, A, B, v, u)
                    except Exception:
                        continue

                    _validate_partition(cand_groups, d=d)
                    key = _key_digest(ctx, cand_groups)

                    prev = seen.get_best(key)
                    if prev is not None and prev <= cur_J:
                        continue

                    # >>> ONLY HERE we solve bits <<<
                    alloc2 = ctx.solve_bits(cand_groups)
                    J2 = float(alloc2.J)
                    b2 = [int(b) for b in alloc2.bits]

                    seen.set_best(key, J2)
                    children.append(_State(cand_groups, b2, J2))

                    dp_evals += 1
                    evaluated += 1

            # ------------------------------------------------------------
            # beam update
            # ------------------------------------------------------------
            candidates = beam + children
            uniq: Dict[Tuple[int, int], _State] = {}
            for st in candidates:
                k = _key_digest(ctx, st.groups)
                if k not in uniq or st.J < uniq[k].J:
                    uniq[k] = st

            uniq_list = sorted(uniq.values(), key=lambda s: s.J)
            beam = uniq_list[: self.cfg.beam_w]

            beam_best = beam[0].J
            if beam_best < global_best - self.cfg.eps_improve:
                global_best = beam_best
                global_best_groups = _deepcopy_groups(beam[0].groups)
                global_best_bits = list(beam[0].bits)
                no_improve = 0
            else:
                no_improve += 1

            if self.cfg.verbose:
                print(
                    f"[mcb] it={it+1:03d}/{self.cfg.iters} "
                    f"beam_best={beam_best:.6f} globalBest={global_best:.6f} "
                    f"beam={len(beam)} cand={len(candidates)} uniq={len(uniq_list)} "
                    f"dp_evals={dp_evals} proposed={proposed} no_improve={no_improve}"
                )

        allocF = ctx.solve_bits(global_best_groups)
        return global_best_groups, list(allocF.bits)