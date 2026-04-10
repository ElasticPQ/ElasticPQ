from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Tuple, Protocol, Optional, Dict, Any, TypeVar, Generic, OrderedDict

import numpy as np

Groups = List[List[int]]
Bits = List[int]

K = TypeVar("K")
V = TypeVar("V")


class TypingOptional:
    pass


class LRUCache(Generic[K, V]):
    """A tiny LRU cache with maxsize.

    - get(key) returns value or None (like dict.get)
    - set(key, value) inserts/updates and evicts LRU if needed
    - pop(key) removes key if exists
    - clear() clears all
    """

    __slots__ = ("maxsize", "_od")

    def __init__(self, maxsize: int):
        ms = int(maxsize)
        if ms <= 0:
            raise ValueError(f"LRUCache.maxsize must be positive, got {maxsize}")
        self.maxsize: int = ms
        self._od: "OrderedDict[K, V]" = OrderedDict[K, V]()

    def __len__(self) -> int:
        return len(self._od)

    def get(self, key: K, default: TypingOptional[V] = None) -> TypingOptional[V]:
        od = self._od
        if key in od:
            od.move_to_end(key, last=True)
            return od[key]
        return default

    def set(self, key: K, value: V) -> None:
        od = self._od
        if key in od:
            od[key] = value
            od.move_to_end(key, last=True)
            return
        od[key] = value
        if len(od) > self.maxsize:
            od.popitem(last=False)  # evict LRU

    def pop(self, key: K) -> None:
        self._od.pop(key, None)

    def clear(self) -> None:
        self._od.clear()

    def items(self):
        return self._od.items()

# ============================================================
# Proxy + Context (place BEFORE Grouper protocols)
# ============================================================

DimsKey = Tuple[int, ...]  # sorted dims
DGKey = Tuple[DimsKey, int]  # (dims_key, bits)


class Proxy(Protocol):
    """A proxy objective evaluator with caching.

    Core API:
      - D(dims, b): per-group holdout MSE proxy for dims at bits=b
      - J(groups, bits): sum_g D(g, b_g)
    """

    def D(self, dims: List[int], b: int) -> float: ...

    def J(self, groups: Groups, bits: Bits) -> float: ...


# ============================================================
# Bit allocation (oracle) protocols
# ============================================================

PartKey = Tuple[DimsKey, ...]


@dataclass
class BitAllocResult:
    """Result of optimal bit allocation under a fixed partition."""
    J: float
    bits: Bits


class BitAllocator(Protocol):
    """Given groups, compute optimal bits (sum=ctx.B) minimizing Σ D(g, b_g),
    with per-group bound 0 <= b_g <= ctx.bmax.
    """

    def solve_bits(
            self,
            ctx: EPQContext,
            *,
            groups: Groups,
            allow_partial: bool = False,
    ) -> BitAllocResult: ...


# ============================================================
# EPQ Context
# ============================================================


@dataclass
class EPQContext:
    """Shared context for groupers and proxies.

    Holds:
      - raw data x, (n, d)
      - fixed d, B, and bmax
      - fixed train/eval split (xt_train / xt_eval)
      - proxy (with cache) for fast repeated scoring
      - bit allocator (DP oracle) for optimal bit assignment under a partition
    """

    x: np.ndarray
    d: int
    B: int
    bmax: int = 12  # global per-group upper bound

    # split policy
    seed: int = 123
    max_train_rows: int = 16384
    max_eval_rows: int = 4096
    eval_frac: float = 0.2

    # fixed split buffers (computed once), used by proxies
    xt_train: np.ndarray = field(init=False)
    xt_eval: np.ndarray = field(init=False)

    # proxy is attached after init (so proxies can capture ctx)
    proxy: Optional[Proxy] = field(default=None, repr=False)

    # allocator is attached after init (default DP allocator is used if None)
    bit_alloc: Optional[BitAllocator] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        x = np.ascontiguousarray(self.x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"EPQContext.x must be 2D, got shape={x.shape}")
        n, dd = x.shape
        if int(dd) != int(self.d):
            raise ValueError(f"EPQContext.d mismatch: x has d={dd}, ctx.d={self.d}")
        if int(self.d) <= 0:
            raise ValueError("EPQContext.d must be positive")
        if int(self.B) < 0:
            raise ValueError("EPQContext.B must be non-negative")
        if int(self.bmax) < 0:
            raise ValueError("EPQContext.bmax must be non-negative")

        self.xt_train, self.xt_eval = split_train_eval_rows(
            x,
            max_train=int(self.max_train_rows),
            max_eval=int(self.max_eval_rows),
            eval_frac=float(self.eval_frac),
            seed=int(self.seed),
        )

        # default allocator if user didn't attach one
        if self.bit_alloc is None:
            self.bit_alloc = DPBitAllocator()

    def require_proxy(self) -> Proxy:
        if self.proxy is None:
            raise ValueError("EPQContext.proxy is None; attach a Proxy (e.g., KMeansHoldoutProxy)")
        return self.proxy

    def require_bit_alloc(self) -> BitAllocator:
        if self.bit_alloc is None:
            raise ValueError("EPQContext.bit_alloc is None; attach a BitAllocator (e.g., DPBitAllocator)")
        return self.bit_alloc

    # ----------------------------
    # Common “oracle” entrypoints
    # ----------------------------

    def solve_bits(
            self,
            groups: Groups,
            *,
            allow_partial: bool = False,
    ) -> BitAllocResult:
        """Optimal bits allocation for a fixed partition.

        allow_partial=False (default):
            groups must be a full partition covering exactly d dims.

        allow_partial=True:
            groups may cover only a subset of [0..d-1]. This is useful for
            forwarders that temporarily exclude some dims (e.g., a fixed 0-bit dead group)
            while running DP on the remaining groups.
        """
        alloc = self.require_bit_alloc()
        return alloc.solve_bits(self, groups=groups, allow_partial=allow_partial)

    def J_opt(
        self,
        groups: Groups,
    ) -> float:
        """Convenience: return optimal J only."""
        return float(self.solve_bits(groups).J)

    def partition_key(self, groups: Groups) -> PartKey:
        """Canonical partition key independent of group ordering."""
        parts: List[DimsKey] = []
        for g in groups:
            dims = tuple(sorted(int(i) for i in g))
            parts.append(dims)
        parts.sort()
        return tuple(parts)


# ============================================================
# Split helper (fixed policy)
# ============================================================


def split_train_eval_rows(
    x: np.ndarray,
    *,
    max_train: int,
    max_eval: int,
    eval_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed splitting policy: reusable train/eval split for proxy."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    n0 = int(x.shape[0])
    if n0 <= 1:
        return x, x

    rs = np.random.RandomState(int(seed))
    want_train = int(max(1, max_train))
    want_eval = int(max(1, max_eval))

    if n0 >= want_train + want_eval:
        idx = rs.choice(n0, size=want_train + want_eval, replace=False)
        tr = idx[:want_train]
        ev = idx[want_train: want_train + want_eval]
        return x[tr], x[ev]

    frac = float(eval_frac)
    frac = min(max(frac, 0.05), 0.5)
    perm = rs.permutation(n0)
    ne = max(1, int(round(n0 * frac)))
    ne = min(ne, n0 - 1)
    ev = perm[:ne]
    tr = perm[ne:]

    if tr.shape[0] > want_train:
        tr = tr[:want_train]
    if ev.shape[0] > want_eval:
        ev = ev[:want_eval]
    return x[tr], x[ev]


# ============================================================
# Stable hashing helper (for proxy seeds)
# ============================================================


def stable_hash_dims(dims_key: DimsKey) -> int:
    """Stable hash for dims across processes (avoid Python's salted hash())."""
    import zlib

    arr = np.asarray(dims_key, dtype=np.int32)
    return int(zlib.crc32(arr.tobytes()) & 0x7FFFFFFF)


# ============================================================
# KMeans holdout proxy
# ============================================================


def kmeans_recon_mse_holdout(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    k: int,
    *,
    niter: int,
    nredo: int,
    seed: int,
    min_points_per_centroid: int,
) -> float:
    """Holdout recon MSE via Faiss KMeans (median over nredo runs)."""
    import faiss

    x_train = np.ascontiguousarray(x_train, dtype=np.float32)
    x_eval = np.ascontiguousarray(x_eval, dtype=np.float32)

    n_tr, ds = x_train.shape
    n_ev = int(x_eval.shape[0])
    if n_tr <= 0 or ds <= 0 or n_ev <= 0:
        return 0.0

    kk = min(int(k), max(1, n_tr))
    if kk == 1:
        mu = x_train.mean(axis=0, keepdims=True)
        diff = x_eval - mu
        return float(np.mean(np.sum(diff * diff, axis=1)))

    R = max(1, int(nredo))
    mses = np.empty(R, dtype=np.float64)

    for r in range(R):
        run_seed = int(seed + 10007 * r)
        km = faiss.Kmeans(
            ds,
            kk,
            niter=int(niter),
            nredo=1,
            verbose=False,
            seed=run_seed,
            min_points_per_centroid=int(min_points_per_centroid),
        )
        km.train(x_train)
        C = np.ascontiguousarray(km.centroids, dtype=np.float32)

        index = faiss.IndexFlatL2(ds)
        index.add(C)
        D, _ = index.search(x_eval, 1)
        mses[r] = float(np.mean(D.reshape(-1)))

    return float(np.median(mses))


@dataclass
class KMeansHoldoutProxy(Proxy):
    """KMeans holdout proxy with cross-grouper cache.

    Cache granularity:
      - D_cache[(dims_key, b)] = D_g(b)
    Optional:
      - slice caches to reduce repeated x[:, dims] materialization

    Memory safety:
      - All caches are bounded LRU (maxsize configurable).
    """

    ctx: EPQContext

    # proxy parameters
    km_niter: int = 8
    km_nredo: int = 1
    min_points_per_centroid: int = 4

    # slice caches (can be memory heavy; turn off by setting to False)
    cache_slices: bool = True

    # ---- LRU sizes (tune these) ----
    # D_cache stores floats, can be larger
    max_D_cache: int = 400_000
    # slice caches store ndarrays; keep smaller
    max_slice_cache: int = 8_192

    # ---- caches (initialized in __post_init__) ----
    D_cache: LRUCache[DGKey, float] = field(init=False, repr=False)
    Xtr_cache: LRUCache[DimsKey, np.ndarray] = field(init=False, repr=False)
    Xev_cache: LRUCache[DimsKey, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.D_cache = LRUCache[DGKey, float](maxsize=int(self.max_D_cache))
        self.Xtr_cache = LRUCache[DimsKey, np.ndarray](maxsize=int(self.max_slice_cache))
        self.Xev_cache = LRUCache[DimsKey, np.ndarray](maxsize=int(self.max_slice_cache))

    @property
    def bmax(self) -> int:
        return int(self.ctx.bmax)

    def _get_slices(self, dims_key: DimsKey) -> Tuple[np.ndarray, np.ndarray]:
        if not self.cache_slices:
            xtr = self.ctx.xt_train[:, dims_key]
            xev = self.ctx.xt_eval[:, dims_key]
            return xtr, xev

        xtr = self.Xtr_cache.get(dims_key, None)
        if xtr is None:
            xtr = self.ctx.xt_train[:, dims_key]
            self.Xtr_cache.set(dims_key, xtr)

        xev = self.Xev_cache.get(dims_key, None)
        if xev is None:
            xev = self.ctx.xt_eval[:, dims_key]
            self.Xev_cache.set(dims_key, xev)

        return xtr, xev

    def D(self, dims: List[int], b: int) -> float:
        dims_i = [int(j) for j in dims]
        if not dims_i:
            return 0.0

        bb = int(b)
        if bb < 0 or bb > int(self.bmax):
            raise ValueError(f"Proxy.D: bits={bb} out of range [0,{int(self.bmax)}]")

        dims_key: DimsKey = tuple(sorted(dims_i))
        key: DGKey = (dims_key, bb)

        hit = self.D_cache.get(key, None)
        if hit is not None:
            return float(hit)

        xtr, xev = self._get_slices(dims_key)

        base_seed = int(self.ctx.seed) + stable_hash_dims(dims_key) % 100000
        run_seed = int(base_seed + 7919 * bb)

        k = 1 << bb
        val = kmeans_recon_mse_holdout(
            xtr,
            xev,
            k,
            niter=int(self.km_niter),
            nredo=int(self.km_nredo),
            seed=int(run_seed),
            min_points_per_centroid=int(self.min_points_per_centroid),
        )
        self.D_cache.set(key, float(val))
        return float(val)

    def J(self, groups: Groups, bits: Bits) -> float:
        if len(groups) != len(bits):
            raise ValueError(f"Proxy.J: len(groups)={len(groups)} != len(bits)={len(bits)}")

        J = 0.0
        for g, b in zip(groups, bits):
            J += float(self.D(g, int(b)))
        return float(J)


# ============================================================
# Default DP BitAllocator (no caps)
# ============================================================


def _validate_partition(
    groups: Groups,
    d: int,
    *,
    require_cover: bool = True,
    allow_empty_group: bool = False,
) -> None:
    """Validate partition of dims.

    require_cover=True (default):
        groups must cover exactly d dims and form a true partition of [0..d-1].

    require_cover=False:
        groups may cover only a subset of [0..d-1], but still enforces:
          - no duplicated dims
          - all dim ids within [0, d)
          - non-empty groups unless allow_empty_group=True
    """
    if not groups:
        raise ValueError("groups is empty")
    if (not allow_empty_group) and any(len(g) == 0 for g in groups):
        raise ValueError("groups contains an empty group")

    flat = [int(i) for g in groups for i in g]

    # no duplicates
    if len(set(flat)) != len(flat):
        raise ValueError("groups contains duplicated dims")

    # range check always
    dd = int(d)
    if any((i < 0) or (i >= dd) for i in flat):
        want = set(range(dd))
        got = set(flat)
        missing = sorted(want - got)
        extra = sorted(got - want)
        raise ValueError(f"groups has invalid dim ids: missing={missing} extra={extra}")

    if require_cover:
        if len(flat) != dd:
            raise ValueError(f"groups must cover exactly d={dd} dims, got {len(flat)}")
        want = set(range(dd))
        got = set(flat)
        if got != want:
            missing = sorted(want - got)
            extra = sorted(got - want)
            raise ValueError(f"groups has invalid dim ids: missing={missing} extra={extra}")



def _validate_bits_vector(
    bits: Bits,
    *,
    M: int,
    B: int,
    bmax: int,
    name: str,
    require_sum: bool = True,
) -> Bits:
    bits_i = [int(b) for b in bits]
    if len(bits_i) != int(M):
        raise ValueError(f"{name} length must equal M={int(M)}, got {len(bits_i)}")
    if any((b < 0) or (b > int(bmax)) for b in bits_i):
        raise ValueError(f"{name} must stay in [0,{int(bmax)}], got {bits_i}")
    if bool(require_sum) and int(sum(bits_i)) != int(B):
        raise ValueError(f"{name} must sum to B={int(B)}, got {sum(bits_i)}")
    return bits_i


def _dp_allocate_no_caps(
    *,
    proxy: Proxy,
    dims_keys: List[DimsKey],
    B: int,
    bmax: int,
) -> Tuple[float, Bits]:
    """Exact DP: min Σ D(g, b_g) s.t. Σ b_g = B, 0<=b_g<=bmax."""
    M = int(len(dims_keys))
    B = int(B)
    bmax = int(bmax)
    if M <= 0:
        raise ValueError("no groups")
    if B < 0:
        raise ValueError("B must be non-negative")
    if bmax < 0:
        raise ValueError("bmax must be non-negative")

    # Quick infeasibility check:
    if B > M * bmax:
        raise RuntimeError(f"Infeasible: B={B} > M*bmax={M*bmax} (M={M}, bmax={bmax})")

    INF = 1e100
    dp = np.full((M + 1, B + 1), INF, dtype=np.float64)
    choice = np.full((M + 1, B + 1), -1, dtype=np.int16)
    dp[0, 0] = 0.0

    for i in range(1, M + 1):
        # b in [0..min(bmax, B)]
        cap = int(min(bmax, B))

        # cost[b] = D(group_i, b)
        costs = np.empty(cap + 1, dtype=np.float64)
        dims = list(dims_keys[i - 1])
        for b in range(cap + 1):
            costs[b] = float(proxy.D(dims, int(b)))

        prev = dp[i - 1]
        cur = dp[i]
        cur_choice = choice[i]

        for t in range(B + 1):
            base = float(prev[t])
            if base >= INF / 2:
                continue
            max_add = min(cap, B - t)
            for b in range(max_add + 1):
                v = base + float(costs[b])
                tt = t + b
                if v < cur[tt]:
                    cur[tt] = v
                    cur_choice[tt] = int(b)

    J = float(dp[M, B])
    if not np.isfinite(J) or J >= INF / 2:
        raise RuntimeError("DP allocation failed")

    bits = [0] * M
    t = B
    for i in range(M, 0, -1):
        b = int(choice[i, t])
        if b < 0:
            raise RuntimeError("DP backtrack failed")
        bits[i - 1] = b
        t -= b
    if t != 0:
        raise RuntimeError(f"DP backtrack ended with residual bits t={t}")

    bits_i = [int(b) for b in bits]
    if any((b < 0) or (b > bmax) for b in bits_i):
        raise RuntimeError(f"DP produced bits outside [0,{bmax}]: {bits_i}")
    if int(sum(bits_i)) != B:
        raise RuntimeError(f"DP produced invalid sum(bits)={sum(bits_i)} for B={B}")

    return float(J), bits_i


@dataclass
class DPBitAllocator(BitAllocator):
    """Default exact DP allocator (no caps)."""

    def solve_bits(
            self,
            ctx: EPQContext,
            *,
            groups: Groups,
            allow_partial: bool = False,
    ) -> BitAllocResult:
        proxy = ctx.require_proxy()
        d = int(ctx.d)
        B = int(ctx.B)
        bmax = int(ctx.bmax)

        _validate_partition(groups, d=d, require_cover=(not bool(allow_partial)))

        M = len(groups)

        # Canonical dims keys aligned with *caller order*
        dims_keys: List[DimsKey] = []
        for g in groups:
            dims_keys.append(tuple(sorted(int(i) for i in g)))

        J, bits = _dp_allocate_no_caps(proxy=proxy, dims_keys=dims_keys, B=B, bmax=bmax)
        bits = _validate_bits_vector(bits, M=M, B=B, bmax=bmax, name="allocated bits")
        return BitAllocResult(J=float(J), bits=bits)


# ============================================================
# Convenience: build ctx + attach default proxy + allocator
# ============================================================


def make_default_context_with_proxy(
    x: np.ndarray,
    *,
    d: int,
    B: int,
    bmax: int = 12,
    seed: int = 123,
    max_train_rows: int = 16384,
    max_eval_rows: int = 4096,
    eval_frac: float = 0.2,
    # proxy params
    km_niter: int = 8,
    km_nredo: int = 1,
    min_points_per_centroid: int = 4,
    cache_slices: bool = True,
) -> EPQContext:
    """Convenience: build ctx + attach KMeansHoldoutProxy + DPBitAllocator."""
    ctx = EPQContext(
        x=x,
        d=int(d),
        B=int(B),
        bmax=int(bmax),
        seed=int(seed),
        max_train_rows=int(max_train_rows),
        max_eval_rows=int(max_eval_rows),
        eval_frac=float(eval_frac),
    )
    ctx.proxy = KMeansHoldoutProxy(
        ctx=ctx,
        km_niter=int(km_niter),
        km_nredo=int(km_nredo),
        min_points_per_centroid=int(min_points_per_centroid),
        cache_slices=bool(cache_slices),
    )
    # ctx.bit_alloc defaults to DPBitAllocator() in __post_init__
    return ctx


# ============================================================
# Grouper protocols (now come AFTER Context/Proxy/Allocator)
# ============================================================

class Grouper(Protocol):
    def build_groups(self, ctx: EPQContext) -> Tuple[Groups, Bits]:
        ...

    def then(self, fwd: "ForwardingGrouper") -> "Grouper":
        return _ChainedGrouper(self, fwd)


class ForwardingGrouper(Protocol):
    def forward_groups(self, ctx: EPQContext, *, groups: Groups, bits: Bits) -> Tuple[Groups, Bits]:
        ...

    def then(self, fwd: "ForwardingGrouper") -> "ForwardingGrouper":
        return _ChainedForwardingGrouper(self, fwd)


@dataclass
class _ChainedGrouper(Grouper):
    first: Grouper
    second: ForwardingGrouper

    def build_groups(self, ctx: EPQContext) -> Tuple[Groups, Bits]:
        g, b = self.first.build_groups(ctx)
        return self.second.forward_groups(ctx, groups=g, bits=b)


@dataclass
class _ChainedForwardingGrouper(ForwardingGrouper):
    first: ForwardingGrouper
    second: ForwardingGrouper

    def forward_groups(self, ctx: EPQContext, *, groups: Groups, bits: Bits) -> Tuple[Groups, Bits]:
        g1, b1 = self.first.forward_groups(ctx, groups=groups, bits=bits)
        return self.second.forward_groups(ctx, groups=g1, bits=b1)


class DefaultGrouper(Grouper):
    """Default grouper: PQ-like grouping with 8 bits per group.

    - Groups dimensions evenly (like classic PQ)
    - Assigns 8 bits per group by default
    - If B < 8, uses a single group with B bits
    """

    def build_groups(self, ctx: EPQContext) -> Tuple[Groups, Bits]:
        d = int(ctx.d)
        B = int(ctx.B)
        if d <= 0:
            raise ValueError("d must be positive")
        if B < 0:
            raise ValueError("B must be non-negative")

        if B >= 8:
            M = B // 8
            bits = [8] * M
            rem = B - 8 * M
            if rem > 0:
                bits[-1] += rem
        else:
            M = 1
            bits = [B]

        if M <= 0:
            raise ValueError("Invalid number of groups")

        base = d // M
        extra = d % M

        groups: List[List[int]] = []
        cur = 0
        for i in range(M):
            size = base + (1 if i < extra else 0)
            if size <= 0:
                raise ValueError("Invalid group size (too many groups for d)")
            groups.append(list(range(cur, cur + size)))
            cur += size

        assert cur == d, "Dimension partition mismatch"
        assert len(groups) == len(bits)
        assert sum(bits) == B

        return groups, bits


class SingletonDimGrouper(Grouper):
    """Initialize from singleton groups and solve bits with the context allocator.

    This is primarily useful for EPQ ablations that disable the grow stage but
    still need a valid full partition as the starting point for later forwarders.
    """

    def build_groups(self, ctx: EPQContext) -> Tuple[Groups, Bits]:
        d = int(ctx.d)
        if d <= 0:
            raise ValueError("d must be positive")

        groups: Groups = [[i] for i in range(d)]
        alloc = ctx.solve_bits(groups)
        bits = [int(b) for b in alloc.bits]
        return groups, bits


# ============================================================
# Structure serialization (for fixed groups/bits)
# ============================================================


@dataclass
class EPQStructure:
    """Serializable structure: (d, B, groups, nbits)."""
    d: int
    B: int
    groups: Groups
    nbits: Bits
    format_version: int = 2
    meta: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        d = int(self.d)
        B = int(self.B)
        groups = self.groups
        nbits = self.nbits

        if d <= 0:
            raise ValueError("EPQStructure.d must be positive")
        if B < 0:
            raise ValueError("EPQStructure.B must be non-negative")

        if not groups:
            raise ValueError("EPQStructure.groups is empty")
        if any(len(g) == 0 for g in groups):
            raise ValueError("EPQStructure.groups contains an empty group")

        flat = [int(i) for g in groups for i in g]
        if len(flat) != d:
            raise ValueError(f"EPQStructure.groups must cover exactly d={d} dims, got {len(flat)}")
        if len(set(flat)) != len(flat):
            raise ValueError("EPQStructure.groups contains duplicated dims")

        want = set(range(d))
        got = set(flat)
        if got != want:
            missing = sorted(want - got)
            extra = sorted(got - want)
            raise ValueError(f"EPQStructure has invalid dim ids: missing={missing} extra={extra}")

        if len(nbits) != len(groups):
            raise ValueError(f"EPQStructure.nbits length must equal M={len(groups)}, got {len(nbits)}")
        nbits_i = [int(b) for b in nbits]
        if any(b < 0 for b in nbits_i):
            raise ValueError(f"EPQStructure.nbits contains negative bits: {nbits_i}")
        if int(sum(nbits_i)) != B:
            raise ValueError(f"EPQStructure sum(nbits) must equal B={B}, got {sum(nbits_i)}")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "format_version": int(self.format_version),
            "d": int(self.d),
            "B": int(self.B),
            "groups": [list(map(int, g)) for g in self.groups],
            "nbits": [int(b) for b in self.nbits],
            "meta": dict(self.meta) if self.meta is not None else None,
        }

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> "EPQStructure":
        if not isinstance(obj, dict):
            raise ValueError("EPQStructure.from_dict expects a dict")
        fmt = int(obj.get("format_version", 2))
        if fmt not in (1, 2):
            raise ValueError(f"Unsupported EPQStructure format_version={fmt}")
        s = EPQStructure(
            d=int(obj["d"]),
            B=int(obj["B"]),
            groups=[list(map(int, g)) for g in obj["groups"]],
            nbits=[int(b) for b in obj["nbits"]],
            format_version=fmt,
            meta=obj.get("meta", None),
        )
        s.validate()
        return s

    def save_json(self, path: str) -> None:
        if not path:
            raise ValueError("save_json path is empty")
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_json(path: str) -> "EPQStructure":
        if not path:
            raise ValueError("load_json path is empty")
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return EPQStructure.from_dict(obj)

    @staticmethod
    def from_grouper(
        grouper: Grouper,
        ctx: EPQContext,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "EPQStructure":
        groups, nbits = grouper.build_groups(ctx)
        s = EPQStructure(
            d=int(ctx.d),
            B=int(ctx.B),
            groups=[list(g) for g in groups],
            nbits=[int(b) for b in nbits],
            meta=meta,
        )
        s.validate()
        return s


class FixedStructureGrouper(Grouper):
    """A Grouper that always returns the same (groups, nbits)."""

    def __init__(self, structure: EPQStructure):
        structure.validate()
        self.structure = structure

    @property
    def is_fixed_structure(self) -> bool:
        return True

    def build_groups(self, ctx: EPQContext) -> Tuple[Groups, Bits]:
        d = int(ctx.d)
        B = int(ctx.B)
        if int(d) != int(self.structure.d):
            raise ValueError(f"FixedStructureGrouper: d mismatch: got {d}, want {self.structure.d}")
        if int(B) != int(self.structure.B):
            raise ValueError(f"FixedStructureGrouper: B mismatch: got {B}, want {self.structure.B}")
        groups = [list(g) for g in self.structure.groups]
        nbits = [int(b) for b in self.structure.nbits]
        return groups, nbits
