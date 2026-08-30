"""Centroid cloud: a fixed-capacity set of cluster features with damped decay,
adaptive absorption, utility-weighted splitting/merging, and reward-protected
exemplar pointers.

Math
----
Vectors are L2-normalised. Each cluster stores a weighted linear sum LS and a
weight n, both "inflated" by a global scalar so decay is O(1):

    effective_LS = LS / iw          effective_n = n / iw
    centroid mu  = LS / n           (scale-free)
    radius       = sqrt(1 - ||mu||^2)   in [0, 1]

Because vectors are unit-norm, sum of squared norms == n, so the BIRCH triple
(n, LS, SS) collapses to a pair. Merging is addition.

Layout (v2)
-----------
Per cluster: LS float32[d] + C float16[d] + n f64 + util/hits/t_off f32
             + alive u8 + EX int32[E] + ex_seen/ex_fill int32 + ex_keep int64

    bytes_per_cluster = 6*dim + 37 + 4*E

v1 used float32 centroids and int64 pointers (8*dim + 33 + 8*E). The exemplar
pointer is the unit of *addressability* -- a cluster can only surface what it
points at -- and v1 spent 95% of a byte budget on geometry and 4% on pointers.
At d=384 the new layout is 2,469 B/cluster at E=32 versus 3,233 B at E=16:
more clusters and twice the reachable items for fewer bytes.

Reward-driven addressability
----------------------------
Two mechanisms, both bounded:

  * PROTECTION. An exemplar the model actually cited gets a bit in `ex_keep`
    and is exempt from reservoir eviction (capped at `protect_frac` of the
    row, so a cluster can never freeze completely).
  * OVERFLOW BLOCKS. A pool of `extra_frac * capacity` blocks of
    `extra_exemplars` pointers. A cluster whose utility crosses `grant_util`
    claims one, stealing from the lowest-utility owner if the pool is empty.
    Resident memory is still constant -- the pool is pre-allocated.

Split thresholding
------------------
The split threshold is RELATIVE by default (a quantile of the live radius
distribution). An absolute `split_radius` is a footgun: radius depends on the
encoder's density (hashing ~0.16, MiniLM ~0.45), so a fixed 0.55 means the
allocator never fires and the utility policy is untestable rather than wrong.
Set `split_radius` only as an extra hard floor.
"""

from __future__ import annotations

import json
import math
import os
from typing import Callable, Iterable, Sequence

import numpy as np

EMPTY = -1
_SIM_CHUNK = 8192          # rows per fp16->fp32 conversion block
_MAX_PROTECT_BIT = 62      # ex_keep is int64; bits above this are ignored
_I32_MAX = 2 ** 31 - 2


class CentroidCloud:
    """Fixed-capacity streaming cluster store.

    Parameters
    ----------
    dim:
        Embedding dimensionality.
    capacity:
        Hard cap on live centroids. Resident memory is O(capacity) and never
        exceeds `capacity * bytes_per_cluster(...)`.
    alpha:
        Absorption slack. tau_i = 1 - alpha * radius_i, so dense clusters
        demand a closer match and diffuse ones absorb freely.
    tau_min, tau_max:
        Clamps on the adaptive threshold.
    half_life_s:
        Half-life of cluster weight in seconds. None disables decay.
    exemplars:
        Base reservoir slots per cluster (int32 pointers into the cold store).
        Slot 0 is always the most recent item, so recency is never sampled out.
    extra_exemplars, extra_frac:
        Utility-granted overflow pointer blocks. `extra_frac * capacity` blocks
        of `extra_exemplars` slots, pre-allocated. 0 disables the tier.
    grant_util:
        Utility a cluster must reach to claim an overflow block.
    protect_frac:
        Ceiling on the fraction of a cluster's row that citation-protection may
        pin. Prevents a hot cluster's pointer set from ossifying.
    radius_max:
        A merge is refused if the merged radius would exceed this.
    split_quantile:
        Radius quantile a cluster must reach to be a split candidate.
    split_radius:
        Optional ABSOLUTE floor on the split threshold. None = purely relative.
    min_weight:
        Clusters whose effective weight falls below this are pruned.
    """

    # ------------------------------------------------------------ accounting

    @staticmethod
    def bytes_per_cluster(dim: int, exemplars: int,
                          extra_exemplars: int = 0,
                          extra_frac: float = 0.0) -> int:
        """Exact resident cost of one cluster slot, including its share of the
        overflow pool. Use this rather than guessing; the benchmark and the
        capacity solver both call it so the numbers cannot drift apart."""
        d, E = int(dim), int(exemplars)
        base = (4 * d          # LS      float32
                + 2 * d        # C       float16
                + 8            # n       float64
                + 4            # util    float32
                + 4            # hits    float32
                + 4            # t_off   float32
                + 1            # alive   bool
                + 4 * E        # EX      int32
                + 4            # ex_seen int32
                + 4            # ex_fill int32
                + 8)           # ex_keep int64
        if extra_exemplars > 0 and extra_frac > 0.0:
            base += 4                                            # x_of int32
            base += extra_frac * (4 * int(extra_exemplars) + 8)   # XE + owner/seen
        return int(math.ceil(base))

    @classmethod
    def capacity_for_bytes(cls, byte_budget: int, dim: int, exemplars: int,
                           extra_exemplars: int = 0, extra_frac: float = 0.0,
                           min_capacity: int = 32) -> int:
        per = cls.bytes_per_cluster(dim, exemplars, extra_exemplars, extra_frac)
        return max(int(min_capacity), int(byte_budget) // per)

    # ----------------------------------------------------------------- init

    def __init__(
        self,
        dim: int,
        capacity: int = 10_000,
        alpha: float = 0.35,
        tau_min: float = 0.30,
        tau_max: float = 0.94,
        half_life_s: float | None = 30 * 24 * 3600.0,
        exemplars: int = 32,
        extra_exemplars: int = 0,
        extra_frac: float = 0.0,
        grant_util: float = 0.60,
        protect_frac: float = 0.50,
        radius_max: float = 0.80,
        split_quantile: float = 0.80,
        split_radius: float | None = None,
        min_weight: float = 0.05,
        util_eta: float = 0.10,
        merge_pool: int = 256,
        seed: int = 0,
    ) -> None:
        self.dim = int(dim)
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.lam = 0.0 if not half_life_s else math.log(2.0) / float(half_life_s)
        self.E = int(exemplars)
        self.X_E = max(0, int(extra_exemplars))
        self.extra_frac = max(0.0, float(extra_frac))
        self.grant_util = float(grant_util)
        self.protect_frac = float(np.clip(protect_frac, 0.0, 0.9))
        self.radius_max = float(radius_max)
        self.split_quantile = float(split_quantile)
        self.split_radius = None if split_radius is None else float(split_radius)
        self.min_weight = float(min_weight)
        self.util_eta = float(util_eta)
        self.merge_pool = int(merge_pool)

        cap, d = self.capacity, self.dim
        self.LS = np.zeros((cap, d), dtype=np.float32)    # inflated linear sum
        self.C = np.zeros((cap, d), dtype=np.float16)     # normalised centroid
        self.n = np.zeros(cap, dtype=np.float64)          # inflated weight
        self.util = np.full(cap, 0.5, dtype=np.float32)
        self.hits = np.zeros(cap, dtype=np.float32)
        self.t_off = np.zeros(cap, dtype=np.float32)      # seconds since origin
        self.alive = np.zeros(cap, dtype=bool)
        self.EX = np.full((cap, max(self.E, 1)), EMPTY, dtype=np.int32)
        self.ex_seen = np.zeros(cap, dtype=np.int32)
        self.ex_fill = np.zeros(cap, dtype=np.int32)
        self.ex_keep = np.zeros(cap, dtype=np.int64)      # citation-protect bits

        n_blocks = int(self.extra_frac * cap) if self.X_E > 0 else 0
        self.n_blocks = n_blocks
        self.XE = np.full((max(n_blocks, 1), max(self.X_E, 1)), EMPTY, dtype=np.int32)
        self.x_owner = np.full(max(n_blocks, 1), -1, dtype=np.int32)
        self.x_of = np.full(cap, -1, dtype=np.int32)
        self.x_free: list[int] = list(range(n_blocks))

        self.hi = 0                    # high-water slot mark
        self.free: list[int] = []
        self.iw = 1.0                  # global inflation factor
        self.t_ref = 0.0
        self.origin = 0.0
        self._split_thresh = 0.0
        self.rng = np.random.default_rng(seed)
        self.counters = {"inserts": 0, "spawns": 0, "absorbs": 0,
                         "merges": 0, "splits": 0, "split_tried": 0,
                         "split_failed": 0, "evictions": 0, "prunes": 0,
                         "protects": 0, "grants": 0, "steals": 0}

    # ---------------------------------------------------------------- state

    @property
    def n_clusters(self) -> int:
        return int(self.alive.sum())

    def weight(self, idx):
        return self.n[idx] / self.iw

    def radius(self, idx):
        n = np.maximum(self.n[idx], 1e-12)
        mu = np.linalg.norm(self.LS[idx], axis=-1) / n
        return np.sqrt(np.clip(1.0 - mu * mu, 0.0, 1.0))

    def value(self, idx):
        """Budget currency. Utility earns resolution; bulk alone does not."""
        return (0.10 + self.util[idx]) * np.log1p(self.weight(idx))

    def resident_bytes(self) -> int:
        b = (self.LS.nbytes + self.C.nbytes + self.n.nbytes + self.util.nbytes
             + self.hits.nbytes + self.t_off.nbytes + self.alive.nbytes
             + self.EX.nbytes + self.ex_seen.nbytes + self.ex_fill.nbytes
             + self.ex_keep.nbytes)
        if self.n_blocks:
            b += self.XE.nbytes + self.x_owner.nbytes + self.x_of.nbytes
        return int(b)

    def addressable(self) -> int:
        """Number of cold-store items currently reachable through a pointer.
        This, not `clusters`, is the recall ceiling."""
        live = np.flatnonzero(self.alive[: self.hi])
        return int(self.ex_fill[live].sum()) if len(live) else 0

    # ---------------------------------------------------------------- decay

    def _advance(self, t: float) -> None:
        if self.origin == 0.0:
            self.origin = float(t)
        if self.lam <= 0.0:
            return
        if self.t_ref == 0.0:
            self.t_ref = t
            return
        dt = t - self.t_ref
        if dt <= 0.0:
            return
        self.t_ref = t
        self.iw *= math.exp(self.lam * dt)
        if self.iw > 1e6:                       # renormalise before fp32 suffers
            self.LS[: self.hi] /= self.iw
            self.n[: self.hi] /= self.iw
            self.iw = 1.0

    # ------------------------------------------------------------ similarity

    def _sims(self, x: np.ndarray) -> np.ndarray:
        """Cosine of x against every live centroid. fp16 storage is upcast in
        bounded blocks so a 10^6-cluster cloud does not allocate a 1.5 GB
        temporary per query."""
        hi = self.hi
        sims = np.full(hi, -2.0, dtype=np.float32)
        if hi == 0:
            return sims
        live = self.alive[:hi]
        for s in range(0, hi, _SIM_CHUNK):
            e = min(hi, s + _SIM_CHUNK)
            m = live[s:e]
            if not m.any():
                continue
            sims[s:e][m] = self.C[s:e][m].astype(np.float32) @ x
        return sims

    # --------------------------------------------------------------- insert

    def insert(self, x: np.ndarray, item_id: int, t: float) -> tuple[int, bool]:
        """Absorb a unit vector. Returns (cluster_slot, spawned)."""
        x = np.asarray(x, dtype=np.float32).ravel()
        nx = float(np.linalg.norm(x))
        if nx == 0.0 or not np.isfinite(nx):
            raise ValueError("zero or non-finite embedding")
        x = x / nx

        self._advance(t)
        self.counters["inserts"] += 1

        if self.n_clusters:
            sims = self._sims(x)
            i = int(np.argmax(sims))
            tau = float(np.clip(1.0 - self.alpha * float(self.radius(i)),
                                self.tau_min, self.tau_max))
            if sims[i] >= tau:
                self._absorb(i, x, item_id, t)
                return i, False

        return self._spawn(x, item_id, t), True

    def _absorb(self, i: int, x: np.ndarray, item_id: int, t: float) -> None:
        w = self.iw
        self.LS[i] += x * w
        self.n[i] += w
        ln = float(np.linalg.norm(self.LS[i]))
        self.C[i] = (self.LS[i] / ln) if ln > 0 else x
        self.t_off[i] = float(t - self.origin)
        self._reservoir_add(i, item_id)
        self.counters["absorbs"] += 1

    def _spawn(self, x: np.ndarray, item_id: int, t: float) -> int:
        slot = self._claim_slot()
        w = self.iw
        self.LS[slot] = x * w
        self.C[slot] = x
        self.n[slot] = w
        self.util[slot] = 0.5
        self.hits[slot] = 0.0
        self.t_off[slot] = float(t - self.origin)
        self.alive[slot] = True
        self.EX[slot, :] = EMPTY
        self.ex_seen[slot] = 0
        self.ex_fill[slot] = 0
        self.ex_keep[slot] = 0
        self._reservoir_add(slot, item_id)
        self.counters["spawns"] += 1
        return slot

    def _claim_slot(self) -> int:
        if self.free:
            return self.free.pop()
        if self.hi < self.capacity:
            self.hi += 1
            return self.hi - 1
        if not self._make_room():
            raise RuntimeError("cloud full and no room could be made")
        return self.free.pop()

    # ------------------------------------------------------ exemplar pointers

    def _ex_width(self, i: int) -> int:
        if self.X_E and self.x_of[i] >= 0:
            return self.E + self.X_E
        return self.E

    def _ex_get(self, i: int, j: int) -> int:
        if j < self.E:
            return int(self.EX[i, j])
        b = int(self.x_of[i])
        return int(self.XE[b, j - self.E]) if b >= 0 else EMPTY

    def _ex_set(self, i: int, j: int, v: int) -> None:
        if j < self.E:
            self.EX[i, j] = v
        else:
            b = int(self.x_of[i])
            if b >= 0:
                self.XE[b, j - self.E] = v

    def _is_protected(self, i: int, j: int) -> bool:
        if j > _MAX_PROTECT_BIT:
            return False
        return bool((int(self.ex_keep[i]) >> j) & 1)

    def _set_protected(self, i: int, j: int, on: bool) -> None:
        if j > _MAX_PROTECT_BIT:
            return
        m = int(self.ex_keep[i])
        m = (m | (1 << j)) if on else (m & ~(1 << j))
        self.ex_keep[i] = m

    def _protected_count(self, i: int) -> int:
        return int(self.ex_keep[i]).bit_count()

    def _pick_victim(self, i: int, w: int) -> int | None:
        """Uniform over unprotected slots in [1, w). Slot 0 is the newest item
        and is never a victim."""
        if w <= 1:
            return None
        mask = int(self.ex_keep[i])
        if mask == 0:
            return int(self.rng.integers(1, w))
        free = [j for j in range(1, w)
                if j > _MAX_PROTECT_BIT or not ((mask >> j) & 1)]
        if not free:
            return None
        return int(free[int(self.rng.integers(0, len(free)))])

    def _reservoir_add(self, i: int, item_id: int) -> None:
        """Algorithm R over a prefix-filled row, slot 0 pinned to the newest
        item, protected slots exempt from eviction.

        Filled slots are always the prefix [0, ex_fill), which is what makes
        hole-finding O(1) after an overflow block is granted or revoked."""
        w = self._ex_width(i)
        fill = int(self.ex_fill[i])
        if fill > w:                                  # block was revoked
            fill = w
            self.ex_fill[i] = w
        self.ex_seen[i] = min(int(self.ex_seen[i]) + 1, _I32_MAX)

        if fill == 0:
            self._ex_set(i, 0, item_id)
            self._set_protected(i, 0, False)
            self.ex_fill[i] = 1
            return

        prev0 = self._ex_get(i, 0)
        p0 = self._is_protected(i, 0)
        self._ex_set(i, 0, item_id)
        self._set_protected(i, 0, False)

        if fill < w:                                  # still growing
            self._ex_set(i, fill, prev0)
            self._set_protected(i, fill, p0)
            self.ex_fill[i] = fill + 1
            return

        seen = int(self.ex_seen[i])
        if self.rng.random() < w / float(max(seen, w)):
            j = self._pick_victim(i, w)
            if j is not None:
                self._ex_set(i, j, prev0)
                self._set_protected(i, j, p0)

    def exemplars(self, idx: Iterable[int]) -> list[int]:
        out: list[int] = []
        for i in idx:
            i = int(i)
            for j in range(int(self.ex_fill[i])):
                v = self._ex_get(i, j)
                if v != EMPTY:
                    out.append(v)
        return out

    def exemplars_by_cluster(self, idx: Iterable[int]) -> list[list[int]]:
        """One pointer list per cluster, in the order given.

        Read paths that flatten every cluster's pointers into a single
        similarity sort collapse onto the nearest cluster and lose coverage,
        which is why aggregate/"list everything" queries score badly. Keeping
        the lists separate lets the caller fuse per cluster."""
        out: list[list[int]] = []
        for i in idx:
            i = int(i)
            row = [self._ex_get(i, j) for j in range(int(self.ex_fill[i]))]
            out.append([v for v in row if v != EMPTY])
        return out

    def protect(self, cluster: int, item_id: int) -> bool:
        """Mark a pointer as citation-earned. Capped at `protect_frac` of the
        row so the reservoir never stops sampling entirely."""
        i = int(cluster)
        if not (0 <= i < self.hi) or not self.alive[i]:
            return False
        w = self._ex_width(i)
        cap = max(1, int(self.protect_frac * w))
        if self._protected_count(i) >= cap:
            return False
        for j in range(int(self.ex_fill[i])):
            if self._ex_get(i, j) == int(item_id):
                if j > _MAX_PROTECT_BIT:
                    return False
                self._set_protected(i, j, True)
                self.counters["protects"] += 1
                return True
        return False

    # ------------------------------------------------------- overflow blocks

    def _grant_block(self, i: int) -> bool:
        if not self.n_blocks or self.X_E <= 0 or self.x_of[i] >= 0:
            return False
        if float(self.util[i]) < self.grant_util:
            return False
        if self.x_free:
            b = self.x_free.pop()
        else:
            owned = np.flatnonzero(self.x_owner >= 0)
            if not len(owned):
                return False
            cl = self.x_owner[owned].astype(int)
            j = int(np.argmin(self.util[cl]))
            b, victim = int(owned[j]), int(cl[j])
            if float(self.util[victim]) >= float(self.util[i]) - 1e-6:
                return False
            self._revoke_block(b)
            self.counters["steals"] += 1
        self.x_owner[b] = i
        self.x_of[i] = b
        self.XE[b, :] = EMPTY
        self.counters["grants"] += 1
        return True

    def _revoke_block(self, b: int) -> None:
        owner = int(self.x_owner[b])
        if owner >= 0:
            self.x_of[owner] = -1
            self.ex_fill[owner] = min(int(self.ex_fill[owner]), self.E)
            m = int(self.ex_keep[owner])
            self.ex_keep[owner] = m & ((1 << min(self.E, 63)) - 1)
        self.x_owner[b] = -1
        self.XE[b, :] = EMPTY
        if b not in self.x_free:
            self.x_free.append(b)

    # ---------------------------------------------------------------- query

    def search(self, q: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=np.float32).ravel()
        nq = float(np.linalg.norm(q))
        if nq == 0.0 or self.hi == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=np.float32)
        q = q / nq
        sims = self._sims(q)
        k = min(int(k), self.n_clusters)
        if k <= 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=np.float32)
        part = np.argpartition(-sims, k - 1)[:k]
        order = part[np.argsort(-sims[part])]
        return order, sims[order]

    # --------------------------------------------------------------- credit

    def credit(self, idx: Sequence[int], reward: float) -> None:
        """reward in [-1, 1]. +1 cited/succeeded, -0.2 injected-and-ignored,
        -1 actively wrong. Mapped to a [0, 1] target and EMA'd. A positive
        update may also earn the cluster an overflow pointer block."""
        if not len(idx):
            return
        idx = np.asarray(list(idx), dtype=int)
        idx = idx[(idx >= 0) & (idx < self.capacity)]
        idx = idx[self.alive[idx]]
        if not len(idx):
            return
        target = (float(np.clip(reward, -1.0, 1.0)) + 1.0) / 2.0
        e = self.util_eta
        self.util[idx] = (1.0 - e) * self.util[idx] + e * target
        self.hits[idx] += 1.0
        if target > 0.5 and self.n_blocks:
            for i in idx[np.argsort(-self.util[idx])][:4]:
                self._grant_block(int(i))

    # ---------------------------------------------------------- maintenance

    def split_threshold(self, live: np.ndarray | None = None) -> float:
        if live is None:
            live = np.flatnonzero(self.alive[: self.hi])
        if not len(live):
            return float("inf")
        r = self.radius(live)
        thr = float(np.quantile(r, self.split_quantile))
        if self.split_radius is not None:
            thr = max(thr, self.split_radius)
        return thr

    def maintain(
        self,
        fetch_vecs: Callable[[Sequence[int]], dict[int, np.ndarray]] | None = None,
        max_splits: int = 4,
    ) -> dict:
        """Prune dead weight, then spend the freed budget splitting clusters
        that earned resolution. When the cloud is full `_claim_slot` merges the
        two lowest-value compatible clusters to pay for a split -- that
        transfer is the entire thesis and must not be silently skipped."""
        report = {"pruned": 0, "split": 0, "merged": 0, "evicted": 0,
                  "split_thresh": 0.0, "candidates": 0}
        report["pruned"] = self.prune()
        if fetch_vecs is None or max_splits <= 0:
            return report

        live = np.flatnonzero(self.alive[: self.hi])
        if len(live) < 8:
            return report
        r = self.radius(live)
        v = self.value(live)
        u = self.util[live]

        r_thr = self.split_threshold(live)
        self._split_thresh = r_thr
        report["split_thresh"] = round(r_thr, 4)
        u_thr = float(np.median(u))
        v_thr = float(np.quantile(v, 0.60))

        cand = live[(r >= r_thr) & (u >= u_thr) & (v >= v_thr)]
        report["candidates"] = int(len(cand))
        if not len(cand):
            return report
        cand = cand[np.argsort(-self.value(cand))][:max_splits]

        for i in cand:
            i = int(i)
            ids = [self._ex_get(i, j) for j in range(int(self.ex_fill[i]))]
            ids = [x for x in ids if x != EMPTY]
            if len(ids) < 4:
                continue
            vecs = fetch_vecs(ids)
            rows = [(j, vecs[j]) for j in ids if vecs.get(j) is not None]
            if len(rows) < 4:
                continue
            self.counters["split_tried"] += 1
            if self._split(i, [j for j, _ in rows],
                           np.stack([v_ for _, v_ in rows]).astype(np.float32)):
                report["split"] += 1
            else:
                self.counters["split_failed"] += 1
        report["merged"] = self.counters["merges"]
        report["evicted"] = self.counters["evictions"]
        return report

    def prune(self) -> int:
        live = np.flatnonzero(self.alive[: self.hi])
        if not len(live):
            return 0
        dead = live[self.weight(live) < self.min_weight]
        for i in dead:
            self._release(int(i))
        self.counters["prunes"] += len(dead)
        return int(len(dead))

    def _split(self, i: int, ids: list[int], V: np.ndarray) -> bool:
        V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        a, b = self._farthest_pair(V)
        ca, cb = V[a].copy(), V[b].copy()
        lab = np.zeros(len(V), dtype=int)
        for _ in range(12):
            lab = (V @ cb > V @ ca).astype(int)
            if lab.sum() in (0, len(V)):
                return False
            ca = V[lab == 0].mean(0); ca /= max(np.linalg.norm(ca), 1e-12)
            cb = V[lab == 1].mean(0); cb /= max(np.linalg.norm(cb), 1e-12)

        if min(int((lab == 0).sum()), int((lab == 1).sum())) < 2:
            return False
        try:
            slot = self._claim_slot()
        except RuntimeError:
            return False

        n_tot = float(self.n[i])
        util = float(self.util[i])
        t_off = float(self.t_off[i])
        m = float(len(V))
        for dst, mask in ((i, lab == 0), (slot, lab == 1)):
            sub = V[mask]
            share = float(mask.sum()) / m
            mu = sub.mean(0)
            self.LS[dst] = (n_tot * share) * mu
            self.n[dst] = n_tot * share
            ln = float(np.linalg.norm(self.LS[dst]))
            self.C[dst] = (self.LS[dst] / ln) if ln > 0 else mu
            self.util[dst] = util
            self.hits[dst] = 0.0
            self.t_off[dst] = t_off
            self.alive[dst] = True
            # The child keeps only pointers it actually owns. dst==i retains
            # any overflow block; the new slot starts on the base tier and
            # must earn its own.
            picked = [int(ids[j]) for j in np.flatnonzero(mask)][: self._ex_width(dst)]
            self.EX[dst, :] = EMPTY
            if self.X_E and self.x_of[dst] >= 0:
                self.XE[int(self.x_of[dst]), :] = EMPTY
            for j, v in enumerate(picked):
                self._ex_set(dst, j, v)
            self.ex_fill[dst] = len(picked)
            self.ex_seen[dst] = len(picked)
            self.ex_keep[dst] = 0
        self.counters["splits"] += 1
        return True

    @staticmethod
    def _farthest_pair(V: np.ndarray) -> tuple[int, int]:
        a = int(np.argmin(V @ V.mean(0)))
        b = int(np.argmin(V @ V[a]))
        return a, b

    def _make_room(self) -> bool:
        """Merge the cheapest compatible pair; fall back to eviction."""
        live = np.flatnonzero(self.alive[: self.hi])
        if len(live) < 2:
            return False
        pool = live[np.argsort(self.value(live))][: min(self.merge_pool, len(live))]
        Cp = self.C[pool].astype(np.float32)
        S = Cp @ Cp.T
        np.fill_diagonal(S, -2.0)
        order = np.dstack(np.unravel_index(np.argsort(-S, axis=None), S.shape))[0]
        for a_i, b_i in order[: 4 * len(pool)]:
            a, b = int(pool[a_i]), int(pool[b_i])
            if a == b:
                continue
            ls = self.LS[a] + self.LS[b]
            nn = self.n[a] + self.n[b]
            mu = float(np.linalg.norm(ls)) / max(nn, 1e-12)
            if math.sqrt(max(0.0, 1.0 - mu * mu)) <= self.radius_max:
                self._merge(a, b)
                return True
        self._release(int(live[np.argmin(self.value(live))]))
        self.counters["evictions"] += 1
        return True

    def _merge(self, a: int, b: int) -> None:
        wa, wb = float(self.n[a]), float(self.n[b])
        self.LS[a] += self.LS[b]
        self.n[a] = wa + wb
        ln = float(np.linalg.norm(self.LS[a]))
        if ln > 0:
            self.C[a] = self.LS[a] / ln
        tot = max(wa + wb, 1e-12)
        self.util[a] = (self.util[a] * wa + self.util[b] * wb) / tot
        self.hits[a] += self.hits[b]
        self.t_off[a] = max(float(self.t_off[a]), float(self.t_off[b]))

        wid = self._ex_width(a)
        keep = [self._ex_get(a, j) for j in range(int(self.ex_fill[a]))][: wid // 2]
        keep += [self._ex_get(b, j) for j in range(int(self.ex_fill[b]))][
            : wid - len(keep)]
        keep = [v for v in keep if v != EMPTY][:wid]
        self.EX[a, :] = EMPTY
        if self.X_E and self.x_of[a] >= 0:
            self.XE[int(self.x_of[a]), :] = EMPTY
        for j, v in enumerate(keep):
            self._ex_set(a, j, v)
        self.ex_fill[a] = len(keep)
        self.ex_keep[a] = 0
        self.ex_seen[a] = min(int(self.ex_seen[a]) + int(self.ex_seen[b]), _I32_MAX)
        self._release(b)
        self.counters["merges"] += 1

    def _release(self, i: int) -> None:
        if self.X_E and self.x_of[i] >= 0:
            self._revoke_block(int(self.x_of[i]))
        self.alive[i] = False
        self.LS[i] = 0.0
        self.C[i] = 0.0
        self.n[i] = 0.0
        self.EX[i, :] = EMPTY
        self.ex_seen[i] = 0
        self.ex_fill[i] = 0
        self.ex_keep[i] = 0
        self.free.append(i)

    # ------------------------------------------------------------------ i/o

    def stats(self) -> dict:
        live = np.flatnonzero(self.alive[: self.hi])
        r = self.radius(live) if len(live) else np.zeros(1)
        addressable = self.addressable()
        return {
            "clusters": int(len(live)),
            "capacity": self.capacity,
            "resident_bytes": self.resident_bytes(),
            "resident_mb": round(self.resident_bytes() / 1e6, 2),
            "bytes_per_cluster": self.bytes_per_cluster(
                self.dim, self.E, self.X_E, self.extra_frac),
            "exemplars": self.E,
            "extra_exemplars": self.X_E,
            "blocks_used": int((self.x_owner >= 0).sum()) if self.n_blocks else 0,
            "blocks_total": self.n_blocks,
            "ex_slots": int(sum(self._ex_width(int(i)) for i in live)),
            "addressable": addressable,
            "mean_radius": round(float(r.mean()), 4),
            "p80_radius": round(float(np.quantile(r, 0.80)), 4),
            "split_thresh": round(float(self._split_thresh), 4),
            "mean_util": round(float(self.util[live].mean()), 4) if len(live) else 0.0,
            "util_spread": (round(float(self.util[live].std()), 4)
                            if len(live) else 0.0),
            "total_weight": (round(float(self.weight(live).sum()), 2)
                             if len(live) else 0.0),
            **self.counters,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path, LS=self.LS, C=self.C, n=self.n, util=self.util, hits=self.hits,
            t_off=self.t_off, alive=self.alive, EX=self.EX,
            ex_seen=self.ex_seen, ex_fill=self.ex_fill, ex_keep=self.ex_keep,
            XE=self.XE, x_owner=self.x_owner, x_of=self.x_of,
            meta=np.frombuffer(json.dumps({
                "v": 2,
                "dim": self.dim, "capacity": self.capacity, "alpha": self.alpha,
                "tau_min": self.tau_min, "tau_max": self.tau_max, "lam": self.lam,
                "E": self.E, "X_E": self.X_E, "extra_frac": self.extra_frac,
                "grant_util": self.grant_util, "protect_frac": self.protect_frac,
                "radius_max": self.radius_max,
                "split_quantile": self.split_quantile,
                "split_radius": self.split_radius, "min_weight": self.min_weight,
                "util_eta": self.util_eta, "merge_pool": self.merge_pool,
                "hi": self.hi, "free": self.free, "iw": self.iw,
                "t_ref": self.t_ref, "origin": self.origin,
                "x_free": self.x_free, "counters": self.counters,
            }).encode(), dtype=np.uint8),
        )

    @classmethod
    def load(cls, path: str) -> "CentroidCloud":
        z = np.load(path, allow_pickle=False)
        m = json.loads(bytes(z["meta"]).decode())
        if int(m.get("v", 1)) < 2:
            raise ValueError(
                "cloud.npz was written by core v1 (float32 centroids, int64 "
                "pointers). The layout changed; re-encode from the cold log.")
        c = cls(m["dim"], capacity=m["capacity"], alpha=m["alpha"],
                tau_min=m["tau_min"], tau_max=m["tau_max"], half_life_s=None,
                exemplars=m["E"], extra_exemplars=m.get("X_E", 0),
                extra_frac=m.get("extra_frac", 0.0),
                grant_util=m.get("grant_util", 0.60),
                protect_frac=m.get("protect_frac", 0.50),
                radius_max=m["radius_max"],
                split_quantile=m.get("split_quantile", 0.80),
                split_radius=m.get("split_radius"),
                min_weight=m["min_weight"],
                util_eta=m["util_eta"], merge_pool=m["merge_pool"])
        c.lam = m["lam"]
        for k in ("LS", "C", "n", "util", "hits", "t_off", "alive", "EX",
                  "ex_seen", "ex_fill", "ex_keep", "XE", "x_owner", "x_of"):
            if k in z:
                getattr(c, k)[...] = z[k]
        c.hi, c.free, c.iw = m["hi"], list(m["free"]), m["iw"]
        c.t_ref, c.origin = m["t_ref"], m.get("origin", 0.0)
        c.x_free = list(m.get("x_free", c.x_free))
        c.counters.update(m["counters"])
        return c