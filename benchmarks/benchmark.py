#!/usr/bin/env python3
"""NIMBUS benchmark harness, v3.

    python -m benchmarks.benchmark --quick --embedder st
    python -m benchmarks.benchmark --items 50000 --queries 5000 --embedder st --core
    python -m benchmarks.benchmark --items 50000 --core --seeds 7,8,9
    python -m benchmarks.benchmark --items 50000 --core --sweep-exemplars 16,32,64,128
    python -m benchmarks.benchmark --items 50000 --core --sweep-budgets 5e5,1e6,2e6,4e6,8e6

READ THIS BEFORE QUOTING ANY NUMBER
-----------------------------------
Synthetic benchmark written by the authors of the system under test. It is a
regression harness and a hypothesis test, not evidence. Published claims come
from LongMemEval (arXiv:2410.10813) and LoCoMo (arXiv:2402.17753) with an LLM
judge.

WHAT WAS WRONG WITH v2
----------------------
1. TIME LEAK. `one_pass` pre-filled the shared ColdLog with the ENTIRE stream
   before running a system, so `cold.bm25()` and `cold.recent()` returned items
   from the future. `bm25-unbounded` and `vector-fifo+bm25` were scoring
   against a 50k corpus at item 300 and their recency lists pinned items that
   had not been written yet. Those rows were void. The log is now appended
   inside `run_system`, one item at a time, in stream order.

2. STALE BYTE ACCOUNTING. The nimbus capacity solver hardcoded
   `dim*8 + 33 + 8*exemplars`, which stopped matching the actual layout. It now
   calls `CentroidCloud.bytes_per_cluster`, so the budget the harness charges
   and the bytes the cloud allocates cannot diverge.

3. ASYMMETRIC RECENCY PIN. The baseline hybrid added a flat +1.0 to its
   recency list, same as the old cloud read path, so both were burning half a
   400-token context on the newest six items. Now `--recency-pin` (default 0)
   applies to every system identically.

4. CIs THAT ASSUMED INDEPENDENCE. Zipf load means the same neighbourhoods and
   the same gold facts recur hundreds of times; treating 4,250 queries as
   independent understates the interval badly. Paired deltas now also report a
   bootstrap resampled over NEIGHBOURHOODS, and `--seeds` pools runs.

5. VACUOUS hard@ctx. When 97% of queries are "hard", hard@ctx is hit@ctx with
   extra steps. The verdict now says so instead of reporting a pass.

THE COMPARISON THAT MATTERS
---------------------------
At equal charged resident bytes, on paraphrased queries, over dense
neighbourhoods:

    nimbus-cloud   vs  bm25-capped              clustering vs bounded lexical
    nimbus-cloud   vs  vector-fifo              clustering vs flat sampling
    nimbus-cloud   vs  nimbus-cloud-no-credit   is the credit policy load-bearing

METRICS
-------
hit@ctx      gold fact in context AFTER hard budget truncation
para@ctx     hit@ctx restricted to PARAPHRASED queries (the semantic task)
lex@ctx      hit@ctx restricted to surface-matching queries (the lexical task)
agg_recall   fraction of a multi-fact gold SET recovered (coverage, not lookup)
mrr          mean reciprocal rank over pre-budget candidates
resident_mb  RAM required to route a query, INCLUDING any inverted index
addr_frac    items reachable through a pointer / items ingested. The ceiling.
stale_err    superseded fact ranked above its replacement
llm_calls    write-path LLM calls. Ours is structurally zero.

Requires: numpy, nimbus. Optional: matplotlib, sentence-transformers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from nimbus import HashEmbedder, Nimbus          # noqa: E402
from nimbus.core import CentroidCloud            # noqa: E402

WORD = re.compile(r"[A-Za-z0-9_#$.\-]+")
CHARS_PER_TOKEN = 4

N_CAND_DEFAULT = 200
N_CAND = N_CAND_DEFAULT


# ======================================================== 1. paired vocabulary
#
# Each pair is (fact_form, query_form): semantically close, sharing NO surface
# token. Facts use fact_forms, paraphrased queries use query_forms, so lexical
# search gets no exact-match handle on the entity and embedding search does.
# Entity identity is the TRIPLE, so 20^3 = 8000 distinguishable entities exist
# and siblings differing in one slot are the hard negatives.

REGION_PAIRS = [
    ("nordic", "scandinavian"), ("alpine", "mountain"), ("coastal", "seaside"),
    ("desert", "arid"), ("urban", "metropolitan"), ("rural", "countryside"),
    ("arctic", "polar"), ("tropical", "equatorial"), ("midland", "inland"),
    ("harbour", "port"), ("prairie", "grassland"), ("delta", "estuary"),
    ("highland", "upland"), ("lakeside", "lakefront"), ("border", "frontier"),
    ("island", "offshore"), ("valley", "basin"), ("plateau", "tableland"),
    ("forest", "woodland"), ("canyon", "gorge"),
]

SECTOR_PAIRS = [
    ("hospital", "healthcare"), ("bank", "financial"), ("airline", "aviation"),
    ("brewery", "beverage"), ("foundry", "metalworks"), ("publisher", "printing"),
    ("shipyard", "maritime"), ("pharmacy", "drugstore"), ("winery", "vineyard"),
    ("bakery", "patisserie"), ("laundry", "cleaning"), ("quarry", "mining"),
    ("orchard", "fruit"), ("tannery", "leather"), ("smelter", "refining"),
    ("dairy", "creamery"), ("cannery", "preserving"), ("sawmill", "lumber"),
    ("textile", "garment"), ("glassworks", "ceramics"),
]

ROLE_PAIRS = [
    ("procurement", "purchasing"), ("logistics", "shipping"),
    ("compliance", "regulatory"), ("payroll", "salary"),
    ("maintenance", "upkeep"), ("training", "onboarding"),
    ("archive", "records"), ("catering", "food"), ("security", "guarding"),
    ("insurance", "coverage"), ("audit", "inspection"),
    ("translation", "localisation"), ("recruiting", "staffing"),
    ("licensing", "permits"), ("warranty", "guarantee"),
    ("packaging", "wrapping"), ("calibration", "tuning"),
    ("disposal", "waste"), ("relocation", "moving"),
    ("subscription", "membership"),
]

N_REGION, N_SECTOR, N_ROLE = len(REGION_PAIRS), len(SECTOR_PAIRS), len(ROLE_PAIRS)

TOPICS = [
    ("billing", "invoice", ["finance", "payment", "account", "charge"]),
    ("deploy", "release", ["ship", "rollout", "server", "cluster"]),
    ("auth", "token", ["login", "session", "credential", "oauth"]),
    ("database", "migration", ["schema", "index", "query", "postgres"]),
    ("support", "ticket", ["customer", "escalation", "sla", "refund"]),
    ("legal", "contract", ["clause", "dpa", "renewal", "liability"]),
]

FILLER = [
    "quick sync on the thing we discussed",
    "moved the meeting to thursday afternoon",
    "reviewed the doc, left a few comments inline",
    "standup was short today, no blockers raised",
    "kicked off the weekly report, will circulate",
    "closed the loop with the other team on scope",
]


def fact_phrase(e: tuple[int, int, int]) -> str:
    r, s, o = e
    return f"{REGION_PAIRS[r][0]} {SECTOR_PAIRS[s][0]} {ROLE_PAIRS[o][0]}"


def query_phrase(e: tuple[int, int, int]) -> str:
    r, s, o = e
    return (f"{ROLE_PAIRS[o][1]} for the {SECTOR_PAIRS[s][1]} "
            f"{REGION_PAIRS[r][1]} client")


# ============================================================ 2. workload


@dataclass
class Item:
    id: int
    ts: float
    text: str
    topic: int
    key: str | None = None
    entity: tuple[int, int, int] | None = None
    nbhd: tuple[int, int] | None = None
    superseded_by: str | None = None


@dataclass
class Query:
    text: str
    kind: str                      # "lookup" | "update" | "aggregate"
    gold_keys: list[str]
    gold_text: str
    stale_key: str | None
    entity: tuple[int, int, int] | None
    nbhd: tuple[int, int]
    paraphrased: bool
    hot: bool = False
    hard: bool = False
    at_item: int = 0
    gold_at: int = 0


class Workload:
    """Dense-neighbourhood, paraphrase-addressed, Zipf-queried stream.

    NEIGHBOURHOODS. An entity is a (region, sector, role) triple; the
    neighbourhood is (region, sector). Siblings share it and differ only in
    role, so a neighbourhood with 15 live siblings is 15 mutually confusable
    facts in a tiny ball of embedding space. Resolving them requires fine
    resolution *there*.

    PARAPHRASE ADDRESSING. Facts use fact_forms, queries use query_forms,
    surface-disjoint. --paraphrase-frac mixes in surface queries so the lexical
    control is not strawmanned; para@ctx and lex@ctx are reported separately.

    ZIPF QUERY LOAD. A few neighbourhoods are hit 50x and most once. This is
    the only regime in which a utility allocator can be right.

    AGGREGATES. "list everything for <neighbourhood>" needs COVERAGE of a
    region rather than one point lookup.
    """

    def __init__(self, n_items: int, n_queries: int, seed: int = 7,
                 filler_ratio: float = 0.45, update_ratio: float = 0.20,
                 zipf_s: float = 1.10, span_days: float = 350.0,
                 paraphrase_frac: float = 0.70, aggregate_frac: float = 0.15,
                 probe_cutoff_frac: float = 0.15, hot_top_frac: float = 0.20,
                 hard_min_siblings: int = 8, nbhd_zipf_s: float = 1.05):
        self.rng = random.Random(seed)
        self.seed = int(seed)
        self.items: list[Item] = []
        self.queries: list[Query] = []
        self.probes: list[Query] = []
        self.span_days = float(span_days)
        self.paraphrase_frac = float(paraphrase_frac)

        nbhds = [(r, s) for r in range(N_REGION) for s in range(N_SECTOR)]
        self.rng.shuffle(nbhds)
        w = np.array([1.0 / (i + 1) ** nbhd_zipf_s for i in range(len(nbhds))])
        self.nbhd_p = w / w.sum()
        self.nbhds = nbhds

        t0 = time.time() - (span_days + 15) * 86400
        dt = (span_days * 86400) / max(n_items, 1)

        live: dict[tuple[int, int, int], Item] = {}
        by_nbhd: dict[tuple[int, int], list[Item]] = defaultdict(list)
        kc = 0
        pend_lookup: list[Query] = []
        cum = np.cumsum(self.nbhd_p)

        for i in range(n_items):
            ts = t0 + i * dt
            topic = self.rng.randrange(len(TOPICS))
            name, noun, vocab = TOPICS[topic]

            if self.rng.random() < filler_ratio:
                self.items.append(
                    Item(i, ts, f"{self.rng.choice(FILLER)} ({name})", topic))
                continue

            ni = int(np.searchsorted(cum, self.rng.random()))
            nb = nbhds[min(ni, len(nbhds) - 1)]
            role = self.rng.randrange(N_ROLE)
            ent = (nb[0], nb[1], role)

            kc += 1
            key = f"{noun.upper()[:3]}-{kc:06d}"
            fp = fact_phrase(ent)

            prev = live.get(ent)
            if prev is not None and self.rng.random() < update_ratio:
                prev.superseded_by = key
                txt = (f"update: the {fp} {noun} {prev.key} is superseded, "
                       f"the current {noun} is {key}, "
                       f"{self.rng.choice(vocab)} reassigned")
                kind, stale = "update", prev.key
            else:
                txt = (f"{name}: the {noun} of record for the {fp} account is "
                       f"{key}, {self.rng.choice(vocab)} set to "
                       f"{self.rng.randint(2, 96)}")
                kind, stale = "lookup", None

            it = Item(i, ts, txt, topic, key=key, entity=ent, nbhd=nb)
            self.items.append(it)
            live[ent] = it
            by_nbhd[nb].append(it)
            pend_lookup.append(Query(
                text="", kind=kind, gold_keys=[key], gold_text=txt,
                stale_key=stale, entity=ent, nbhd=nb, paraphrased=False,
                at_item=0, gold_at=i))

        self.live = live
        self.by_nbhd = by_nbhd
        self.sib_count = {nb: len({x.entity for x in v})
                          for nb, v in by_nbhd.items()}

        eligible = [q for q in pend_lookup if q.gold_at < n_items - 30]
        if not eligible:
            raise SystemExit("stream too short to schedule any query")
        by_nb_q: dict[tuple[int, int], list[Query]] = defaultdict(list)
        for q in eligible:
            by_nb_q[q.nbhd].append(q)
        keys = list(by_nb_q)
        kw = np.array([1.0 / (i + 1) ** zipf_s for i in range(len(keys))])
        kw /= kw.sum()
        kcum = np.cumsum(kw)

        n_agg = int(aggregate_frac * n_queries)
        n_look = max(0, n_queries - n_agg)
        chosen: list[Query] = []
        nb_hits: dict[tuple[int, int], int] = defaultdict(int)

        for _ in range(n_look):
            nb = keys[int(np.searchsorted(kcum, self.rng.random())) % len(keys)]
            src = self.rng.choice(by_nb_q[nb])
            at = (self.rng.randint(src.gold_at + 20, n_items - 1)
                  if src.gold_at + 20 < n_items - 1 else n_items - 1)
            if at <= src.gold_at:
                continue
            chosen.append(self._instantiate(src, at))
            nb_hits[nb] += 1

        for _ in range(n_agg):
            nb = keys[int(np.searchsorted(kcum, self.rng.random())) % len(keys)]
            members = by_nb_q[nb]
            if len(members) < 2:
                continue
            latest = {}
            for m in members:
                latest[m.entity] = m
            picks = list(latest.values())[:6]
            gold_at = max(p.gold_at for p in picks)
            if gold_at + 20 >= n_items - 1:
                continue
            at = self.rng.randint(gold_at + 20, n_items - 1)
            chosen.append(self._make_aggregate(nb, picks, at, gold_at))
            nb_hits[nb] += 1

        if not chosen:
            raise SystemExit("no queries scheduled; raise --items")

        cut = sorted(nb_hits.values(), reverse=True)
        hot_i = max(0, int(hot_top_frac * len(cut)) - 1)
        hot_thresh = cut[hot_i] if cut else 10 ** 9
        hot_set = {nb for nb, c in nb_hits.items() if c >= max(2, hot_thresh)}
        for q in chosen:
            q.hot = q.nbhd in hot_set
            q.hard = self.sib_count.get(q.nbhd, 0) >= hard_min_siblings

        self.queries = sorted(chosen, key=lambda q: q.at_item)
        self.nb_hits = dict(nb_hits)
        self.hot_nbhds = hot_set
        self.hard_frac = (sum(1 for q in self.queries if q.hard)
                          / max(len(self.queries), 1))

        cutoff = max(1, int(probe_cutoff_frac * n_items))
        self.probe_cutoff = cutoff
        used = {(tuple(q.gold_keys), q.at_item) for q in chosen}
        pool = [q for q in eligible
                if q.gold_at <= cutoff
                and (tuple(q.gold_keys), q.at_item) not in used]
        self.probes = [self._instantiate(q, n_items - 1) for q in pool[:200]]

    def _instantiate(self, src: Query, at: int) -> Query:
        para = self.rng.random() < self.paraphrase_frac
        return Query(text=self._q_text(src, para), kind=src.kind,
                     gold_keys=list(src.gold_keys), gold_text=src.gold_text,
                     stale_key=src.stale_key, entity=src.entity,
                     nbhd=src.nbhd, paraphrased=para, at_item=at,
                     gold_at=src.gold_at)

    def _make_aggregate(self, nb, picks, at, gold_at) -> Query:
        para = self.rng.random() < self.paraphrase_frac
        r, s = nb
        if para:
            who = f"the {SECTOR_PAIRS[s][1]} {REGION_PAIRS[r][1]} client"
        else:
            who = f"the {REGION_PAIRS[r][0]} {SECTOR_PAIRS[s][0]} account"
        return Query(
            text=f"list every reference we have on file for {who}, all areas",
            kind="aggregate", gold_keys=[p.gold_keys[0] for p in picks],
            gold_text="", stale_key=None, entity=None, nbhd=nb,
            paraphrased=para, at_item=at, gold_at=gold_at)

    def _q_text(self, src: Query, para: bool) -> str:
        ent = src.entity
        noun = "reference"
        for _, n, _ in TOPICS:
            if n.upper()[:3] == src.gold_keys[0][:3]:
                noun = n
                break
        who = query_phrase(ent) if para else f"{fact_phrase(ent)} account"
        if src.kind == "update":
            return f"what is the current {noun} for {who} after the change?"
        return f"remind me of the {noun} we recorded for {who}"


# ============================================ 3. cold log + indexes


class ColdLog:
    """Append-only shared store, built IN STREAM ORDER by the runner.

    Text lives on disk for every system, which is fair. The routing structure
    is not free: index_bytes() is charged to any system with uses_index=True
    when --count-index-bytes is on (the default).

    v2 pre-filled this object with the whole stream before running a system,
    which let `bm25()` and `recent()` see the future. Never pre-fill it."""

    def __init__(self) -> None:
        self.by_id: dict[int, Item] = {}
        self.order: list[int] = []
        self.inv: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.dl: dict[int, int] = {}
        self.total_len = 0

    def add(self, it: Item) -> None:
        if it.id in self.by_id:
            return
        self.by_id[it.id] = it
        self.order.append(it.id)
        toks = [t.lower() for t in WORD.findall(it.text)]
        tf: dict[str, int] = defaultdict(int)
        for t in toks:
            tf[t] += 1
        for t, c in tf.items():
            self.inv[t].append((it.id, c))
        self.dl[it.id] = len(toks)
        self.total_len += len(toks)

    def text(self, i: int) -> str:
        it = self.by_id.get(i)
        return it.text if it else ""

    def index_bytes(self) -> int:
        posts = sum(len(p) for p in self.inv.values())
        return posts * 16 + len(self.dl) * 16 + sum(len(t) + 8 for t in self.inv)

    def bm25(self, query: str, k: int | None = None, k1: float = 1.5,
             b: float = 0.75) -> list[int]:
        k = N_CAND if k is None else k
        toks = {t.lower() for t in WORD.findall(query)}
        if not toks or not self.order:
            return []
        N = len(self.order)
        avg = self.total_len / N
        scores: dict[int, float] = defaultdict(float)
        for t in toks:
            post = self.inv.get(t)
            if not post or len(post) > N * 0.3:
                continue
            idf = math.log(1 + (N - len(post) + 0.5) / (len(post) + 0.5))
            for did, tf in post:
                dl = self.dl[did]
                scores[did] += idf * (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * dl / avg))
        return [i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:k]]

    def recent(self, k: int) -> list[int]:
        if k <= 0:
            return []
        return list(reversed(self.order[-k:]))


class CappedIndex:
    """Inverted index with a hard RAM ceiling, evicting oldest documents.

    The correct lexical control under a fixed memory budget must FORGET,
    exactly as the centroid cloud must. Byte accounting is now incremental:
    v2 recomputed the whole index size on every eviction, which is O(index)
    per evicted document and dominated the run."""

    def __init__(self, byte_budget: int) -> None:
        self.budget = int(byte_budget)
        self.inv: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.dl: dict[int, int] = {}
        self.docs: deque[int] = deque()
        self.doc_toks: dict[int, list[str]] = {}
        self.total_len = 0
        self.bytes = 0
        self.evicted = 0

    def add(self, it: Item) -> None:
        toks = [t.lower() for t in WORD.findall(it.text)]
        tf: dict[str, int] = defaultdict(int)
        for t in toks:
            tf[t] += 1
        delta = 16                                  # dl entry
        for t, c in tf.items():
            if t not in self.inv:
                delta += len(t) + 8                 # new term key
            self.inv[t].append((it.id, c))
            delta += 16                             # posting
        self.dl[it.id] = len(toks)
        self.doc_toks[it.id] = list(tf)
        self.docs.append(it.id)
        self.total_len += len(toks)
        self.bytes += delta
        while self.bytes > self.budget and len(self.docs) > 1:
            self._evict()

    def _evict(self) -> None:
        d = self.docs.popleft()
        delta = 16
        for t in self.doc_toks.pop(d, ()):
            post = self.inv.get(t)
            if post is None:
                continue
            keep = [p for p in post if p[0] != d]
            delta += 16 * (len(post) - len(keep))
            if keep:
                self.inv[t] = keep
            else:
                del self.inv[t]
                delta += len(t) + 8
        self.total_len -= self.dl.pop(d, 0)
        self.bytes = max(0, self.bytes - delta)
        self.evicted += 1

    def resident_bytes(self) -> int:
        posts = sum(len(p) for p in self.inv.values())
        return posts * 16 + len(self.dl) * 16 + sum(len(t) + 8 for t in self.inv)

    def bm25(self, query: str, k: int, k1: float = 1.5, b: float = 0.75):
        toks = {t.lower() for t in WORD.findall(query)}
        N = len(self.dl)
        if not toks or not N:
            return []
        avg = max(self.total_len / N, 1e-9)
        scores: dict[int, float] = defaultdict(float)
        for t in toks:
            post = self.inv.get(t)
            if not post or len(post) > N * 0.3:
                continue
            idf = math.log(1 + (N - len(post) + 0.5) / (len(post) + 0.5))
            for did, tf in post:
                dl = self.dl.get(did, 1)
                scores[did] += idf * (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * dl / avg))
        return [i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:k]]


# ============================================ 4. system interface


@dataclass
class ReadResult:
    block: str
    ranked: list[str]
    tokens: int = 0
    handle: object = None


class System(Protocol):
    name: str
    fixed_footprint: bool
    uses_index: bool
    def write(self, it: Item) -> None: ...
    def read(self, q: str, budget_tokens: int) -> ReadResult: ...
    def credit(self, res: ReadResult, hit: bool, gold: str, rank: int) -> None: ...
    def resident_bytes(self) -> int: ...
    def llm_calls(self) -> int: ...
    def capacity_info(self) -> dict: ...
    def close(self) -> None: ...


def pack(texts: Sequence[str], budget_tokens: int) -> str:
    cap = budget_tokens * CHARS_PER_TOKEN
    lines: list[str] = []
    used, seen = 0, set()
    for t in texts:
        if not t:
            continue
        norm = re.sub(r"\W+", "", t.lower())[:120]
        if norm in seen:
            continue
        seen.add(norm)
        line = f"[m{len(lines) + 1}] {t}"
        room = cap - used
        if len(line) + 1 > room:
            if not lines and room > 48:
                lines.append(line[:room])
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def pack_ids(ids: Sequence[int], cold: ColdLog, budget_tokens: int) -> str:
    return pack([cold.text(i) for i in ids], budget_tokens)


def weighted_rrf(votes: Sequence[tuple[Sequence[int], float]],
                 pin: Sequence[int] = (), rrf_k: float = 60.0,
                 k: int | None = None) -> list[int]:
    """Shared fusion so every system gets identical rank arithmetic. `pin`
    force-promotes ids to the front; pass an empty list for no pin."""
    fused: dict[int, float] = defaultdict(float)
    for lst, w in votes:
        if w <= 0.0:
            continue
        for r, i in enumerate(lst):
            fused[i] += w / (rrf_k + r + 1)
    order = [i for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
    if pin:
        ps = list(dict.fromkeys(pin))
        order = ps + [i for i in order if i not in set(ps)]
    return order[:k] if k else order


class Base:
    fixed_footprint = True
    uses_index = False
    charged_index_bytes = 0
    def credit(self, res, hit, gold, rank): return None
    def llm_calls(self) -> int: return 0
    def capacity_info(self) -> dict: return {}
    def close(self) -> None: return None


# ============================================ 5. baselines


class RecencyWindow(Base):
    """Buffer-window memory. Charged for retained TEXT, not integer ids."""
    name = "recency-window"

    def __init__(self, cold: ColdLog, byte_budget: int, **kw):
        self.cold = cold
        self.budget = int(byte_budget)
        self.buf: deque[tuple[int, int]] = deque()
        self.bytes = 0
        self.evicted = 0

    def write(self, it):
        b = len(it.text.encode()) + 8
        self.buf.append((it.id, b))
        self.bytes += b
        while self.bytes > self.budget and len(self.buf) > 1:
            _, ob = self.buf.popleft()
            self.bytes -= ob
            self.evicted += 1

    def read(self, q, budget_tokens):
        ids = [i for i, _ in reversed(self.buf)]
        return ReadResult(pack_ids(ids, self.cold, budget_tokens),
                          [self.cold.text(i) for i in ids[:N_CAND]])

    def resident_bytes(self): return self.bytes

    def capacity_info(self):
        return {"slots": len(self.buf), "cap": len(self.buf) + self.evicted,
                "bound": self.evicted > 0}


class Bm25Unbounded(Base):
    """Lexical search over the full log SO FAR. Not budget-limited: reported so
    you can see the accuracy an unbounded index buys and what it costs in
    resident bytes. Its resident_mb grows without limit; that is the row."""
    name = "bm25-unbounded"
    uses_index = True
    fixed_footprint = False

    def __init__(self, cold: ColdLog, byte_budget: int, **kw):
        self.cold = cold

    def write(self, it): return None      # the runner appends to the shared log

    def read(self, q, budget_tokens):
        ids = self.cold.bm25(q, k=N_CAND)
        return ReadResult(pack_ids(ids, self.cold, budget_tokens),
                          [self.cold.text(i) for i in ids])

    def resident_bytes(self): return 0
    def capacity_info(self): return {"slots": 0, "cap": 0, "bound": False}


class Bm25Capped(Base):
    """*** THE CONTROL THAT CAN KILL THE PROJECT ***

    Bounded lexical memory at the SAME byte budget as the cloud. If this ties
    us on para@ctx at equal bytes, the centroid cloud is decoration and the
    honest write-up says so."""
    name = "bm25-capped"

    def __init__(self, cold: ColdLog, byte_budget: int, **kw):
        self.cold = cold
        self.idx = CappedIndex(byte_budget)

    def write(self, it): self.idx.add(it)

    def read(self, q, budget_tokens):
        ids = self.idx.bm25(q, k=N_CAND)
        return ReadResult(pack_ids(ids, self.cold, budget_tokens),
                          [self.cold.text(i) for i in ids])

    def resident_bytes(self): return self.idx.resident_bytes()

    def capacity_info(self):
        return {"slots": len(self.idx.dl),
                "cap": len(self.idx.dl) + self.idx.evicted,
                "bound": self.idx.evicted > 0}


class VectorStore(Base):
    """Flat cosine index. policy='none' is the unbounded RAG baseline;
    'fifo'/'reservoir' are capped at the same byte budget as the cloud.

    A flat store gets ~2x the slots of a cloud at equal bytes, because a
    cluster carries LS + centroid + pointers. We have to win from behind.
    hybrid=True grants the SAME bm25+vector+recency fusion and the SAME
    recency pin the cloud gets -- symmetry here is not optional, since an
    asymmetric pin silently decides the comparison."""

    def __init__(self, cold: ColdLog, byte_budget: int, embedder,
                 policy: str = "fifo", hybrid: bool = False, seed: int = 0,
                 recency_k: int = 6, recency_pin: int = 0, **kw):
        self.cold, self.emb, self.policy, self.hybrid = cold, embedder, policy, hybrid
        self.d = embedder.dim
        self.slot_bytes = self.d * 4 + 8
        self.cap = (10 ** 9 if policy == "none"
                    else max(64, byte_budget // self.slot_bytes))
        first = min(self.cap, 4096)
        self.V = np.zeros((first, self.d), dtype=np.float32)
        self.ids = np.zeros(first, dtype=np.int64)
        self.n = 0
        self.seen = 0
        self.evicted = 0
        self.recency_k = int(recency_k)
        self.recency_pin = int(recency_pin)
        self.rng = random.Random(seed)
        self.fixed_footprint = policy != "none"
        self.uses_index = hybrid
        self.name = {"none": "vector-unbounded", "fifo": "vector-fifo",
                     "reservoir": "vector-reservoir"}[policy]
        if hybrid:
            self.name += "+bm25"

    def _grow(self):
        if self.n < self.V.shape[0]:
            return
        new = min(self.cap, max(self.V.shape[0] * 2, 1024))
        if new == self.V.shape[0]:
            return
        V = np.zeros((new, self.d), dtype=np.float32); V[: self.n] = self.V[: self.n]
        I = np.zeros(new, dtype=np.int64); I[: self.n] = self.ids[: self.n]
        self.V, self.ids = V, I

    def write(self, it):
        v = self.emb.encode([it.text])[0]
        self.seen += 1
        self._grow()
        if self.n < self.V.shape[0]:
            self.V[self.n], self.ids[self.n] = v, it.id
            self.n += 1
            return
        self.evicted += 1
        if self.policy == "fifo":
            self.V[:-1] = self.V[1:]; self.ids[:-1] = self.ids[1:]
            self.V[-1], self.ids[-1] = v, it.id
        elif self.policy == "reservoir":
            j = self.rng.randrange(self.seen)
            if j < self.n:
                self.V[j], self.ids[j] = v, it.id

    def read(self, q, budget_tokens):
        if self.n == 0:
            return ReadResult("", [])
        qv = self.emb.encode([q])[0]
        s = self.V[: self.n] @ qv
        k = min(N_CAND, self.n)
        part = np.argpartition(-s, k - 1)[:k]
        ids = [int(self.ids[i]) for i in part[np.argsort(-s[part])]]
        if self.hybrid:
            lex = self.cold.bm25(q, k=N_CAND)
            rec = self.cold.recent(self.recency_k)
            ids = weighted_rrf([(ids, 1.0), (lex, 1.0), (rec, 1.0)],
                               pin=rec[: self.recency_pin], k=N_CAND)
        return ReadResult(pack_ids(ids, self.cold, budget_tokens),
                          [self.cold.text(i) for i in ids])

    def resident_bytes(self): return self.n * self.slot_bytes

    def capacity_info(self):
        return {"slots": self.n, "cap": (0 if self.policy == "none" else self.cap),
                "bound": self.evicted > 0}


class SummaryProxy(Base):
    """*** PROXY. NOT MEM0. NOT ZEP. DO NOT LABEL IT AS EITHER. ***
    Fixed slots of extractively-truncated text, modelling summarisation's tail
    loss without an LLM. slot_chars auto-sizes to the full byte budget."""
    name = "summary-proxy"

    def __init__(self, cold: ColdLog, byte_budget: int, slots: int = 64,
                 slot_chars: int = 0, corrupt_digits: float = 0.0,
                 seed: int = 0, **kw):
        self.cold = cold
        self.slots = slots
        self.slot_chars = (max(400, byte_budget // slots) if slot_chars <= 0
                           else int(slot_chars))
        self.S = [""] * slots
        self.calls = 0
        self.dropped = 0
        self.corrupt = corrupt_digits
        self.rng = random.Random(seed)

    def write(self, it):
        self.calls += 1
        j = (hash(it.nbhd) if it.nbhd else it.topic) % self.slots
        txt = it.text
        if self.corrupt and self.rng.random() < self.corrupt:
            txt = re.sub(r"\d", lambda _: str(self.rng.randint(0, 9)), txt, count=2)
        s = (self.S[j] + " | " + txt).strip(" |")
        if len(s) > self.slot_chars:
            s = s[-self.slot_chars:]
            self.dropped += 1
        self.S[j] = s

    def read(self, q, budget_tokens):
        toks = {t.lower() for t in WORD.findall(q)}
        chunks: list[tuple[int, str]] = []
        for s in self.S:
            if not s:
                continue
            for c in s.split(" | "):
                if c:
                    ov = len(toks & {t.lower() for t in WORD.findall(c)})
                    if ov:
                        chunks.append((ov, c))
        chunks.sort(key=lambda kv: -kv[0])
        texts = [c for _, c in chunks]
        return ReadResult(pack(texts, budget_tokens), texts[:N_CAND])

    def resident_bytes(self): return sum(len(s.encode()) for s in self.S)
    def llm_calls(self): return self.calls

    def capacity_info(self):
        return {"slots": sum(1 for s in self.S if s), "cap": self.slots,
                "bound": self.dropped > 0}


# ============================================ 6. nimbus + ablations


class NimbusAdapter(Base):
    """`lexical=False` ("nimbus-cloud") is the row that carries the claim: pure
    centroid routing, no inverted index, so nothing it does is secretly BM25.
    Ablations MUST be compared against the matching lexical setting.

    Capacity comes from CentroidCloud.bytes_per_cluster, not a hardcoded
    formula, so the charged budget and the allocated arrays cannot drift."""

    def __init__(self, cold: ColdLog, byte_budget: int, embedder,
                 credit: bool = True, lexical: bool = True,
                 label: str = "nimbus", half_life_s: float | None = None,
                 exemplars: int = 32, extra_exemplars: int = 0,
                 extra_frac: float = 0.0, maintain_every: int = 256,
                 commit_every: int = 256, recency_k: int = 6,
                 recency_pin: int = 0, coverage_weight: float = 1.0,
                 clusters: int = 8, lexical_k: int = 100,
                 cloud_kwargs: dict | None = None):
        self.cold, self.emb = cold, embedder
        d = embedder.dim
        E = int(exemplars)
        cap = CentroidCloud.capacity_for_bytes(
            byte_budget, d, E, extra_exemplars, extra_frac)
        self.per_cluster = CentroidCloud.bytes_per_cluster(
            d, E, extra_exemplars, extra_frac)
        self.dir = tempfile.mkdtemp(prefix=f"nimbus-bench-{label}-")
        self.mem = Nimbus(embedder=embedder, path=self.dir, capacity=cap,
                          exemplars=E, extra_exemplars=extra_exemplars,
                          extra_frac=extra_frac, maintain_every=maintain_every,
                          commit_every=commit_every, half_life_s=half_life_s,
                          recency_k=recency_k, recency_pin=recency_pin,
                          coverage_weight=coverage_weight,
                          **(cloud_kwargs or {}))
        self.use_credit, self.use_lex = credit, lexical
        self.uses_index = lexical
        self.clusters, self.lexical_k = int(clusters), int(lexical_k)
        if not lexical:
            self.mem.store.lexical = lambda *a, **k: []
        self.map: dict[int, int] = {}
        self.name = label
        self.capacity = cap

    def write(self, it):
        nid = self.mem.write(it.text, ts=it.ts)
        if nid > 0:
            self.map[nid] = it.id

    def read(self, q, budget_tokens):
        ret = self.mem.read(q, budget_tokens=budget_tokens,
                            clusters=self.clusters, lexical_k=self.lexical_k,
                            candidates=N_CAND, timestamps=False,
                            header="", cite_hint="")
        ranked = [self.cold.text(self.map.get(i, -1)) for i in ret.candidates]
        return ReadResult(ret.block, ranked, handle=ret)

    def credit(self, res, hit, gold, rank):
        if not self.use_credit or res.handle is None:
            return
        ret = res.handle
        cited = [tag for tag, nid in ret.tags.items()
                 if gold and gold in self.cold.text(self.map.get(nid, -1))]
        # Graded, not binary. A hit at rank 40 that only survived because the
        # budget was generous is worth much less than a hit at rank 1, and a
        # binary reward hides that distinction from the allocator entirely.
        if hit and 1 <= rank <= 3:
            outcome = 1.0
        elif hit:
            outcome = 0.5
        else:
            outcome = -0.25
        self.mem.credit(ret, cited=cited, outcome=outcome)

    def resident_bytes(self): return self.mem.cloud.resident_bytes()
    def stats(self): return self.mem.stats()

    def capacity_info(self):
        s = self.mem.cloud.stats()
        return {"slots": s["clusters"], "cap": self.capacity,
                "bound": (s["merges"] + s["evictions"]) > 0}

    def close(self):
        try:
            self.mem.close()
        finally:
            shutil.rmtree(self.dir, ignore_errors=True)


# ============================================ 7. runner


@dataclass
class Result:
    name: str
    fixed: bool
    seed: int = 0
    n_q: int = 0
    hits: int = 0
    hot_q: int = 0
    hot_hits: int = 0
    cold_q: int = 0
    cold_hits: int = 0
    hard_q: int = 0
    hard_hits: int = 0
    para_q: int = 0
    para_hits: int = 0
    lex_q: int = 0
    lex_hits: int = 0
    verbatim: int = 0
    mrr: float = 0.0
    ndcg: float = 0.0
    agg_n: int = 0
    agg_rec: float = 0.0
    stale_err: int = 0
    stale_n: int = 0
    over_budget: int = 0
    hitvec: list[int] = field(default_factory=list)
    hitgrp: list[tuple] = field(default_factory=list)
    paravec: list[int] = field(default_factory=list)
    paragrp: list[tuple] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    w_lat: list[float] = field(default_factory=list)
    r_lat: list[float] = field(default_factory=list)
    resident: int = 0
    items: int = 0
    llm: int = 0
    curve: list[tuple[int, float, int]] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def acc(self): return self.hits / max(self.n_q, 1)
    def tok(self): return statistics.mean(self.tokens) if self.tokens else 0.0
    def eff(self):
        t = self.tok()
        return (self.acc() * 1000.0 / t) if t else 0.0

    def row(self, cost_per_call: float) -> dict:
        p = lambda xs, q: (sorted(xs)[min(len(xs) - 1, int(q * len(xs)))]
                           if xs else 0.0)
        addr = self.extra.get("addressable", 0)
        return {
            "system": self.name,
            "seed": self.seed,
            "fixed_footprint": self.fixed,
            "hit@ctx": round(self.acc(), 4),
            "para@ctx": round(self.para_hits / max(self.para_q, 1), 4),
            "lex@ctx": round(self.lex_hits / max(self.lex_q, 1), 4),
            "hard@ctx": round(self.hard_hits / max(self.hard_q, 1), 4),
            "agg_recall": round(self.agg_rec / max(self.agg_n, 1), 4),
            "mrr": round(self.mrr / max(self.n_q, 1), 4),
            "ndcg@10": round(self.ndcg / max(self.n_q, 1), 4),
            "verbatim": round(self.verbatim / max(self.hits, 1), 4),
            "hot": round(self.hot_hits / max(self.hot_q, 1), 4),
            "cold": round(self.cold_hits / max(self.cold_q, 1), 4),
            "stale_err": round(self.stale_err / max(self.stale_n, 1), 4),
            "resident_mb": round(self.resident / 1e6, 2),
            "bytes_per_item": round(self.resident / max(self.items, 1), 2),
            "addr_frac": round(addr / max(self.items, 1), 4) if addr else "",
            "tok_per_query": round(self.tok(), 1),
            "acc_per_1k_tok": round(self.eff(), 4),
            "over_budget": self.over_budget,
            "write_p50_ms": round(p(self.w_lat, 0.50) * 1e3, 3),
            "read_p50_ms": round(p(self.r_lat, 0.50) * 1e3, 3),
            "read_p99_ms": round(p(self.r_lat, 0.99) * 1e3, 3),
            "llm_calls": self.llm,
            "est_usd": round(self.llm * cost_per_call, 2),
            **self.extra,
        }


def run_system(sys_obj: System, wl: Workload, cold_shared: ColdLog,
               budget_tokens: int, checkpoint_every: int,
               count_index: bool = True) -> Result:
    """The shared log is appended HERE, in stream order, immediately before the
    system sees the same item. No system can observe an item it has not been
    given, and `cold.recent()` cannot return the future."""
    res = Result(sys_obj.name, getattr(sys_obj, "fixed_footprint", True),
                 seed=wl.seed)
    by_at: dict[int, list[Query]] = defaultdict(list)
    for q in wl.queries:
        by_at[q.at_item].append(q)

    for idx, it in enumerate(wl.items):
        cold_shared.add(it)
        t = time.perf_counter()
        sys_obj.write(it)
        res.w_lat.append(time.perf_counter() - t)
        res.items = idx + 1

        for q in by_at.get(idx, ()):
            t = time.perf_counter()
            r = sys_obj.read(q.text, budget_tokens)
            res.r_lat.append(time.perf_counter() - t)

            tok = len(r.block) // CHARS_PER_TOKEN
            if tok > budget_tokens:
                res.over_budget += 1
            res.tokens.append(tok)

            if q.kind == "aggregate":
                found = sum(1 for k in q.gold_keys if k in r.block)
                res.agg_n += 1
                res.agg_rec += found / max(len(q.gold_keys), 1)
                continue

            gold = q.gold_keys[0]
            hit = int(gold in r.block)
            grp = (wl.seed, q.nbhd, gold)
            res.n_q += 1
            res.hits += hit
            res.hitvec.append(hit)
            res.hitgrp.append(grp)
            if q.paraphrased:
                res.para_q += 1; res.para_hits += hit
                res.paravec.append(hit); res.paragrp.append(grp)
            else:
                res.lex_q += 1; res.lex_hits += hit
            if q.hard:
                res.hard_q += 1; res.hard_hits += hit
            if q.hot:
                res.hot_q += 1; res.hot_hits += hit
            else:
                res.cold_q += 1; res.cold_hits += hit
            if hit and q.gold_text and q.gold_text in r.block:
                res.verbatim += 1

            rank = 0
            for i, txt in enumerate(r.ranked, start=1):
                if gold in txt:
                    rank = i
                    break
            if rank:
                res.mrr += 1.0 / rank
                if rank <= 10:
                    res.ndcg += 1.0 / math.log2(rank + 1)

            if q.stale_key:
                res.stale_n += 1
                gp = r.block.find(gold)
                sp = r.block.find(q.stale_key)
                if sp != -1 and (gp == -1 or sp < gp):
                    res.stale_err += 1

            sys_obj.credit(r, bool(hit), gold, rank or 10 ** 6)

        if checkpoint_every and (idx + 1) % checkpoint_every == 0:
            acc, n = probe(sys_obj, wl, budget_tokens, idx)
            res.curve.append((idx + 1, acc, n))

    res.resident = sys_obj.resident_bytes()
    if count_index and getattr(sys_obj, "uses_index", False):
        res.resident += cold_shared.index_bytes()
        res.extra["index_charged"] = True
    res.llm = sys_obj.llm_calls()
    res.extra.update(sys_obj.capacity_info())
    if hasattr(sys_obj, "stats"):
        s = sys_obj.stats()
        res.extra.update({k: s[k] for k in
                          ("clusters", "absorbs", "spawns", "splits",
                           "split_tried", "merges", "evictions", "prunes",
                           "protects", "grants", "steals", "addressable",
                           "ex_slots", "blocks_used", "mean_util",
                           "util_spread", "mean_radius", "p80_radius",
                           "split_thresh")
                          if k in s})
    return res


def probe(sys_obj: System, wl: Workload, budget_tokens: int,
          upto_item: int, n: int = 200) -> tuple[float, int]:
    qs = [q for q in wl.probes
          if q.gold_at <= upto_item and q.kind != "aggregate"][:n]
    if not qs:
        return float("nan"), 0
    ok = 0
    for q in qs:
        try:
            ok += q.gold_keys[0] in sys_obj.read(q.text, budget_tokens).block
        except Exception:
            pass
    return ok / len(qs), len(qs)


# ============================================ 8. reporting


COLS = ["system", "hit@ctx", "para@ctx", "lex@ctx", "agg_recall", "mrr",
        "hot", "cold", "stale_err", "resident_mb", "bytes_per_item",
        "addr_frac", "tok_per_query", "acc_per_1k_tok", "splits", "protects",
        "slots", "cap", "bound", "read_p50_ms", "llm_calls"]


def table(rows: list[dict]) -> str:
    w = {c: max([len(c)] + [len(str(r.get(c, ""))) for r in rows]) for c in COLS}
    out = [" | ".join(c.ljust(w[c]) for c in COLS),
           "-|-".join("-" * w[c] for c in COLS)]
    for r in rows:
        out.append(" | ".join(str(r.get(c, "")).ljust(w[c]) for c in COLS))
    return "\n".join(out)


def paired_delta(a: list[int], b: list[int]) -> tuple[float, float, float]:
    """Naive per-query normal interval. Reported for continuity, but it assumes
    independent queries, which Zipf load makes false."""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0, 0.0, 0.0
    d = np.asarray(a[:n], dtype=float) - np.asarray(b[:n], dtype=float)
    m = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        return m, m, m
    se = sd / math.sqrt(n)
    return m, m - 1.96 * se, m + 1.96 * se


def cluster_bootstrap(a: list[int], b: list[int], groups: list[tuple],
                      iters: int = 2000, seed: int = 0
                      ) -> tuple[float, float, float, int]:
    """Paired delta with a bootstrap resampled over NEIGHBOURHOODS.

    The same neighbourhood and the same gold fact are queried dozens of times
    under Zipf load, so per-query intervals treat correlated observations as
    independent and come out far too narrow. Resampling groups is the honest
    interval, and it is the one that decides whether the credit ablation
    survives."""
    n = min(len(a), len(b), len(groups))
    if n < 2:
        return 0.0, 0.0, 0.0, 0
    d = np.asarray(a[:n], dtype=float) - np.asarray(b[:n], dtype=float)
    keys = {g: i for i, g in enumerate(dict.fromkeys(groups[:n]))}
    inv = np.fromiter((keys[g] for g in groups[:n]), dtype=int, count=n)
    G = len(keys)
    sums = np.bincount(inv, weights=d, minlength=G)
    cnts = np.bincount(inv, minlength=G).astype(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(iters, dtype=float)
    for i in range(iters):
        pick = rng.integers(0, G, G)
        draws[i] = sums[pick].sum() / max(cnts[pick].sum(), 1e-9)
    return (float(d.mean()), float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)), G)


def verdict(rows: list[dict], hv: dict[str, list[int]], hg: dict[str, list],
            pv: dict[str, list[int]], pg: dict[str, list],
            wl: Workload, args) -> str:
    """Refuses to interpret an ablation off a saturated, floored or trivially
    satisfied metric."""
    by = {r["system"]: r for r in rows}
    primary = "nimbus-cloud" if "nimbus-cloud" in by else "nimbus"
    n = by.get(primary)
    out: list[str] = []
    if not n:
        return ""
    out.append(f"primary row = {primary} (cloud routing only; no inverted index)")

    best = max(r["hit@ctx"] for r in rows)
    if best >= 0.98:
        out.append(f"SATURATED: best hit@ctx={best:.3f}. Above ~0.95 the metric "
                   f"has almost no variance, every paired CI collapses onto "
                   f"zero, and NO ablation below can be believed in either "
                   f"direction. Lower --budget-tokens (currently "
                   f"{args.budget_tokens}) or raise --items.")
    if best < 0.12:
        out.append(f"FLOOR: best hit@ctx={best:.3f}. Either the paraphrase "
                   f"mapping is beyond the embedder or the budget is too tight "
                   f"to hold anything. Check --embedder.")

    if args.embedder == "hash" and args.paraphrase_frac > 0:
        out.append(f"INVALID FOR PARAPHRASE: --embedder hash cannot map "
                   f"'purchasing/healthcare' onto 'procurement/hospital'. "
                   f"{args.paraphrase_frac:.0%} of queries are unanswerable by "
                   f"ANY system for reasons unrelated to memory architecture.")

    lex = by.get("bm25-unbounded") or by.get("bm25-capped")
    if lex and lex.get("ndcg@10", 0) > 0.85:
        out.append(f"TASK IS LEXICALLY TRIVIAL: {lex['system']} ndcg@10="
                   f"{lex['ndcg@10']}. Raise --paraphrase-frac.")

    for r in rows:
        if r.get("over_budget"):
            out.append(f"INVALID: {r['system']} exceeded the token budget on "
                       f"{r['over_budget']} queries; its row is not comparable.")
    unbound = [r["system"] for r in rows
               if r.get("fixed_footprint") and r.get("bound") is False]
    if unbound:
        out.append("NOT A COMPARISON: capacity never bound for "
                   + ", ".join(unbound) +
                   ". Lower --byte-budget or raise --items until every fixed "
                   "row shows bound=True.")

    spread = [r["tok_per_query"] for r in rows if r["tok_per_query"] > 0]
    if spread and max(spread) > 1.35 * min(spread):
        out.append(f"TOKEN SPREAD {min(spread):.0f}..{max(spread):.0f}: if the "
                   f"leader is also the fattest, hit@ctx is partly measuring "
                   f"list LENGTH. Read acc_per_1k_tok.")

    if not args.count_index_bytes:
        out.append("INDEX BYTES NOT CHARGED: every BM25-using row is getting "
                   "unbounded recall at zero reported RAM. The bounded-memory "
                   "claim cannot be tested under this setting.")

    # --- addressability: the ceiling nobody looks at
    addr = n.get("addressable", 0)
    if addr:
        frac = addr / max(n.get("bytes_per_item", 1) and args.items, 1)
        out.append(f"ADDRESSABLE: {addr} of {args.items} items reachable "
                   f"through a pointer ({frac:.1%}). para@ctx cannot exceed "
                   f"this by more than what BM25 adds, and this row has no "
                   f"BM25. If para@ctx is close to {frac:.2f}, you are "
                   f"pointer-bound, not geometry-bound: raise --exemplars "
                   f"before touching anything else.")

    if wl.hard_frac > 0.90:
        out.append(f"hard@ctx IS VACUOUS: {wl.hard_frac:.1%} of queries are "
                   f"'hard' at --hard-min-siblings {args.hard_min_siblings}, so "
                   f"hard@ctx is hit@ctx renamed. Raise the threshold or stop "
                   f"quoting the metric.")

    for ctrl, bad, ok in (
        ("bm25-capped",
         "The centroid cloud does not beat a bounded inverted index at equal "
         "bytes. On this workload the cloud is decoration.",
         "Cloud beats bounded lexical at equal bytes."),
        ("vector-fifo",
         "Clustering buys nothing over flat capped sampling. This control "
         "matters most.",
         "Clustering beats flat sampling at equal bytes."),
        (f"{primary}-no-credit",
         "Utility-driven allocation is not contributing -- the novel claim is "
         "unsupported here.",
         "Utility-driven allocation is contributing."),
    ):
        if ctrl not in hv or primary not in hv:
            continue
        for tag, vecs, grps in (("hit@ctx", hv, hg), ("para@ctx", pv, pg)):
            if ctrl not in vecs or len(vecs[ctrl]) < 2:
                continue
            m, lo, hi = paired_delta(vecs[primary], vecs[ctrl])
            cm, clo, chi, G = cluster_bootstrap(
                vecs[primary], vecs[ctrl], grps.get(primary, []), seed=args.seed)
            zero = clo <= 0.0 <= chi
            out.append(
                f"vs {ctrl} [{tag}]: {m:+.4f} naive [{lo:+.4f},{hi:+.4f}] | "
                f"cluster-bootstrap over {G} neighbourhoods "
                f"[{clo:+.4f},{chi:+.4f}] " +
                ("NOT DISTINGUISHABLE FROM ZERO once query correlation is "
                 "accounted for. " + bad if zero
                 else (ok if cm > 0 else "WE LOSE. " + bad)))

    if n.get("splits", 0) == 0:
        out.append(f"splits=0: the allocator never fired. p80_radius="
                   f"{n.get('p80_radius')} split_thresh={n.get('split_thresh')} "
                   f"tried={n.get('split_tried')}. The credit ablation above is "
                   f"VACUOUS, not negative.")
    elif n.get("util_spread", 0) < 0.02:
        out.append(f"util_spread={n.get('util_spread')}: utilities never "
                   f"differentiated, so value-ordered splitting had nothing to "
                   f"order on. Raise --util-eta or --queries.")

    if n.get("protects", 0) == 0 and args.exemplars > 0:
        out.append("protects=0: no cited pointer was ever protected, so reward "
                   "affected centroid geometry only. The pointer-level channel "
                   "is inert -- check that credit() is being called with the "
                   "tags actually cited.")
    if n.get("blocks_total", 0) and n.get("blocks_used", 0) == 0:
        out.append("blocks_used=0: no cluster reached --grant-util, so the "
                   "overflow pointer tier is pre-allocated and idle. It is "
                   "costing you bytes for nothing; lower --grant-util or set "
                   "--extra-frac 0.")

    for ub in ("vector-unbounded", "bm25-unbounded"):
        if ub in by:
            u = by[ub]
            m, lo, hi = paired_delta(hv.get(primary, []), hv.get(ub, []))
            rb = u["resident_mb"] or 1e-9
            out.append(f"vs {ub}: {m:+.4f} hit@ctx [{lo:+.4f}, {hi:+.4f}] at "
                       f"{n['resident_mb']}MB vs {rb}MB "
                       f"({rb / max(n['resident_mb'], 1e-9):.1f}x RAM). State "
                       f"this cost yourself; a reviewer will find it. One point "
                       f"is not a curve; use --sweep-budgets.")

    hc = n["hot"] - n["cold"]
    out.append(f"hot-cold = {hc:+.4f} on Zipf-skewed neighbourhood load. " +
               ("Reallocation IS concentrating resolution on repeatedly-queried "
                "regions -- this is the claim's fingerprint." if hc > 0.03 else
                "No concentration. Either credit is not flowing, or the "
                "allocator has no slack to reallocate WITH: check "
                "splits/merges > 0 and util_spread."))
    return "\n".join("  - " + s for s in out)


def plot(results: list[Result], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[plot] matplotlib not installed, skipping")
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for r in results:
        pts = [(x, y) for x, y, k in r.curve if not math.isnan(y) and k > 0]
        if pts:
            x, y = zip(*pts)
            ax[0].plot(x, y, marker="o", ms=3, label=r.name)
    ax[0].set(xlabel="items ingested", ylabel="probe accuracy", ylim=(0, 1),
              title="Recall of EARLY facts vs stream length (fixed probes)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    for r in results:
        x = max(r.resident / 1e6, 1e-3)
        ax[1].scatter(x, r.acc(), s=45)
        ax[1].annotate(r.name, (x, r.acc()), fontsize=7)
    ax[1].set(xscale="log", xlabel="resident MB (log)", ylabel="hit@ctx",
              title="Accuracy per resident byte (index charged)")
    ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=140)
    print(f"[plot] wrote {path}")


def plot_sweep(sweep: list[dict], path: str, xkey: str = "resident_mb",
               xlabel: str = "resident MB (log)") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[plot] matplotlib not installed, skipping")
        return
    by: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in sweep:
        by[r["system"]].append((r.get(xkey, 0.0), r["para@ctx"]))
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, pts in by.items():
        pts.sort()
        x, y = zip(*pts)
        ax.plot([max(v, 1e-3) for v in x], y, marker="o", ms=4, label=name)
    ax.set(xscale="log", xlabel=xlabel, ylabel="para@ctx",
           title="Paraphrase accuracy -- the frontier")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=140)
    print(f"[plot] wrote {path}")


# ============================================ 9. cli


DEFAULT = ("recency-window,bm25-unbounded,bm25-capped,vector-unbounded,"
           "vector-fifo,vector-reservoir,vector-fifo+bm25,summary-proxy,"
           "nimbus,nimbus-cloud,nimbus-cloud-no-credit")

CORE = "bm25-capped,vector-fifo,nimbus-cloud,nimbus-cloud-no-credit"


def build(name: str, cold: ColdLog, budget: int, emb, args):
    ck = {"alpha": args.alpha, "tau_max": args.tau_max,
          "split_quantile": args.split_quantile, "util_eta": args.util_eta,
          "grant_util": args.grant_util, "protect_frac": args.protect_frac}
    if args.split_radius is not None:
        ck["split_radius"] = args.split_radius
    nk = dict(half_life_s=args.half_life_s, exemplars=args.exemplars,
              extra_exemplars=args.extra_exemplars, extra_frac=args.extra_frac,
              maintain_every=args.maintain_every,
              commit_every=args.commit_every, recency_k=args.recency_k,
              recency_pin=args.recency_pin,
              coverage_weight=args.coverage_weight,
              clusters=args.clusters, lexical_k=args.lexical_k,
              cloud_kwargs=ck)
    vk = dict(recency_k=args.recency_k, recency_pin=args.recency_pin)
    if name == "recency-window": return RecencyWindow(cold, budget)
    if name == "bm25-unbounded": return Bm25Unbounded(cold, budget)
    if name == "bm25-capped": return Bm25Capped(cold, budget)
    if name == "vector-unbounded":
        return VectorStore(cold, budget, emb, "none", **vk)
    if name == "vector-fifo":
        return VectorStore(cold, budget, emb, "fifo", **vk)
    if name == "vector-reservoir":
        return VectorStore(cold, budget, emb, "reservoir", **vk)
    if name == "vector-fifo+bm25":
        return VectorStore(cold, budget, emb, "fifo", hybrid=True, **vk)
    if name == "summary-proxy":
        return SummaryProxy(cold, budget, corrupt_digits=args.corrupt_digits,
                            slot_chars=args.summary_slot_chars)
    if name == "nimbus":
        return NimbusAdapter(cold, budget, emb, label="nimbus", **nk)
    if name == "nimbus-cloud":
        return NimbusAdapter(cold, budget, emb, lexical=False,
                             label="nimbus-cloud", **nk)
    if name == "nimbus-cloud-no-credit":
        return NimbusAdapter(cold, budget, emb, lexical=False, credit=False,
                             label="nimbus-cloud-no-credit", **nk)
    if name == "nimbus-no-credit":
        return NimbusAdapter(cold, budget, emb, credit=False,
                             label="nimbus-no-credit", **nk)
    raise SystemExit(f"unknown system: {name}")


def get_embedder(kind: str, dim: int):
    if kind == "hash":
        print("[warn] HashEmbedder is a bag-of-words hash with NO semantics. "
              "Paraphrased queries are unanswerable under it for every system. "
              "Use --embedder st for anything you intend to quote.")
        return HashEmbedder(dim=dim)
    if kind == "st":
        from nimbus import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder()
    if kind == "openai":
        from nimbus import OpenAIEmbedder
        return OpenAIEmbedder(dim=dim)
    raise SystemExit(f"unknown embedder: {kind}")


def one_pass(names: list[str], wl: Workload, emb, budget: int,
             args) -> list[Result]:
    results: list[Result] = []
    for name in names:
        cold = ColdLog()          # EMPTY. run_system fills it in stream order.
        print(f"[run] {name} @ {budget/1e6:.2f}MB seed={wl.seed} ...",
              end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            s = build(name, cold, budget, emb, args)
        except ImportError as e:
            print(f"skipped ({e})"); continue
        except KeyError as e:
            print(f"skipped (missing env {e})"); continue
        try:
            r = run_system(s, wl, cold, args.budget_tokens,
                           args.checkpoint_every, args.count_index_bytes)
            results.append(r)
            print(f"hit={r.acc():.3f} para={r.para_hits/max(r.para_q,1):.3f} "
                  f"agg={r.agg_rec/max(r.agg_n,1):.3f} "
                  f"mrr={r.mrr/max(r.n_q,1):.3f} tok={r.tok():.0f} "
                  f"{r.resident/1e6:.2f}MB "
                  f"addr={r.extra.get('addressable','-')} "
                  f"splits={r.extra.get('splits','-')} "
                  f"bound={r.extra.get('bound')} ({time.perf_counter()-t0:.1f}s)")
        finally:
            s.close()
    return results


def make_workload(args, seed: int) -> Workload:
    return Workload(args.items, args.queries, seed=seed,
                    span_days=args.span_days, zipf_s=args.zipf_s,
                    nbhd_zipf_s=args.nbhd_zipf_s,
                    paraphrase_frac=args.paraphrase_frac,
                    aggregate_frac=args.aggregate_frac,
                    probe_cutoff_frac=args.probe_cutoff_frac,
                    hot_top_frac=args.hot_top_frac,
                    hard_min_siblings=args.hard_min_siblings)


def describe(wl: Workload) -> None:
    n_hot = sum(1 for q in wl.queries if q.hot)
    n_hard = sum(1 for q in wl.queries if q.hard)
    n_para = sum(1 for q in wl.queries if q.paraphrased)
    n_agg = sum(1 for q in wl.queries if q.kind == "aggregate")
    sib = list(wl.sib_count.values())
    print(f"[workload seed={wl.seed}] {len(wl.items)} items, "
          f"{len(wl.queries)} queries ({n_para} paraphrased, {n_hot} hot, "
          f"{n_hard} dense [{wl.hard_frac:.0%}], {n_agg} aggregates), "
          f"{sum(1 for i in wl.items if i.key)} facts across "
          f"{len(wl.by_nbhd)} neighbourhoods "
          f"(siblings: median {int(statistics.median(sib)) if sib else 0}, "
          f"max {max(sib) if sib else 0}), "
          f"{sum(1 for q in wl.queries if q.stale_key)} update queries, "
          f"{len(wl.probes)} probes (gold in first {wl.probe_cutoff} items)")


def main() -> None:
    global N_CAND

    ap = argparse.ArgumentParser(description="NIMBUS benchmark harness v3")
    ap.add_argument("--items", type=int, default=50000)
    ap.add_argument("--queries", type=int, default=5000)
    ap.add_argument("--quick", action="store_true", help="6k items, 800 queries")
    ap.add_argument("--systems", default=DEFAULT)
    ap.add_argument("--core", action="store_true",
                    help="only the four rows that decide the thesis")
    ap.add_argument("--embedder", default="st", choices=["hash", "st", "openai"])
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--byte-budget", type=int, default=2_000_000)
    ap.add_argument("--sweep-budgets", default="")
    ap.add_argument("--sweep-exemplars", default="",
                    help="hold bytes fixed, vary pointer density. If para@ctx "
                         "rises with E, you are pointer-bound and the "
                         "fixed-byte frontier is the claim to publish")
    ap.add_argument("--seeds", default="",
                    help="comma-separated seeds; results pooled with a "
                         "cluster bootstrap over (seed, neighbourhood)")
    ap.add_argument("--budget-tokens", type=int, default=400,
                    help="context tokens per query, hard capped. Tight on "
                         "purpose: at 1200 the metric saturates")
    ap.add_argument("--n-cand", type=int, default=N_CAND_DEFAULT)
    ap.add_argument("--count-index-bytes", dest="count_index_bytes",
                    action="store_true", default=True)
    ap.add_argument("--no-count-index-bytes", dest="count_index_bytes",
                    action="store_false",
                    help="free unbounded index. Makes the bounded-memory "
                         "claim untestable")
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--span-days", type=float, default=350.0)
    ap.add_argument("--half-life-days", type=float, default=0.0)
    ap.add_argument("--paraphrase-frac", type=float, default=0.70)
    ap.add_argument("--aggregate-frac", type=float, default=0.15)
    ap.add_argument("--hot-top-frac", type=float, default=0.20)
    ap.add_argument("--hard-min-siblings", type=int, default=8)
    ap.add_argument("--probe-cutoff-frac", type=float, default=0.15)
    ap.add_argument("--zipf-s", type=float, default=1.10)
    ap.add_argument("--nbhd-zipf-s", type=float, default=1.05)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--tau-max", type=float, default=0.94)
    ap.add_argument("--split-quantile", type=float, default=0.80)
    ap.add_argument("--split-radius", type=float, default=None)
    ap.add_argument("--util-eta", type=float, default=0.10)
    ap.add_argument("--exemplars", type=int, default=32)
    ap.add_argument("--extra-exemplars", type=int, default=0,
                    help="utility-granted overflow pointers per block")
    ap.add_argument("--extra-frac", type=float, default=0.0,
                    help="overflow blocks as a fraction of capacity")
    ap.add_argument("--grant-util", type=float, default=0.60)
    ap.add_argument("--protect-frac", type=float, default=0.50)
    ap.add_argument("--recency-k", type=int, default=6)
    ap.add_argument("--recency-pin", type=int, default=0,
                    help="items force-promoted to the top of every ranking. "
                         "v2 effectively used 6 and it cost half the context")
    ap.add_argument("--coverage-weight", type=float, default=1.0,
                    help="weight of per-cluster coverage votes; 0 reverts to "
                         "a single global similarity sort")
    ap.add_argument("--clusters", type=int, default=8)
    ap.add_argument("--lexical-k", type=int, default=100)
    ap.add_argument("--maintain-every", type=int, default=256)
    ap.add_argument("--commit-every", type=int, default=256)
    ap.add_argument("--summary-slot-chars", type=int, default=0)
    ap.add_argument("--corrupt-digits", type=float, default=0.0)
    ap.add_argument("--llm-cost-per-call", type=float, default=0.0002)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", default="benchmarks/results.json")
    ap.add_argument("--csv", default="benchmarks/curve.csv")
    ap.add_argument("--sweep-csv", default="benchmarks/sweep.csv")
    ap.add_argument("--plot", default="benchmarks/results.png")
    ap.add_argument("--sweep-plot", default="benchmarks/frontier.png")
    args = ap.parse_args()

    N_CAND = int(args.n_cand)
    if args.core:
        args.systems = CORE
    if args.quick:
        args.items, args.queries = 6000, 800
        args.checkpoint_every = args.checkpoint_every or 1000
    args.half_life_s = (args.half_life_days * 86400.0
                        if args.half_life_days > 0 else None)

    print(__doc__.split("READ THIS")[0])
    per = CentroidCloud.bytes_per_cluster(
        args.dim, args.exemplars, args.extra_exemplars, args.extra_frac)
    cap = CentroidCloud.capacity_for_bytes(
        args.byte_budget, args.dim, args.exemplars,
        args.extra_exemplars, args.extra_frac)
    print(f"[cfg] items={args.items} queries={args.queries} "
          f"budget={args.byte_budget/1e6:.2f}MB ctx={args.budget_tokens}tok "
          f"n_cand={N_CAND} embedder={args.embedder} "
          f"paraphrase={args.paraphrase_frac} agg={args.aggregate_frac} "
          f"index_charged={args.count_index_bytes} "
          f"recency_k={args.recency_k} pin={args.recency_pin}")
    print(f"[budget] {per} B/cluster at dim={args.dim} E={args.exemplars} "
          f"(+{args.extra_exemplars}x{args.extra_frac}) -> capacity={cap}, "
          f"{cap * (args.exemplars + int(args.extra_frac * args.extra_exemplars))}"
          f" pointer slots for {args.items} items")

    emb = get_embedder(args.embedder, args.dim)
    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             if args.seeds else [args.seed])
    names = [s.strip() for s in args.systems.split(",") if s.strip()]
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)

    # ---- sweeps ------------------------------------------------------------
    if args.sweep_budgets or args.sweep_exemplars:
        wl = make_workload(args, seeds[0])
        describe(wl)
        sweep_rows: list[dict] = []
        if args.sweep_budgets:
            for b in [int(float(x)) for x in args.sweep_budgets.split(",")]:
                for r in one_pass(names, wl, emb, b, args):
                    row = r.row(args.llm_cost_per_call)
                    row["byte_budget"] = b
                    row["exemplars"] = args.exemplars
                    sweep_rows.append(row)
        if args.sweep_exemplars:
            base_E = args.exemplars
            for E in [int(x) for x in args.sweep_exemplars.split(",")]:
                args.exemplars = E
                for r in one_pass(names, wl, emb, args.byte_budget, args):
                    row = r.row(args.llm_cost_per_call)
                    row["byte_budget"] = args.byte_budget
                    row["exemplars"] = E
                    sweep_rows.append(row)
            args.exemplars = base_E
        with open(args.sweep_csv, "w", newline="") as f:
            keys = sorted({k for r in sweep_rows for k in r})
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            w.writerows(sweep_rows)
        print(f"[out] {args.sweep_csv}")
        if args.sweep_exemplars and not args.sweep_budgets:
            plot_sweep(sweep_rows, args.sweep_plot, "exemplars",
                       "exemplars per cluster (bytes fixed)")
        else:
            plot_sweep(sweep_rows, args.sweep_plot)
        with open(args.json, "w") as f:
            json.dump({"config": vars(args), "sweep": sweep_rows}, f,
                      indent=2, default=str)
        print(f"[out] {args.json}")
        return

    # ---- normal run, possibly multi-seed -----------------------------------
    all_rows: list[dict] = []
    hv: dict[str, list[int]] = defaultdict(list)
    hg: dict[str, list] = defaultdict(list)
    pv: dict[str, list[int]] = defaultdict(list)
    pg: dict[str, list] = defaultdict(list)
    last_results: list[Result] = []
    last_wl: Workload | None = None

    for sd in seeds:
        wl = make_workload(args, sd)
        describe(wl)
        results = one_pass(names, wl, emb, args.byte_budget, args)
        if not results:
            continue
        rows = [r.row(args.llm_cost_per_call) for r in results]
        rows.sort(key=lambda r: -r["para@ctx"])
        print("\n" + table(rows))
        all_rows.extend(rows)
        for r in results:
            hv[r.name] += r.hitvec; hg[r.name] += r.hitgrp
            pv[r.name] += r.paravec; pg[r.name] += r.paragrp
        last_results, last_wl = results, wl

    if not all_rows or last_wl is None:
        print("no systems ran"); return

    if len(seeds) > 1:
        pooled: dict[str, dict] = {}
        for r in all_rows:
            pooled.setdefault(r["system"], []).append(r)  # type: ignore
        print(f"\nPOOLED OVER {len(seeds)} SEEDS (mean +- sd)")
        for name, rs in pooled.items():                    # type: ignore
            for m in ("hit@ctx", "para@ctx", "agg_recall", "mrr"):
                vals = [x[m] for x in rs]
                sd_ = statistics.stdev(vals) if len(vals) > 1 else 0.0
                print(f"  {name:28s} {m:11s} "
                      f"{statistics.mean(vals):.4f} +- {sd_:.4f}")

    v = verdict(all_rows, hv, hg, pv, pg, last_wl, args)
    if v:
        print("\nVERDICT (read this before you tweet a number)\n" + v)
    print("\nNOTE: summary-proxy is NOT Mem0 or Zep. Publishable numbers "
          "require LongMemEval/LoCoMo with an LLM judge. This harness can "
          "only tell you whether the idea is worth taking there.")

    with open(args.json, "w") as f:
        json.dump({"config": vars(args), "rows": all_rows}, f,
                  indent=2, default=str)
    print(f"[out] {args.json}")

    if any(r.curve for r in last_results):
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["system", "items", "probe_acc", "n_probes"])
            for r in last_results:
                for x, y, k in r.curve:
                    if not math.isnan(y):
                        w.writerow([r.name, x, round(y, 4), k])
        print(f"[out] {args.csv}")
        plot(last_results, args.plot)


if __name__ == "__main__":
    main()