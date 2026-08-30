"""Nimbus: the agent-facing memory.

    mem.write(text)                 -> encode + insert. no LLM. ~50us + encoder
    ctx = mem.read(query)           -> verbatim context block + [m#] tags
    mem.credit(ctx, cited=...)      -> the learning loop

The LLM is only a reader. It never merges, summarises or rewrites anything.

Two changes from v1 that matter more than they look
---------------------------------------------------
RECENCY IS NO LONGER PINNED. v1 added a flat +1.0 to the fused score of the
last `recency_k` items while the maximum RRF contribution was ~3/61 = 0.05.
Ranks 1..k were therefore ALWAYS the newest k items regardless of the query.
At a 400-token budget that is ~6 of 13 available lines spent on items the
query never asked about: hit rate survives (the gold fact lands at rank ~7)
but MRR collapses and any tighter budget fails outright. Recency now joins the
RRF vote like any other retriever. If your agent does not already carry the
last turns in its prompt, set `recency_pin` explicitly and pay for it knowingly.

FUSION IS PER CLUSTER, NOT GLOBAL. v1 flattened every routed cluster's
exemplars into one similarity sort, which collapses onto the nearest cluster
and returns k near-duplicates. Each cluster now contributes its own ranked
list, weighted by centroid similarity rank, so a "list everything about X"
query gets coverage instead of the same fact five times. Point lookups are
unaffected because the global similarity list still votes first and loudest.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from nimbus.core import CentroidCloud
from nimbus.store import ColdStore, Embedder, HashEmbedder

TAG_RE = re.compile(r"\[m(\d+)\]")
CITE_HINT = "Cite the [m#] tags you relied on."


def parse_tags(text: str) -> list[int]:
    """Extract [m#] citations from a model response. Zero cost, no judge."""
    seen, out = set(), []
    for m in TAG_RE.finditer(text or ""):
        i = int(m.group(1))
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def tool_schema(name: str = "search_memory") -> dict:
    """OpenAI/Anthropic-compatible tool definition for agent-driven recall."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                "Search long-term memory for facts from earlier sessions. "
                "Returns verbatim excerpts tagged [m1], [m2], ... Cite the "
                "tags you rely on."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to recall."},
                    "budget_tokens": {"type": "integer", "default": 1200},
                },
                "required": ["query"],
            },
        },
    }


@dataclass
class Retrieval:
    rid: int
    query: str
    block: str
    tags: dict[int, int] = field(default_factory=dict)            # tag -> item_id
    clusters: dict[int, list[int]] = field(default_factory=dict)  # tag -> clusters
    all_clusters: list[int] = field(default_factory=list)
    cluster_sims: dict[int, float] = field(default_factory=dict)
    candidates: list[int] = field(default_factory=list)  # PRE-budget fused order
    tokens: int = 0

    def item_ids(self) -> list[int]:
        return list(self.tags.values())


class Nimbus:
    """Constant-footprint agent memory.

    Resident footprint is fixed by `capacity`, `dim` and the exemplar
    configuration alone. Ingesting a thousand items or a billion changes
    nothing about RAM. Use `Nimbus.plan(byte_budget, dim, ...)` to pick a
    capacity from a byte target instead of guessing.
    """

    @staticmethod
    def plan(byte_budget: int, dim: int, exemplars: int = 32,
             extra_exemplars: int = 0, extra_frac: float = 0.0) -> dict:
        cap = CentroidCloud.capacity_for_bytes(
            byte_budget, dim, exemplars, extra_exemplars, extra_frac)
        per = CentroidCloud.bytes_per_cluster(
            dim, exemplars, extra_exemplars, extra_frac)
        return {"capacity": cap, "bytes_per_cluster": per,
                "resident_bytes": cap * per,
                "addressable_slots": cap * (exemplars
                                            + int(extra_frac * extra_exemplars))}

    def __init__(
        self,
        embedder: Embedder | None = None,
        path: str = "./nimbus_data",
        capacity: int = 10_000,
        recency_k: int = 6,
        recency_pin: int = 0,
        maintain_every: int = 512,
        commit_every: int = 1,
        rrf_k: float = 60.0,
        vec_weight: float = 1.0,
        lex_weight: float = 1.0,
        rec_weight: float = 1.0,
        coverage_weight: float = 1.0,
        ignored_reward: float = -0.2,
        lexical_max_terms: int = 8,
        **cloud_kwargs,
    ) -> None:
        self.emb = embedder or HashEmbedder()
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.store = ColdStore(os.path.join(path, "cold.sqlite"), dim=self.emb.dim)

        cloud_path = os.path.join(path, "cloud.npz")
        if os.path.exists(cloud_path):
            self.cloud = CentroidCloud.load(cloud_path)
            if self.cloud.dim != self.emb.dim:
                raise ValueError(
                    f"encoder dim {self.emb.dim} != stored cloud dim "
                    f"{self.cloud.dim}. Centroids live in one embedding space; "
                    "re-encode from the cold log to migrate."
                )
        else:
            self.cloud = CentroidCloud(self.emb.dim, capacity=capacity,
                                       **cloud_kwargs)

        self.recency_k = int(recency_k)
        self.recency_pin = int(recency_pin)
        self.maintain_every = int(maintain_every)
        self.commit_every = max(1, int(commit_every))
        self.rrf_k = float(rrf_k)
        self.vec_weight = float(vec_weight)
        self.lex_weight = float(lex_weight)
        self.rec_weight = float(rec_weight)
        self.coverage_weight = float(coverage_weight)
        self.ignored_reward = float(ignored_reward)
        self.lexical_max_terms = int(lexical_max_terms)
        self._since_maintain = 0
        self._since_commit = 0
        self._rid = 0
        self._open: dict[int, Retrieval] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ write

    def _maybe_commit(self, n: int = 1) -> None:
        """Batch sqlite commits. One commit per write turns a ~50us insert
        into a fsync-bound one and dominates any write-latency measurement."""
        self._since_commit += n
        if self._since_commit >= self.commit_every:
            self._since_commit = 0
            self.store.commit()

    def write(self, text: str, ts: float | None = None, kind: str = "msg",
              meta: dict | None = None) -> int:
        """Ingest one item. No LLM call. Fire-and-forget, off the critical path."""
        text = (text or "").strip()
        if not text:
            return -1
        ts = time.time() if ts is None else float(ts)
        v = self.emb.encode([text])[0]
        with self._lock:
            item_id = self.store.add(text, ts=ts, kind=kind, meta=meta, vec=v)
            slot, _ = self.cloud.insert(v, item_id, ts)
            self.store.set_cluster(item_id, slot)
            self._since_maintain += 1
            if self._since_maintain >= self.maintain_every:
                self._since_maintain = 0
                self.store.commit()          # maintain() reads vecs back
                self._since_commit = 0
                self.cloud.maintain(self.store.vecs)
            self._maybe_commit()
        return item_id

    def write_many(self, texts: Sequence[str], ts: float | None = None,
                   kind: str = "doc") -> list[int]:
        texts = [t.strip() for t in texts if t and t.strip()]
        if not texts:
            return []
        ts = time.time() if ts is None else float(ts)
        V = self.emb.encode(texts)
        ids = []
        with self._lock:
            for t, v in zip(texts, V):
                iid = self.store.add(t, ts=ts, kind=kind, vec=v)
                slot, _ = self.cloud.insert(v, iid, ts)
                self.store.set_cluster(iid, slot)
                ids.append(iid)
            self._since_maintain += len(ids)
            if self._since_maintain >= self.maintain_every:
                self._since_maintain = 0
                self.store.commit()
                self._since_commit = 0
                self.cloud.maintain(self.store.vecs)
            self.store.commit()
            self._since_commit = 0
        return ids

    # ------------------------------------------------------------------- read

    def read(self, query: str, budget_tokens: int = 1200, clusters: int = 8,
             lexical_k: int = 20, as_of: float | None = None,
             candidates: int = 30, timestamps: bool = True,
             header: str = "## Memory (retrieved, verbatim)",
             cite_hint: str = CITE_HINT) -> Retrieval:
        """Hybrid retrieval: centroid routing + BM25 + recency, fused with
        weighted RRF, filled to a byte budget, returned verbatim.

        `candidates` controls how much of the pre-budget fused ranking is kept
        on the Retrieval. Evaluation harnesses need it to compute ranking
        metrics against the same list length as other systems; scoring only
        the post-truncation tags understates ranking quality.

        The header and cite hint are charged AGAINST `budget_tokens`. They used
        to be free, which meant the returned block exceeded the caller's stated
        budget by ~40 tokens and no accounting caught it.
        """
        with self._lock:
            self._rid += 1
            rid = self._rid

            # (ranked ids, weight) votes
            votes: list[tuple[list[int], float]] = []
            tag_clusters: dict[int, list[int]] = {}
            cluster_sims: dict[int, float] = {}

            q = self.emb.encode([query])[0]
            idx, sims = self.cloud.search(q, k=clusters)
            per_cluster = self.cloud.exemplars_by_cluster(idx)
            flat = sorted({int(i) for lst in per_cluster for i in lst})

            if flat:
                keep, M = self.store.vec_matrix(flat)
                if keep:
                    s = M @ np.asarray(q, dtype=np.float32)
                    sim_of = {i: float(v) for i, v in zip(keep, s)}
                    glob: list[tuple[float, int]] = []
                    for ci, (c, ids) in enumerate(zip(idx, per_cluster)):
                        c = int(c)
                        cluster_sims[c] = float(sims[ci])
                        scored = sorted(
                            ((sim_of[i], i) for i in ids if i in sim_of),
                            reverse=True)
                        if not scored:
                            continue
                        glob.extend(scored)
                        # Coverage vote: this cluster's own best items, damped
                        # by how far down the centroid ranking it sits.
                        if self.coverage_weight > 0.0:
                            votes.append(([i for _, i in scored],
                                          self.coverage_weight / (1.0 + ci)))
                        for i in ids:
                            tag_clusters.setdefault(int(i), []).append(c)
                    if glob:
                        glob.sort(reverse=True)
                        votes.insert(0, ([i for _, i in glob], self.vec_weight))

            lex = self.store.lexical(query, k=lexical_k, as_of=as_of,
                                     max_terms=self.lexical_max_terms)
            if lex:
                votes.append((lex, self.lex_weight))
            rec = self.store.recent(self.recency_k, as_of=as_of)
            if rec:
                votes.append((rec, self.rec_weight))

            fused: dict[int, float] = {}
            for lst, w in votes:
                if w <= 0.0:
                    continue
                for r, iid in enumerate(lst):
                    fused[iid] = fused.get(iid, 0.0) + w / (self.rrf_k + r + 1)

            order = [i for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])]

            # Explicit, accounted-for recency pin. Default 0: v1's unconditional
            # +1.0 on `recency_k` items silently owned the top of every ranking.
            if self.recency_pin > 0 and rec:
                pin = [i for i in rec[: self.recency_pin]]
                order = pin + [i for i in order if i not in set(pin)]

            rows = self.store.get(order)
            if as_of is not None:
                order = [i for i in order if i in rows and rows[i]["ts"] <= as_of]

            overhead = (len(header) + 1 if header else 0)
            overhead += (len(cite_hint) + 3 if cite_hint else 0)
            budget_chars = max(64, budget_tokens * 4 - overhead)

            lines, tags, used, seen_norm = [], {}, 0, set()
            for iid in order:
                row = rows.get(iid)
                if not row:
                    continue
                norm = re.sub(r"\W+", "", row["text"].lower())[:120]
                if norm in seen_norm:
                    continue
                seen_norm.add(norm)
                tag = len(tags) + 1
                if timestamps:
                    stamp = time.strftime("%Y-%m-%d", time.localtime(row["ts"]))
                    line = f"[m{tag}] {stamp} — {row['text']}"
                else:
                    line = f"[m{tag}] {row['text']}"
                if used + len(line) + 1 > budget_chars:
                    if tags:
                        break
                    line = line[:budget_chars]
                lines.append(line)
                tags[tag] = iid
                used += len(line) + 1
                if used >= budget_chars:
                    break

            all_cl = sorted({c for iid in tags.values()
                             for c in tag_clusters.get(iid, [])})
            block = ""
            if lines:
                parts = ([header] if header else []) + lines
                block = "\n".join(parts)
                if cite_hint:
                    block += "\n\n" + cite_hint + "\n"
            ret = Retrieval(
                rid=rid, query=query, block=block, tags=tags,
                clusters={t: tag_clusters.get(i, []) for t, i in tags.items()},
                all_clusters=all_cl, cluster_sims=cluster_sims,
                candidates=order[:candidates], tokens=len(block) // 4,
            )
            self._open[rid] = ret
            if len(self._open) > 4096:
                for k in list(self._open)[:2048]:
                    self._open.pop(k, None)
            return ret

    # ----------------------------------------------------------------- credit

    def credit(self, retrieval: Retrieval | int,
               cited: Sequence[int] | str = (), outcome: float = 1.0) -> dict:
        """Cite-or-drop. Clusters whose excerpts the model cited earn utility;
        clusters injected and ignored decay.

        A cited excerpt ALSO earns its pointer a protection bit, so the item
        stops being a reservoir-eviction candidate. Utility changes where the
        cloud is fine-grained; protection changes what remains addressable at
        all — and pointers, not centroids, are the recall ceiling.

        Note for evaluation code: if you pass every injected tag as `cited`,
        every routed cluster gets the same update and utility carries no
        signal at all. Cite only what was actually used.
        """
        ret = self._open.get(retrieval) if isinstance(retrieval, int) else retrieval
        if ret is None:
            return {"rewarded": 0, "penalised": 0, "protected": 0}
        if isinstance(cited, str):
            cited = parse_tags(cited)
        cited_set = {int(c) for c in cited}

        pos, neg = set(), set()
        for tag, cl in ret.clusters.items():
            (pos if tag in cited_set else neg).add(tuple(cl))
        pos_cl = {c for t in pos for c in t}
        neg_cl = {c for t in neg for c in t} - pos_cl

        protected = 0
        with self._lock:
            self.cloud.credit(sorted(pos_cl), float(np.clip(outcome, -1.0, 1.0)))
            self.cloud.credit(sorted(neg_cl), self.ignored_reward)
            if outcome > 0.0:
                for tag in cited_set:
                    iid = ret.tags.get(tag)
                    if iid is None:
                        continue
                    for c in ret.clusters.get(tag, ()):
                        protected += int(self.cloud.protect(int(c), int(iid)))
        return {"rewarded": len(pos_cl), "penalised": len(neg_cl),
                "protected": protected}

    # ------------------------------------------------------------------- misc

    def search_memory(self, query: str, budget_tokens: int = 1200) -> str:
        """Body for the `search_memory` tool. Returns the block as a string."""
        return self.read(query, budget_tokens=budget_tokens).block or "(no memory)"

    def maintain(self, max_splits: int = 8) -> dict:
        with self._lock:
            self.store.commit()
            self._since_commit = 0
            return self.cloud.maintain(self.store.vecs, max_splits=max_splits)

    def stats(self) -> dict:
        s = self.cloud.stats()
        s["items_ingested"] = self.store.count()
        s["bytes_per_item"] = (round(s["resident_bytes"] / s["items_ingested"], 3)
                               if s["items_ingested"] else None)
        s["addressable_frac"] = (round(s["addressable"] / s["items_ingested"], 4)
                                 if s["items_ingested"] else None)
        return s

    def save(self) -> None:
        with self._lock:
            self.cloud.save(os.path.join(self.path, "cloud.npz"))
            with open(os.path.join(self.path, "config.json"), "w") as f:
                json.dump({"encoder": getattr(self.emb, "name", "?"),
                           "dim": self.emb.dim,
                           "capacity": self.cloud.capacity,
                           "exemplars": self.cloud.E,
                           "extra_exemplars": self.cloud.X_E}, f, indent=2)
            self.store.commit()
            self._since_commit = 0

    def close(self) -> None:
        self.save()
        self.store.close()

    def __enter__(self) -> "Nimbus":
        return self

    def __exit__(self, *exc) -> None:
        self.close()