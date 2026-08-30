"""Cold store and embedders.

The cold store is append-only, ID-addressed, and holds the ONLY copy of the
original text. Nothing is ever paraphrased or rewritten, so nothing is ever
corrupted. It lives on disk and is not part of the resident footprint.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from typing import Iterator, Protocol, Sequence

import numpy as np

_WORD = re.compile(r"[A-Za-z0-9_]+")

# SQLite's default SQLITE_MAX_VARIABLE_NUMBER is 999 on many builds. Raising
# `exemplars` pushes a single read past that and the query fails at runtime,
# which is how a memory-capacity change turns into a crash.
_SQL_CHUNK = 800

# Query-side only. Never affects what is stored.
_STOP = frozenset("""
a an and are as at be been being by for from had has have how i if in into is it
its me my no not of on or our so that the their them then there these this to
too was we were what when where which who whom why will with would you your do
does did can could should shall may might must now after before again also just
about above below over under only very same such than
""".split())


def _chunks(seq: Sequence, n: int = _SQL_CHUNK) -> Iterator[list]:
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class Embedder(Protocol):
    dim: int
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class ColdStore:
    """Append-only item log with lexical (BM25) and vector retrieval.

    A tiny document-frequency table is maintained alongside FTS5. It is not a
    luxury: an OR-query over every token in a natural-language question makes
    SQLite score the entire posting list of words like "the" and "current",
    which is what turned a sub-millisecond read into a 23 ms read. Knowing df
    lets us keep only the discriminative terms.
    """

    def __init__(self, path: str = "nimbus.sqlite", dim: int = 384) -> None:
        self.dim = int(dim)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA cache_size=-32000")
        self.db.execute("PRAGMA mmap_size=268435456")
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS items(
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   ts REAL NOT NULL, kind TEXT, text TEXT NOT NULL,
                   meta TEXT, vec BLOB, cluster INTEGER)"""
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS items_ts ON items(ts)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS df(term TEXT PRIMARY KEY, n INTEGER NOT NULL)"
        )
        self.fts = True
        try:
            self.db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS items_fts "
                "USING fts5(text, content='items', content_rowid='id')"
            )
        except sqlite3.OperationalError:
            self.fts = False  # SQLite built without FTS5; LIKE fallback
        self.db.commit()

        self._df_pending: Counter[str] = Counter()
        self._n_docs = self.count()

    # ------------------------------------------------------------------ write

    def add(self, text: str, ts: float | None = None, kind: str = "msg",
            meta: dict | None = None, vec: np.ndarray | None = None) -> int:
        ts = time.time() if ts is None else float(ts)
        blob = None
        if vec is not None:
            blob = np.asarray(vec, dtype=np.float32).ravel().tobytes()
        cur = self.db.execute(
            "INSERT INTO items(ts,kind,text,meta,vec) VALUES(?,?,?,?,?)",
            (ts, kind, text, json.dumps(meta or {}), blob),
        )
        rid = int(cur.lastrowid)
        if self.fts:
            self.db.execute("INSERT INTO items_fts(rowid,text) VALUES(?,?)", (rid, text))
        for t in {w.lower() for w in _WORD.findall(text)}:
            self._df_pending[t] += 1
        self._n_docs += 1
        return rid

    def set_cluster(self, item_id: int, cluster: int) -> None:
        self.db.execute("UPDATE items SET cluster=? WHERE id=?",
                        (int(cluster), int(item_id)))

    def commit(self) -> None:
        if self._df_pending:
            self.db.executemany(
                "INSERT INTO df(term,n) VALUES(?,?) "
                "ON CONFLICT(term) DO UPDATE SET n=n+excluded.n",
                list(self._df_pending.items()),
            )
            self._df_pending.clear()
        self.db.commit()

    # ------------------------------------------------------------------- read

    def get(self, ids: Sequence[int]) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for part in _chunks(ids):
            if not part:
                continue
            q = ",".join("?" * len(part))
            for r in self.db.execute(
                f"SELECT id,ts,kind,text,meta FROM items WHERE id IN ({q})",
                [int(i) for i in part],
            ):
                out[r[0]] = {"id": r[0], "ts": r[1], "kind": r[2], "text": r[3],
                             "meta": json.loads(r[4] or "{}")}
        return out

    def vecs(self, ids: Sequence[int]) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for part in _chunks(ids):
            if not part:
                continue
            q = ",".join("?" * len(part))
            for i, blob in self.db.execute(
                f"SELECT id,vec FROM items WHERE id IN ({q})",
                [int(x) for x in part],
            ):
                if blob:
                    out[i] = np.frombuffer(blob, dtype=np.float32)
        return out

    def vec_matrix(self, ids: Sequence[int]) -> tuple[list[int], np.ndarray]:
        """Batched form of `vecs` for scoring. One matmul beats N dot products
        once a cluster row holds 32-128 pointers."""
        vm = self.vecs(ids)
        keep = [int(i) for i in ids if i in vm]
        if not keep:
            return [], np.zeros((0, self.dim), dtype=np.float32)
        return keep, np.stack([vm[i] for i in keep]).astype(np.float32)

    def dfs(self, terms: Sequence[str]) -> dict[str, int]:
        if not terms:
            return {}
        out = {t: int(self._df_pending.get(t, 0)) for t in terms}
        for part in _chunks(list(terms)):
            q = ",".join("?" * len(part))
            for t, n in self.db.execute(
                f"SELECT term,n FROM df WHERE term IN ({q})", list(part)
            ):
                out[t] = out.get(t, 0) + int(n)
        return out

    def select_terms(self, query: str, max_terms: int = 8,
                     max_df_frac: float = 0.25) -> list[str]:
        """Keep only discriminative query terms, rarest first.

        This is the difference between a 0.6 ms lexical read and a 23 ms one.
        Dropping a term whose posting list is a quarter of the corpus costs
        nothing in recall and saves scoring 12,500 documents.
        """
        seen, cand = set(), []
        for w in _WORD.findall(query):
            t = w.lower()
            if t in seen:
                continue
            seen.add(t)
            if t in _STOP:
                continue
            if len(t) < 3 and not any(c.isdigit() for c in t):
                continue
            cand.append(t)
        if not cand:
            return []
        df = self.dfs(cand)
        n = max(self._n_docs, 1)
        keep = [t for t in cand if df.get(t, 0) <= max_df_frac * n] or cand
        keep.sort(key=lambda t: (df.get(t, 0), -len(t)))
        return keep[:max_terms]

    def lexical(self, query: str, k: int = 20, as_of: float | None = None,
                max_terms: int = 8) -> list[int]:
        """BM25 over the cold log. Not optional — centroids cannot recall
        'invoice #4471'. This is how exact IDs, numbers and error strings
        get found."""
        terms = self.select_terms(query, max_terms=max_terms)
        if not terms:
            return []
        if self.fts:
            expr = " OR ".join(f'"{t}"' for t in terms)
            try:
                if as_of is None:
                    # No JOIN. The join to items forced a rowid lookup per
                    # match, over tens of thousands of matches.
                    rows = self.db.execute(
                        "SELECT rowid FROM items_fts WHERE items_fts MATCH ? "
                        "ORDER BY bm25(items_fts) LIMIT ?", (expr, k)
                    ).fetchall()
                    return [r[0] for r in rows]
                rows = self.db.execute(
                    "SELECT f.rowid FROM items_fts f JOIN items i ON i.id=f.rowid "
                    "WHERE items_fts MATCH ? AND i.ts<=? "
                    "ORDER BY bm25(items_fts) LIMIT ?", (expr, as_of, k)
                ).fetchall()
                return [r[0] for r in rows]
            except sqlite3.OperationalError:
                pass
        clause = " OR ".join("text LIKE ?" for _ in terms[:8])
        args: list = [f"%{t}%" for t in terms[:8]]
        sql = f"SELECT id FROM items WHERE ({clause})"
        if as_of is not None:
            sql += " AND ts<=?"; args.append(as_of)
        sql += " ORDER BY ts DESC LIMIT ?"; args.append(k)
        return [r[0] for r in self.db.execute(sql, args)]

    def recent(self, k: int = 6, as_of: float | None = None) -> list[int]:
        if k <= 0:
            return []
        sql, args = "SELECT id FROM items", []
        if as_of is not None:
            sql += " WHERE ts<=?"; args.append(as_of)
        sql += " ORDER BY id DESC LIMIT ?"; args.append(k)
        return [r[0] for r in self.db.execute(sql, args)]

    def count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM items").fetchone()[0])

    def close(self) -> None:
        self.commit()
        self.db.close()


# --------------------------------------------------------------------- embedders


class HashEmbedder:
    """Dependency-free hashing embedder over word and char trigrams.

    Deterministic and instant. Good enough for tests, CI and smoke demos.
    Use a real encoder in production — this has no semantics and cannot map a
    paraphrase onto its target.
    """

    name = "hash-v1"

    def __init__(self, dim: int = 384) -> None:
        self.dim = int(dim)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            low = t.lower()
            grams = _WORD.findall(low)
            grams += [low[i:i + 3] for i in range(max(0, len(low) - 2))]
            for g in grams:
                h = hashlib.blake2b(g.encode(), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                out[r, idx] += 1.0 if h[4] & 1 else -1.0
            nrm = np.linalg.norm(out[r])
            if nrm > 0:
                out[r] /= nrm
        return out


class SentenceTransformerEmbedder:
    """Local encoder. `pip install sentence-transformers`."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.m = SentenceTransformer(model)
        self.dim = int(self.m.get_sentence_embedding_dimension())
        self.name = model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        v = self.m.encode(list(texts), normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


class OpenAIEmbedder:
    """Hosted encoder. `pip install openai`, set OPENAI_API_KEY."""

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 512) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model, self.dim, self.name = model, int(dim), model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        r = self.client.embeddings.create(
            model=self.model, input=list(texts), dimensions=self.dim
        )
        v = np.asarray([d.embedding for d in r.data], dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)