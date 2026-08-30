<h1 align="center">NIMBUS</h1>
<p align="center"><b>N</b>on-growing <b>I</b>ncremental <b>M</b>emory with <b>B</b>udgeted <b>U</b>tility <b>S</b>plitting</p>
<p align="center">Constant-footprint agent memory. Zero LLM calls to write. Verbatim recall. Resolution follows reward.</p>

---

## What it is

A fixed array of cluster features that absorbs embeddings by updating running
statistics. Insert is one matrix-vector product and an add. Resident memory is
constant **by construction** — clusters are capped, merged and split, never
appended.

```
capacity 10,000 clusters, dim 384, exemplars 32
  -> 2,469 B/cluster  ->  24.7 MB resident, forever
     1,000 items ingested        -> 24.7 MB
     1,000,000,000 ingested      -> 24.7 MB
```

The only model in the write path is the embedding encoder. Nothing generates,
nothing summarises, nothing rewrites your data.

## Why it exists

Two failure modes in agent memory today:

1. **LLM-per-write pipelines** buy compression with a generation call. That is
   cost and latency on every write, and it *paraphrases your facts* — the
   invoice number drifts, the version pin drifts, and you cannot tell when.
2. **Flat vector stores** don't paraphrase, but they grow. A float32 index costs
   ~1.5 KB per item, so a 2 MB budget holds ~1,300 items. Past that you evict,
   and recall collapses — not because ranking got worse, but because the answer
   is no longer resident.

NIMBUS splits the two jobs apart:

| | |
|---|---|
| **Vectors route.** | A capped centroid cloud decides *where to look*. Fixed size. |
| **Cold store answers.** | An append-only SQLite log returns the *original text, unaltered*. |

Nothing was ever paraphrased, so nothing was ever corrupted.

## The core idea: spend bytes on addresses, not contents

At a fixed budget, what limits recall is not geometry quality — it's how many
items you can still *point at*.

```
flat float32 index    1,544 B/item (384 x 4 B + id)   -> 2 MB reaches ~1,300 items
nimbus cluster row    2,853 B/cluster at E=128        -> 2 MB reaches 31,296 items
                      (centroid + 128 x 4 B pointers)
```

A cluster row buys 128 addresses for less than twice the price of one flat slot.
The pointer, not the centroid, is the unit of addressability — `stats()` reports
it as `addressable`, and that number is your recall ceiling.

## The novel part

The mechanics above are well-trodden: BIRCH cluster features (1996), DenStream
damped windows, IVF centroids, reciprocal rank fusion. The contribution is
**what allocates the budget**.

Every retrieval is logged. When the model answers, NIMBUS parses the `[m#]`
citations out of the response — free, no judge model, no labels — and updates a
per-cluster utility EMA:

```
cited              -> outcome (default +1.0, or pass your own task reward)
injected, ignored  -> -0.2      (redundant is not wrong; don't over-punish)
actively wrong     -> -1.0
```

Two bounded channels consume that signal:

- **Splitting.** A cluster that is wide *and* above-median utility *and*
  above-60th-percentile value earns a split. When the cloud is full, the split
  is paid for by merging the two lowest-value compatible clusters. That transfer
  is the entire thesis.
- **Protection.** A pointer the model actually cited gets a bit in `ex_keep` and
  becomes exempt from reservoir eviction, capped at `protect_frac` of the row so
  a hot cluster can never freeze completely.

`value(i) = (0.10 + util_i) * log1p(weight_i)` — utility earns resolution, bulk
alone does not.

**Status of this claim:** splitting is measurably firing (727 splits/run at 2 MB)
and byte efficiency is real. Whether the *credit* signal driving it is
load-bearing is **not** currently supported by the numbers — see
[Results](#results) limitation 2. Do not cite the reward loop as validated.

## Install

```bash
pip install numpy                     # the only hard requirement
pip install sentence-transformers     # optional: local encoder (recommended)
pip install openai                    # optional: hosted encoder
pip install matplotlib                # optional: benchmark plots
git clone https://github.com/you/nimbus && cd nimbus
```

Python 3.10+. SQLite with FTS5 (standard on CPython builds; falls back to `LIKE`
if absent, with degraded lexical recall).

## Quickstart

```python
from nimbus import Nimbus, SentenceTransformerEmbedder

mem = Nimbus(embedder=SentenceTransformerEmbedder(), path="./data",
             capacity=10_000, exemplars=32)

mem.write("deploy target is fly.io, not vercel. billing under acme-eu.")
mem.write("invoice #4471, $8,200, due March 3")
mem.write("build fails when NODE_OPTIONS is set; we removed it")

ctx = mem.read("where do we deploy and what's the billing account?")
print(ctx.block)
```

```
## Memory (retrieved, verbatim)
[m1] 2026-02-14 — deploy target is fly.io, not vercel. billing under acme-eu.
[m2] 2026-02-14 — build fails when NODE_OPTIONS is set; we removed it

Cite the [m#] tags you relied on.
```

Close the loop:

```python
response = llm(ctx.block + user_msg)
mem.credit(ctx, cited=response)     # parses [m#] out of the text for you
mem.save()
```

## Sizing from a byte budget

Don't guess `capacity`. Solve for it:

```python
from nimbus import Nimbus

Nimbus.plan(byte_budget=2_000_000, dim=384, exemplars=128)
# {'capacity': 701, 'bytes_per_cluster': 2853,
#  'resident_bytes': 1999953, 'addressable_slots': 89728}
```

`bytes_per_cluster(dim, E, extra_exemplars, extra_frac)` is the single source of
truth; the benchmark's capacity solver and the allocated arrays both call it, so
the charged budget and the real footprint cannot drift apart.

```
bytes_per_cluster = 6*dim + 37 + 4*E        (+ overflow-tier share if enabled)

  LS  float32[d]  4d     C   float16[d]  2d    n  f64  8
  util/hits/t_off f32 x3 12   alive u8   1
  EX  int32[E]    4E     ex_seen/ex_fill i32 8  ex_keep i64 8
```

| dim | E | B/cluster | clusters @ 2 MB | pointer slots |
|----:|--:|----------:|----------------:|--------------:|
| 384 | 16 | 2,341 | 854 | 13,664 |
| 384 | 32 | 2,469 | 810 | 25,920 |
| 384 | 64 | 2,725 | 733 | 46,912 |
| 384 | 128 | 2,853 | 701 | 89,728 |

More pointers per cluster costs surprisingly little. If accuracy still rises with
`E`, you are pointer-bound, not geometry-bound — fix that before touching any
clustering parameter. `--sweep-exemplars` measures exactly this at fixed bytes.

## Agent integration

### Pattern A — auto-inject (use for chat)

```python
from openai import OpenAI
from nimbus import Nimbus, OpenAIEmbedder

client = OpenAI()
mem = Nimbus(embedder=OpenAIEmbedder(dim=512), path="./data")

SYSTEM = ("You are a helpful assistant. A memory block may be provided. "
          "Trust it over your priors for user-specific facts. "
          "Cite the [m#] tags you relied on.")

def turn(user_msg: str) -> str:
    ctx = mem.read(user_msg, budget_tokens=1200)
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM + "\n\n" + ctx.block},
                  {"role": "user", "content": user_msg}],
    )
    out = r.choices[0].message.content
    mem.credit(ctx, cited=out)          # the learning loop
    mem.write(user_msg, kind="user")
    mem.write(out, kind="assistant")
    return out
```

### Pattern B — tool call (use for long agent loops)

The model decides when it needs history, so you don't pay 1,200 tokens on every
step of a 200-step loop.

```python
from nimbus import tool_schema

tools = [tool_schema()]                 # OpenAI/Anthropic-compatible

# when the model calls search_memory:
ctx = mem.read(args["query"], budget_tokens=args.get("budget_tokens", 1200))
tool_result = ctx.block                 # or: mem.search_memory(args["query"])
# ... after the model's final answer:
mem.credit(ctx, cited=final_text)
```

### Anthropic

```python
import anthropic

client = anthropic.Anthropic()
ctx = mem.read(user_msg)
r = client.messages.create(
    model="claude-sonnet-4-5", max_tokens=1024,
    system=SYSTEM + "\n\n" + ctx.block,
    messages=[{"role": "user", "content": user_msg}],
)
mem.credit(ctx, cited=r.content[0].text)
```

### Task rewards beat citations

Citations are dense but noisy (models comply ~60–80% of the time). When you have
a hard signal, pass it — it dominates:

```python
mem.credit(ctx, cited=out, outcome=+1.0 if tests_passed else -1.0)
```

Graded outcomes work and are usually better than binary: a hit at rank 1 and a
hit at rank 40 should not pay the same reward.

### Batch ingestion

```python
mem.write_many(chunks_from_pdf, kind="doc")   # one encoder batch, no LLM
mem.maintain()                                # split what earned it
```

### Don't do this

```python
mem.credit(ctx, cited=[t for t in ctx.tags])   # WRONG
```

Crediting every injected tag gives every routed cluster the same update, so
utility carries no signal at all and the allocator has nothing to order on.
Cite only what was used.

## Library API

```python
from nimbus import (Nimbus, tool_schema, parse_tags,
                    HashEmbedder, SentenceTransformerEmbedder, OpenAIEmbedder)
from nimbus.core import CentroidCloud
from nimbus.store import ColdStore
```

### `Nimbus`

| Method | What it does |
|---|---|
| `write(text, ts=None, kind="msg", meta=None) -> int` | Encode + insert. No LLM. Returns cold-store id (`-1` if empty). |
| `write_many(texts, ts=None, kind="doc") -> list[int]` | One encoder batch. |
| `read(query, budget_tokens=1200, clusters=8, lexical_k=20, as_of=None, candidates=30, timestamps=True, header=..., cite_hint=...) -> Retrieval` | Centroid routing + BM25 + recency, weighted-RRF fused, truncated to budget. Header and hint are charged **against** `budget_tokens`. |
| `credit(retrieval, cited=(), outcome=1.0) -> dict` | `cited` may be a list of tag ints or the raw model response string. Returns `{rewarded, penalised, protected}`. |
| `maintain(max_splits=8) -> dict` | Force a prune/split pass. Runs automatically every `maintain_every` writes. |
| `stats() -> dict` | Everything below, plus `items_ingested`, `bytes_per_item`, `addressable_frac`. |
| `save()` / `close()` | Persist `cloud.npz` + `config.json`; `close()` also closes SQLite. Usable as a context manager. |
| `search_memory(query, budget_tokens=1200) -> str` | Tool body; returns the block or `"(no memory)"`. |
| `Nimbus.plan(byte_budget, dim, exemplars=32, ...)` | Static. Capacity from a byte target. |

`as_of=<epoch>` filters retrieval to items written at or before that time — the
mechanism for point-in-time queries over superseded facts.

### `Retrieval`

```python
ret.block        # the verbatim text block you put in the prompt
ret.tags         # {tag_int: item_id} for what survived the budget
ret.candidates   # PRE-budget fused ranking (for ranking metrics)
ret.clusters     # {tag: [cluster ids]}
ret.cluster_sims # {cluster id: centroid similarity}
ret.tokens       # estimated tokens in block
```

Keep the `Retrieval` (or its `rid`) until you can call `credit`. The last 4,096
are held internally, so `credit(rid)` works for a while after the fact.

### `CentroidCloud` (advanced)

`bytes_per_cluster()`, `capacity_for_bytes()`, `insert()`, `search()`,
`exemplars_by_cluster()`, `credit()`, `protect()`, `maintain()`, `prune()`,
`addressable()`, `resident_bytes()`, `save()`/`load()`. You normally do not touch
this; `Nimbus` owns it.

## Architecture

```
                 write (no LLM)                     read
                      |                              |
                 [ encoder ]                    [ encoder ]
                      |                              |
        +-------------v-------------+          +------v--------------------+
        |  CENTROID CLOUD           |          | centroid top-k            |
        |  fixed capacity x dim     |          | + per-cluster coverage    |
        |  LS, n, util, exemplars   |<---------+ + BM25 (exact IDs)        |
        |  adaptive absorb / merge  |  credit  | + recency (a vote, not    |
        |  / split / decay          |          |   a pin, unless you pin)  |
        +-------------+-------------+          | -> weighted RRF -> budget |
                      |                        +------+--------------------+
                      | exemplar pointers             |
        +-------------v-------------------------------v----------------+
        |  COLD STORE - append-only SQLite + FTS5. Not resident.       |
        |  The only copy of the original text. Never rewritten.        |
        +--------------------------------------------------------------+
```

### Absorption is adaptive

`tau_i = clip(1 - alpha * radius_i, tau_min, tau_max)`. Dense clusters demand a
close match; diffuse ones absorb freely. A single global threshold is the
difference between working and not — embedding density varies by orders of
magnitude across the space.

### Splitting is relative, not absolute

The split threshold is a **quantile** of the live radius distribution
(`split_quantile=0.80`). An absolute `split_radius` is a footgun: radius depends
on encoder density (hashing ~0.16, MiniLM ~0.45), so a hardcoded 0.55 means the
allocator never fires and the whole utility policy looks broken rather than
untested. Set `split_radius` only as an extra hard floor.

### Decay is O(1)

Uniform exponential decay is a single global inflation scalar, not a loop over
clusters, renormalised before fp32 precision suffers. Old items lose *weight*, so
new items pull centroids faster and the cloud tracks drift instead of fossilising.

### Merges are guarded

A merge is refused if the merged radius would exceed `radius_max`. Two tight,
distant clusters must never collapse into a meaningless mean. If no compatible
pair exists, the lowest-value cluster is evicted instead — and counted in
`evictions`.

### Recency is a vote, not a pin

Earlier versions added a flat +1.0 to the last `recency_k` items while the
maximum RRF contribution was ~0.05. Ranks 1..k were therefore *always* the newest
k items, regardless of query — at a 400-token budget that is ~6 of 13 available
lines spent on things nobody asked about. Recency now joins the RRF vote like any
other retriever. If your agent doesn't already carry recent turns in its prompt,
set `recency_pin` explicitly and pay for it knowingly.

### Fusion is per cluster

Flattening every routed cluster's pointers into one similarity sort collapses
onto the nearest cluster and returns near-duplicates. Each cluster contributes
its own ranked list, damped by centroid rank
(`coverage_weight / (1 + cluster_rank)`), so "list everything about X" gets
coverage. Set `coverage_weight=0` to revert to a single global sort. See
limitation 3 — this path is currently *losing* on aggregates and is suspect.

## Configuration

Cloud (passed through `Nimbus(**cloud_kwargs)` to `CentroidCloud`):

| Parameter | Default | What it controls |
|---|---:|---|
| `capacity` | `10_000` | Hard cluster cap. **Sets your footprint.** |
| `exemplars` | `32` | Base pointers per cluster. **The real recall bottleneck.** |
| `alpha` | `0.35` | Absorption slack. Higher = fewer, broader clusters. |
| `tau_min` / `tau_max` | `0.30` / `0.94` | Clamps on the adaptive threshold. |
| `half_life_s` | 30 days | Weight half-life. `None` disables decay. |
| `split_quantile` | `0.80` | Radius quantile to become a split candidate. |
| `split_radius` | `None` | Optional **absolute** floor on the threshold. Leave `None`. |
| `radius_max` | `0.80` | Merge refusal ceiling. |
| `min_weight` | `0.05` | Below this effective weight a cluster is pruned. |
| `util_eta` | `0.10` | Utility EMA rate. Higher = faster, twitchier. |
| `protect_frac` | `0.50` | Max fraction of a row citation-protection may pin. |
| `extra_exemplars` / `extra_frac` | `0` / `0.0` | Utility-granted overflow pointer blocks. Off by default. |
| `grant_util` | `0.60` | Utility needed to claim an overflow block. |
| `merge_pool` | `256` | Candidates considered when making room. |

Read path (`Nimbus`):

| Parameter | Default | What it controls |
|---|---:|---|
| `recency_k` | `6` | Recent items that join the RRF vote. |
| `recency_pin` | `0` | Items force-promoted to the top. Costs context; opt in. |
| `maintain_every` | `512` | Writes between prune/split passes. |
| `commit_every` | `1` | SQLite commits batched per N writes. Raise it for throughput. |
| `rrf_k` | `60.0` | RRF constant. |
| `vec_weight` / `lex_weight` / `rec_weight` | `1.0` | Retriever vote weights. |
| `coverage_weight` | `1.0` | Per-cluster coverage votes. `0` = global sort. |
| `ignored_reward` | `-0.2` | Penalty for injected-and-ignored clusters. |
| `lexical_max_terms` | `8` | Discriminative query terms kept for BM25. |

`commit_every=1` fsyncs on every write and will dominate any write-latency
measurement. Set it to 256+ for ingestion.

## Performance

| | |
|---|---|
| Resident memory | `capacity * (6*dim + 37 + 4*E)`, flat forever |
| | 24.7 MB at 10k clusters / dim 384 / E=32 |
| Insert | one fp16 matvec over `capacity x dim` + argmax + SQLite insert |
| Query | same matvec, top-k clusters, one batched matmul over their pointers, BM25 over FTS5, disk fetch |
| LLM calls on write | **zero** |
| Cost per 1M items | encoder only; memory operations are free |

Two things that were real optimisations, not micro-tuning:

- **Lexical term selection.** An OR-query over every token in a natural-language
  question makes SQLite score the posting list of "the" and "current". Keeping
  only terms with `df <= 0.25 * N`, rarest first, turned a 23 ms read into ~0.6 ms
  at no measured recall cost.
- **No JOIN on the FTS path.** Joining `items_fts` to `items` forced a rowid
  lookup per match across tens of thousands of matches.

Measure your own numbers: `write_p50_ms`, `read_p50_ms`, `read_p99_ms` are in
`benchmarks/results.json`. The absolute latencies here are structural claims, not
benchmarked ones.

## Results

Synthetic harness, written by the authors of the system under test. It is a
regression harness and a hypothesis test, **not evidence**. Publishable numbers
require LongMemEval / LoCoMo with an LLM judge (roadmap).

50,000 items / 5,000 queries / `st` embedder (MiniLM-L6-v2) / E=128 /
400-token context / seed 7 / index bytes charged. `para@ctx` = paraphrased-query
accuracy after hard budget truncation. All rows verified `bound=True` — every
store is genuinely at capacity, which is the only regime where the comparison
means anything.

| budget | bm25-capped | vector-fifo | nimbus-cloud | nimbus (no-credit) |
|-------:|------------:|------------:|-------------:|-------------------:|
| 0.06 MB| 0.013       | 0.009       | 0.022        | 0.022              |
| 0.12 MB| 0.014       | 0.011       | 0.025        | 0.028              |
| 0.25 MB| 0.009       | 0.024       | 0.050        | 0.045              |
| 0.50 MB| 0.008       | 0.038       | 0.072        | **0.122**          |
| 1.00 MB| 0.006       | 0.064       | 0.293        | 0.242              |
| 2.00 MB| 0.006       | 0.106       | **0.410**    | 0.405              |

**Byte efficiency (the headline).** nimbus (no-credit) at 0.50 MB scores 0.122,
beating vector-fifo at 2.00 MB (0.106) — equal accuracy at **4x fewer bytes**.
For the credit-enabled config the crossover falls between 0.50 and 1.00 MB,
roughly 3.3x.

**Peak accuracy.** 0.410 vs 0.106 at 2 MB = 3.9x over the strongest baseline.

**Mechanism.** At 2 MB: `addr=31296` of 50,000 items reachable through a pointer
(63%) versus ~1,300 for a flat float32 index, with `splits=727`. The gain tracks
addressability, exactly as the design predicts.

**bm25-capped is flat-to-declining** on paraphrases (0.013 -> 0.006). A bounded
lexical index cannot follow "purchasing for the healthcare scandinavian client"
to a fact that says "hospital procurement", and extra budget does not help it.
This is the control that could have killed the project; it didn't.

### Limitations — read before citing any number above

1. **Missing control: `binary-fifo` is not in the sweep.** A binary-quantized
   flat index costs ~56 B/item (48 B vector + 8 B id) and therefore reaches
   ~36,000 items in 2 MB — comparable addressability to nimbus's 31,296. Until
   that baseline runs, the 4x **cannot be attributed** to the cluster
   architecture rather than to coarser vector compression. This is the single
   most important open experiment in the repo.
2. **The credit mechanism is inverted at low budgets.** no-credit wins at 0.06,
   0.12 and 0.50 MB, and ties at 2 MB (0.410 vs 0.405). The byte-efficiency
   headline comes from the configuration *without* credit. Credit is currently a
   knob, not a demonstrated contribution — the split/merge budget transfer is
   what's working, and it works on geometry alone.
3. **Aggregate queries regress as budget grows**: 0.166 -> 0.080 -> 0.038 at
   0.50 / 1.00 / 2.00 MB, while vector-fifo climbs to 0.199. nimbus loses
   aggregates ~5x at 2 MB. Non-monotonicity in the favourable direction is a bug
   smell, not a tradeoff — suspect the per-cluster coverage fusion or pointer
   churn under heavy splitting.
4. **Single seed.** Everything is seed 7, n=1. The 0.50 MB crossover carrying the
   headline is one measurement. `--seeds 7,8,9` before anything is published.
5. **Protection bits cap out at slot 62.** `ex_keep` is an `int64`, so at the
   headline E=128 more than half of every row can never be citation-protected.
   Any conclusion about the protection channel at E>62 is confounded by this.

### Reproducing

```bash
# sanity check, ~1 min
python -m benchmarks.benchmark --quick --embedder st

# the four rows that decide the thesis, at 2 MB
python -m benchmarks.benchmark --items 50000 --queries 5000 --embedder st --core

# the budget sweep that produces the table above (24 cells, ~2-4 min each)
python -m benchmarks.benchmark --items 50000 --queries 5000 \
  --embedder st --core --exemplars 128 \
  --sweep-budgets 6.25e4,1.25e5,2.5e5,5e5,1e6,2e6

# multi-seed, required for anything load-bearing
python -m benchmarks.benchmark --items 50000 --core --seeds 7,8,9

# pointer density at fixed bytes: does accuracy rise with E?
python -m benchmarks.benchmark --items 50000 --core --sweep-exemplars 16,32,64,128
```

Outputs: `benchmarks/sweep.csv`, `benchmarks/results.json`,
`benchmarks/frontier.png`.

### Reading the harness output

```
[run] nimbus-cloud @ 2.00MB seed=7 ... hit=0.522 para=0.410 agg=0.038
      mrr=0.302 tok=386 2.00MB addr=31296 splits=727 bound=True
```

- `para` — paraphrased-query accuracy; the headline metric
- `hit` / `mrr` — recall and mean reciprocal rank over pre-budget candidates
- `agg` — multi-fact coverage (see limitation 3)
- `addr` — items reachable through a pointer; the mechanism metric and the ceiling
- `bound=True` — **the store is at capacity.** If `False`, the budget never bound
  and the row is not a memory-constrained result. Discard it.

The harness prints a `VERDICT` block that refuses to interpret ablations off
saturated, floored or vacuous metrics. Read it. It is designed to tell you your
result is invalid, and it will.

`--embedder hash` cannot map a paraphrase onto its target under any
configuration. It is for CI and smoke tests only; the harness warns and the
verdict marks the run invalid for paraphrase claims.

## `stats()`

```python
{'clusters': 701, 'capacity': 701, 'resident_bytes': 1999953,
 'resident_mb': 2.0, 'bytes_per_cluster': 2853, 'exemplars': 128,
 'extra_exemplars': 0, 'blocks_used': 0, 'blocks_total': 0,
 'ex_slots': 89728, 'addressable': 31296,
 'mean_radius': 0.44, 'p80_radius': 0.51, 'split_thresh': 0.51,
 'mean_util': 0.5, 'util_spread': 0.03, 'total_weight': 48211.0,
 'inserts': 50000, 'spawns': 1428, 'absorbs': 48572, 'merges': 0,
 'splits': 727, 'split_tried': 812, 'split_failed': 85,
 'evictions': 0, 'prunes': 0, 'protects': 0, 'grants': 0, 'steals': 0,
 'items_ingested': 50000, 'bytes_per_item': 40.0,
 'addressable_frac': 0.6259}
```

Diagnostics that matter more than accuracy:

- `addressable` / `addressable_frac` — **the ceiling.** If accuracy is near this,
  you are pointer-bound: raise `exemplars` before touching anything else.
- `splits == 0` — the allocator never fired. Any credit ablation is vacuous, not
  negative. Check `p80_radius` vs `split_thresh` and `split_tried`.
- `util_spread < 0.02` — utilities never differentiated, so value-ordered
  splitting had nothing to order on. Raise `util_eta` or send more retrievals.
- `protects == 0` — no cited pointer was ever protected; the pointer-level reward
  channel is inert. Verify `credit()` receives the tags actually cited (and see
  limitation 5).
- `blocks_total > 0, blocks_used == 0` — the overflow tier is pre-allocated and
  idle. It is costing bytes for nothing: lower `grant_util` or set
  `extra_frac=0`.

## Known limits — read this

1. **Centroids cannot store facts.** A mean vector will never return
   "invoice #4471". Exact recall comes *only* from exemplar pointers plus BM25
   over the cold log. The lexical path is table stakes, not an optimisation —
   remove it and the demo dies on the first specific question. (The
   `nimbus-cloud` benchmark row disables it deliberately, to prove the routing
   claim isn't secretly BM25. That is not the configuration you ship.)
2. **Exemplars are the bottleneck, not centroids.** A cluster that absorbed
   10,000 items can only surface `E` of them. Raising `E` is cheap: E=32 -> 128
   costs 16% more bytes per cluster and buys 3.5x the pointer slots.
3. **Changing the encoder invalidates every centroid.** They live in one
   embedding space; `Nimbus` raises on a dim mismatch but *cannot* detect a
   same-dim model swap. Pin the model version. Re-encoding from the cold log is
   the migration path — which is why the cold log exists.
4. **"Constant footprint" means the resident working set.** The cold log grows on
   cheap disk. Say this out loud; if you imply you delete everything, engineers
   will call it.
5. **Loading pins your config.** If `cloud.npz` exists, `capacity`, `exemplars`
   and the overflow layout come from the file and your constructor arguments are
   silently ignored. Delete the directory to re-plan, or re-encode from the log.
6. **Contradictions are not averaged, but they aren't resolved either.** Both
   versions live in the cold log with timestamps; `as_of=` filters by time. The
   harness measures this as `stale_err` (superseded fact ranked above its
   replacement). Genuine belief revision is roadmap.
7. **Cold start.** Every cluster begins at `util=0.5`. Utility only carries signal
   after a few thousand retrievals; before that, routing is pure geometry — and
   per limitation 2 above, pure geometry is currently what's winning.
8. **`half_life_s` is not restored across save/load in a documented way.** `lam`
   is persisted directly; if you change the half-life you must rebuild.
9. **Aggregate/coverage queries are a known weakness**, not a known strength.
   See Results limitation 3.

## Prior art

The mechanics are standard and deliberately so — BIRCH cluster features,
DenStream damped windows, IVF/FAISS centroid indexing, reciprocal rank fusion,
Algorithm R reservoir sampling. Recent work supports the regime: LLM-free memory
operations can match generative pipelines at much lower latency, and lean
retrieved contexts beat full-context baselines on LongMemEval.

**The open ground, and NIMBUS's claim:** a hard constant on the resident working
set, with addressability — not just geometry — as the thing the budget buys, and
resolution reallocated from observed retrieval outcomes.

The second half of that claim is not yet supported by this repo's numbers. The
first half is, pending one missing control.

## Roadmap

- [ ] **`binary-fifo` baseline in the sweep** — decides whether the 4x is
      architecture or compression. Nothing else should ship first.
- [ ] Multi-seed re-run of the budget sweep (`--seeds 7,8,9`)
- [ ] Diagnose the aggregate regression (coverage fusion vs pointer churn)
- [ ] `ex_keep` as a bitarray so protection works past slot 62
- [ ] `nimbus.eval` — LongMemEval / LoCoMo harness with an LLM judge
- [ ] float16 LS + PQ centroids for more clusters per MB
- [ ] Belief revision: supersession edges, not just timestamps
- [ ] Learned allocator — only if the closed-form value function proves insufficient
- [ ] HNSW over centroids for `capacity` > 10^6

## Contributing

Numbers welcome, opinions less so. Open an issue with a repro command and a
`mem.stats()` dump. The two most useful contributions right now:

1. The `binary-fifo` baseline (roadmap item 1).
2. An adversarial ingestion stream that breaks centroid stability. That failure
   mode is itself a result.

## Status

Research prototype. The 4x byte-efficiency claim is single-seed and pending one
control. Not ready for production use.

## License

Apache-2.0.
