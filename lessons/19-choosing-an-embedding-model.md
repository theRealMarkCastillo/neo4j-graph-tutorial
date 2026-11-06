# Lesson 19 — Choosing an embedding model (and re-measuring everything that depends on it)

Since Lesson 8, the companion has embedded memories with `BAAI/bge-small-en-v1.5`: 384
dimensions on the CPU, no key. Lesson 18's **Try recall** tab showed its limit. "dogs" returned
"Mark's son is sick" with almost the same score as a real match. That's because small models
squeeze every score into a narrow band. This lesson swaps in a bigger remote model. Doing that
safely means more than changing a model name:

1. **Measure** which model separates related from unrelated memories best, on your own data.
2. **Re-measure every threshold**: eight cut-offs in the store were tuned on bge-small's scores.
3. **Keep the vectors apart**: one index has one size, and tests shouldn't need an API key.
4. **Backfill**: memories stored before the switch have no vector from the new model.

```bash
# .env
EMBED_BASE_URL=https://api.deepinfra.com/v1/openai   # any OpenAI-compatible /embeddings endpoint
EMBED_MODEL=Qwen/Qwen3-Embedding-8B
# EMBED_API_KEY defaults to LLM_API_KEY

.venv/bin/python -m companion.reembed      # give existing memories a Qwen vector
.venv/bin/python -m companion.web          # same app, new embeddings
```

Leave `EMBED_BASE_URL` empty and everything runs on bge-small, as before.

## 1. Pick a model by measuring, not by leaderboard

The question recall asks: *does this message's best-matching memory score higher when the
message is related than when it isn't?* We wrote 19 related messages ("how is my son doing?",
"my puppy chewed my shoes", "my meds don't seem to help") and 18 unrelated ones ("dogs" for
someone with no dog, "tell me a joke", "my car broke down") against three users' real
memories. For each model we computed the **AUC**: the chance that a random related message
outscores a random unrelated one (0.5 = coin flip, 1.0 = perfect).

| Model | Dims | Where | AUC |
|---|---|---|---|
| BAAI/bge-small-en-v1.5 | 384 | local CPU | 0.854 |
| intfloat/e5-large-v2 | 1024 | DeepInfra | 0.681 |
| thenlper/gte-large | 1024 | DeepInfra | 0.942 |
| BAAI/bge-large-en-v1.5 | 1024 | DeepInfra | 0.956 |
| Qwen/Qwen3-Embedding-4B | 2560 | DeepInfra | 0.962 |
| Qwen/Qwen3-Embedding-0.6B | 1024 | DeepInfra | 0.974 |
| **Qwen/Qwen3-Embedding-8B** | **4096** | DeepInfra | **0.994** |

Three things to notice:

- **Bigger isn't monotonic.** The 4B scored below the 0.6B on this set. Leaderboards average
  over many tasks; yours is one task. Measure it.
- **e5-large scored the worst** despite being a well-known model, because e5 needs `query: ` and
  `passage: ` prefixes and is sensitive to them. A model's usage notes are part of the model.
- **The instruction matters.** Qwen3-Embedding is trained to read an instruction before a
  *query* and nothing before a stored text. Without the instruction, the 8B fell from 0.994 to
  about 0.90. That's a bigger drop than the 1–5% Qwen's model card reports on standard
  benchmarks. Short chat messages leave the model little else to go on.

With Qwen3-8B, a single cut-off got 36 of 37 messages right. bge-small couldn't do that at any
cut-off, and that's why Lesson 18 had to add a cross-encoder.

## 2. Query vs. stored text — [`companion/embeddings.py`](../companion/embeddings.py)

```python
embed(["Logan has a fever"])               # stored text: as is
embed(["how is my kid?"], query=True)      # a message searched against memories:
                                           # "Instruct: Given a message from a user, retrieve
                                           #  memories about the user's life...\nQuery: how is my kid?"
```

Which side is which:

| Comparison | Embedded as |
|---|---|
| message → memories (recall, extraction candidates) | query → stored |
| message → entities (the graph hop) | query → stored |
| new memory → stored memory (dedupe), tombstone | stored ↔ stored |
| entity → entity (same person?), name ↔ name | stored ↔ stored |
| memory ↔ memory (habits, Lesson 13) | both with a *clustering* instruction |

The last row was a surprise, and it's covered under step 3. bge-small has no instructions, so all
three calls are the same for it.

`embed()` also normalizes every vector to length 1, so a dot product is the cosine. It keeps a
small cache, because one extraction embeds the same names and summaries several times and a
remote call takes about 0.3–0.6 s.

## 3. Every threshold belongs to a model — [`companion/calibrate.py`](../companion/calibrate.py)

The store makes eight yes/no decisions with a similarity cut-off. Each was measured on
bge-small (the comments in `store.py` keep those measurements). Qwen spreads its scores
differently, so a value that worked for bge-small is wrong for Qwen, and the result fails
quietly: duplicates pile up, two people merge into one, or a habit is never noticed.

So each model has a **Profile** with its own numbers:

```python
Profile("Qwen/Qwen3-Embedding-8B", dims=4096, duplicate=0.966, forgotten=0.89, entity_match=0.94,
        name_match=0.84, pattern=0.665, sensitive_floor=0.655, sensitive_margin=0.015,
        entity_hop=0.62, query_instruction=..., cluster_instruction=...)
```

A model without a profile is refused at startup, with a message pointing here. The numbers
come from `calibrate.py`, which holds labelled pairs for each decision. They're the cases that
went wrong once, plus near misses:

```bash
.venv/bin/python -m companion.calibrate                    # the model in .env
EMBED_BASE_URL= .venv/bin/python -m companion.calibrate    # bge-small, for comparison
```

```
duplicate  (score, text)  Is the new memory a reworded copy of a stored one? (yes: reinforce it)
  yes 0.977-0.997   no 0.889-0.956   gap 0.021
  current 0.966: 16/16 right   suggested 0.966
```

What the two models showed:

| Decision | Scale | bge-small: yes / no | Qwen3-8B: yes / no | Qwen value |
|---|---|---|---|---|
| duplicate (reworded copy) | score | 0.983+ / ≤ 0.964 | 0.977+ / ≤ 0.956 | 0.966 |
| forgotten (tombstone) | cosine | 0.978+ / ≤ 0.887 | 0.925+ / ≤ 0.858 | 0.89 |
| entity match (same kind) | score | 0.986+ / ≤ 0.946 | 0.949+ / ≤ 0.933 | 0.94 |
| name match ("Mom" ~ "mother") | cosine | 0.803+ / ≤ 0.681 | 0.768+ / ≤ 0.854 overlap | 0.84 |
| pattern (same habit) | cosine | 0.805+ / ≤ 0.803 | 0.682+ / ≤ 0.645 (with instruction) | 0.665 |
| sensitive floor | score | 0.762+ / ≤ 0.737 | 0.695+ / ≤ 0.615 | 0.655 |
| entity hop ("my kid" → Logan) | score | 0.760+ / ≤ 0.731 | 0.645+ / ≤ 0.592 | 0.62 |

Here's what the table says:

- **The better model's scores are lower, not higher.** Qwen puts unrelated pairs far below
  related ones. Floors set for bge-small (0.72, 0.85) would cut off most of Qwen's real matches.
- **The entity hop never worked on bge-small.** Its threshold of 0.85 sits above every
  implicit mention (0.76–0.81), so "how is my kid?" only found Logan through other routes. With
  Qwen at 0.62, the hop finds him. A threshold can look fine for months while it's doing
  nothing, and measuring it is how you find out.
- **Bare names are Qwen's weak spot.** "Ohio" ~ "Iowa" scores 0.854 and "Liz" ~ "Elizabeth"
  scores 0.768. That's tolerable because the name check never decides alone: it runs only after
  the entity match, and nicknames usually arrive as aliases from the extractor.
- **Habits needed their own instruction.** Without one, "heading to the gym" vs. "went rock
  climbing" scored 0.77, higher than two gym sentences (0.73). Complete-linkage clustering
  (Lesson 13) then can't find the habit at any cut-off. With *"Identify the activity or habit
  described"* in front of every memory, the gym sentences score 0.68 or more and climbing
  scores 0.65 or less. Consolidation embeds again with that instruction (the vectors aren't
  stored).

### "Loosely related", decided by the model itself

Lesson 18 used a cross-encoder to label vector matches that aren't about the message, because
bge-small's scores overlap. Qwen's scores don't, so its profile sets `weak_below=0.656`, which
is the single cut from step 1. Recall then labels a match by its own score and skips the
cross-encoder. On real data the cross-encoder had a direct question backwards:

| Message | Memory | Qwen score | Cross-encoder |
|---|---|---|---|
| "how is my kid doing?" | Mark's son Logan attends Elk Grove Elementary | 0.694, related | weak |
| "dogs" | Mark's son's name is Logan | 0.617, weak | weak |

(A current status, like "trying to get Logan to eat while he's sick", is still never labelled
weak. It's context for any message, as in Lesson 18.)

The trace shows `match 0.617 < 0.656` instead of a relevance logit. With bge-small (no
`weak_below`), the cross-encoder still decides.

None of the calibration pairs prove the numbers are right on their own. The scenarios
(`python -m companion.scenarios`) run the whole companion with the new profile, end to end. With
the values above, all 53 scenario checks passed on Qwen3-8B, in two runs out of two.

## 4. Two models, one database

A Neo4j vector index has **one size**, so 384-dim and 4096-dim vectors can't share one. Each
model writes its own property and has its own indexes:

| Model | Property | Indexes |
|---|---|---|
| bge-small (default) | `embedding` | `memory_vec`, `entity_vec`, `episode_vec` |
| Qwen3-Embedding-8B | `embedding_qwen3_embedding_8b` | `memory_vec_qwen3_embedding_8b`, ... |

The Cypher takes both as parameters, which `MemoryStore._q` fills in on every query:

```cypher
CALL db.index.vector.queryNodes($memory_vec, 50, $vec) YIELD node, score ...
CALL db.create.setNodeVectorProperty(m, $vprop, $vec)
RETURN m[$vprop] AS vec
```

This gives three things:

- **The default test run stays local.** `tests/conftest.py` empties `EMBED_BASE_URL` before
  `.env` loads, so the suite runs offline in seconds and doesn't touch the vectors your chats
  use. `TEST_REMOTE_EMBEDDINGS=1` runs it on the remote model: the
  [release check](../README.md#release-check-on-the-model-you-ship).
- **Switching back costs nothing.** The old vectors are still there.
- **Lessons 5–7** (`embed.py`, `graphrag.py`, `team_agent.py`) keep `LOCAL_EMBED_MODEL` and
  their own 384-dim index.

### Where each model runs, after this lesson

| What | Model | Runs | Why there |
|---|---|---|---|
| Your chats (CLI and web UI) | Qwen3-Embedding-8B, if `EMBED_BASE_URL` is set | remote API | best separation of related and unrelated memories |
| The same, with `EMBED_BASE_URL` empty | bge-small | fastembed, CPU | no key, offline, the course default |
| The test suite (default run) | bge-small | fastembed, CPU | offline, free, seconds |
| The release check (`TEST_REMOTE_EMBEDDINGS=1`) | the model in `.env` | remote API | tests the model you ship |
| Lessons 5–7 scripts | bge-small | fastembed, CPU | the learning examples need no account |
| "Loosely related" labels | the cross-encoder `ms-marco-MiniLM-L-6-v2` | fastembed, CPU | only for models without a measured `weak_below` (bge-small) |
| Lesson 15 graph embeddings | FastRP | inside Neo4j (GDS) | they come from graph structure, not text |

So fastembed stays. It's the zero-setup path, the tests depend on it, and it's the fallback
when the network or the provider is down. A remote model is an upgrade you opt into, not a
requirement.

## 5. Backfill — [`companion/reembed.py`](../companion/reembed.py)

Memories stored before the switch have no Qwen vector, so vector search can't see them.

```bash
.venv/bin/python -m companion.reembed              # only nodes missing a vector for this model
.venv/bin/python -m companion.reembed --all        # all of them (e.g. after changing an instruction)
.venv/bin/python -m companion.reembed --user Mark
```

It embeds the **same text the write path embeds**: `"Logan (person): Mark's son"` for an entity
and `"Mark, the user"` for the user node. A vector built from different text would match
differently from one the store writes later.

One kind of node can't be backfilled: a `/forget` tombstone keeps only its vector, never the
text. Those matter only while their episode is open, so a tombstone from before a switch is
ignored (`_tombstones` skips nodes without this model's vector).

## Cost and speed

| | bge-small (local) | Qwen3-8B (DeepInfra) |
|---|---|---|
| one call | ~2 ms | ~300–600 ms |
| per turn | recall + extraction: a handful of calls | the same calls, cached repeats are free |
| storage per vector | 1.5 KB | 16 KB |
| price | free | per token (check your provider's pricing) |
| offline | yes | no |

Recall now waits on one remote call per turn. In the web UI's trace, `recall_ms` grows by
roughly the call time. For a companion that's fine. For a hot path you'd batch calls or run a
smaller model locally.

## Concepts and design choices

*Where this fits:* Step 5 swapped out, and every threshold that depended on it re-measured. The whole pipeline, step by step, is in [design decisions → the pipeline as concepts](../docs/design-decisions.md#the-pipeline-as-concepts).

**Terms with a specific meaning here** (every sense is in the [glossary](../docs/glossary.md#words-with-several-meanings)):

| Term | Means here | Not to be confused with |
|---|---|---|
| [embedding profile](../docs/glossary.md#profile) | sense 2: a model plus its measured thresholds | the core profile (sense 1) |
| [query instruction](../docs/glossary.md#instruction) | sense 2: text before a query for the embedding model | the chat model's system prompt |
| [score / cosine](../docs/glossary.md#similarity-and-score) | each threshold is labelled with its scale | one number meaning the same on every model |
| [semantic](../docs/glossary.md#semantic) | sense 2: vector search quality is what's being compared | semantic memory |
| calibration | picking a threshold from labelled yes/no pairs | probability calibration in classifiers |

**Design choices** (with alternatives in [design-decisions.md](../docs/design-decisions.md)):

| We chose | Because | It costs | Instead, you could | |
|---|---|---|---|---|
| Thresholds kept with the model, unmeasured models refused | scores from different models aren't comparable | a calibration step per model | one global set of thresholds | [D5](../docs/design-decisions.md#d5) |
| Vectors versioned by model: own property and indexes | tests stay local, rollback is free, index sizes can differ | storage while both exist; a backfill command | overwrite and re-index in place | [D5](../docs/design-decisions.md#d5) |
| A large remote model (Qwen3-8B) for the app | AUC 0.994 vs 0.854 for the local model | a network call per recall; per-token cost; not offline | local bge-small; Qwen3-0.6B (0.974, smaller) | [D5](../docs/design-decisions.md#d5) |
| Let the vector score label weak matches when the model separates well | one cut got 36/37 right; the cross-encoder had direct questions backwards | only valid for models with a measured `weak_below` | keep the cross-encoder for every model | [D12](../docs/design-decisions.md#d12) |

## Test it — [`tests/test_19_embeddings.py`](../tests/test_19_embeddings.py)

```bash
.venv/bin/python -m pytest tests/test_19_embeddings.py -v    # 11 tests, ~1 s, no key
```

A fake endpoint stands in for the remote model. The tests check that:

- the default model keeps the property and index names Lessons 8–18 use, and Qwen gets its own
  4096-dim indexes;
- a model with no measured thresholds is refused;
- a query gets the instruction and stored text doesn't; clustering uses its own instruction;
- vectors are unit length, and repeats come from the cache;
- the active model gets every `duplicate`, `forgotten` and `name_match` calibration pair right,
  except its listed `KNOWN_MISSES` (Qwen: three name pairs, including Bob ~ Robert);
- `reembed` fills in a missing vector, and recall finds that memory again;
- a tombstone with no vector for this model is ignored, not a crash;
- with `weak_below` set, the vector score labels weak matches and the cross-encoder never runs.

### Tests that hold on both models

Run the whole suite with `TEST_REMOTE_EMBEDDINGS=1` and, before this lesson, 10 of 197 tests
failed on Qwen. Nine of them were about bge-small, not about memory:

| The test checked | On Qwen | Now it checks |
|---|---|---|
| a typed threshold (`≥ 0.975` in the log) | Qwen's cut is 0.966 | `PROFILE.duplicate` |
| the cross-encoder's labels (`relevance < -8.5`; "the reranker misses ADHD") | no cross-encoder: Qwen labels by its own score, and gets ADHD *right* | whichever labeller `relevance_by` names |
| habit clustering with plain vectors | consolidation uses the cluster instruction | `embed_for_clustering`, and "the three are in one group" |
| the `embedding` property, bge-small's profile | Qwen has its own property and profile | the active profile's |

The tenth was real: Qwen scores "Bob" ~ "Robert" at 0.837, under its 0.84 name cut, so "Bob"
isn't kept as Robert's alias. It's a measured tradeoff (Qwen's name scores overlap, and a
lower cut would merge different people), so it's listed in `KNOWN_MISSES` and the alias test
uses Mike ~ Michael, which both models get. A test pinned to one model's numbers can't tell a
regression from a model change. Write tests against the *profile*, and list a model's known
misses explicitly.

A remote model also changes **timing**. On a slow afternoon the same suite took 13 minutes
instead of 5, and two background-worker tests timed out: they gave a write 5–10 seconds, and
an embedding call alone can take that long. Worse, a test that times out leaves its job
running, and the late write lands in the *next* test's data (Miso was "reinforced" on its first
mention). Waits in tests that touch a remote service are now 60 seconds: the tests still
check that the reply doesn't wait for memory, without assuming how fast the network is.

## Takeaways

- **Thresholds belong to the model**, not to the code. Keep them together (a profile), and
  refuse a model that hasn't been measured.
- **Measure on your own task.** Neither size nor reputation predicted the ranking here.
- **Read the model's usage notes**: query instructions and prefixes changed results more than
  the choice between models did.
- **Keep labelled pairs for every cut-off**, and add each new failure to them. The next model
  is then measured on everything that ever went wrong.
- **Version vectors by model**, so tests, rollbacks and migrations don't fight over one index.

## Next

That's the end of the course. A good place to keep going: try `Qwen/Qwen3-Embedding-0.6B`. It
scored 0.974 at a quarter of the size, so you'd add its profile and run the calibration and
the scenarios.

<!-- nav -->
---

[← Lesson 18: Watching memory work, one turn at a time](18-seeing-memory-work.md) · [All lessons](../README.md#lessons)
