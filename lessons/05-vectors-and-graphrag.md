# Lesson 5 — Vector search and GraphRAG

> This lesson is steps 5 and 8 of the [memory pipeline](../docs/design-decisions.md#the-pipeline-as-concepts):
> embedding text, and *hybrid* retrieval: vectors find where to start, the graph expands from there.
> Why vectors live inside Neo4j instead of a separate vector database: [D1](../docs/design-decisions.md#d1).

## 5.1 — Vector index basics (toy example)

Three-number "embeddings" — imagine the axes *cute animal*, *small*, *vehicle*:

```cypher
UNWIND [{n:'Kitten', v:[0.9, 0.1, 0.0]}, {n:'Puppy', v:[0.8, 0.3, 0.0]},
        {n:'Truck', v:[0.0, 0.1, 0.9]}, {n:'Bicycle', v:[0.1, 0.2, 0.8]}] AS row
CREATE (t:Toy {name: row.n, embedding: row.v})
```
```cypher
CREATE VECTOR INDEX toy_vec IF NOT EXISTS FOR (t:Toy) ON (t.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 3, `vector.similarity_function`: 'cosine'}}
```
```cypher
CALL db.index.vector.queryNodes('toy_vec', 4, [0.85, 0.2, 0.05]) YIELD node, score
RETURN node.name AS toy, round(score, 3) AS similarity
```
```
toy, similarity
"Kitten", 0.996
"Puppy", 0.995
"Bicycle", 0.612
"Truck", 0.541
```
Neo4j rescales cosine to 0–1, so "unrelated" sits near 0.5. Real models use 384–3,072 dims.
(`embed.py` deletes these toy nodes and the `toy_vec` index.)

## 5.2 — Real embeddings: [`embed.py`](../embed.py)

Gives 10 people/projects a text `description`, embeds them with a **local** model
(`BAAI/bge-small-en-v1.5`, 384 dims, no API key), stores the vectors, adds an `:Embedded`
label, and creates one vector index `embedded_vec` over all of them.

```bash
.venv/bin/python embed.py
```
```
Embedded 10 nodes (384 dims)
Vector index 'embedded_vec' is online
```

## 5.3 — Semantic search in the Browser

> **Term:** *semantic search* here means search by meaning through embeddings
> ([semantic, sense 2](../docs/glossary.md#semantic)). It has nothing to do with *semantic
> memory* (general facts, Lesson 8) or the *Semantic Web* (RDF/OWL).

"What's closest in *meaning* to Eve's bio?" (data scientist, ranking models, A/B tests on search)

```cypher
MATCH (eve:Person {name:'Eve'})
CALL db.index.vector.queryNodes('embedded_vec', 4, eve.embedding) YIELD node, score
WHERE node <> eve
RETURN labels(node)[0] AS type, node.name AS name, round(score, 3) AS similarity,
       node.description AS description
```
```
type, name, similarity
"Project", "Search Revamp", 0.877
"Person", "Finn", 0.8
"Person", "Bob", 0.793
```
Search Revamp found by **meaning alone** — no WORKS_ON edge was followed.

## 5.4 — Vector search + graph traversal

```cypher
MATCH (eve:Person {name:'Eve'})
CALL db.index.vector.queryNodes('embedded_vec', 5, eve.embedding) YIELD node, score
WHERE node:Project
WITH node ORDER BY score DESC LIMIT 1
MATCH path = (node)-[:REQUIRES|WORKS_ON]-(x)
OPTIONAL MATCH path2 = (x)<-[:HAS_SKILL]-(:Person)
RETURN path, path2
```
```
Nodes 6, Relationships 8 (HAS_SKILL 4, REQUIRES 2, WORKS_ON 2)
Search Revamp <- Alice, Eve (WORKS_ON); -> Python, ML (REQUIRES); Python <- Gia, Alice, Eve
```
Vectors pick the entry point; the graph supplies the connected facts. This is the retrieval
half of **GraphRAG**.

## 5.5 — GraphRAG end to end: [`graphrag.py`](../graphrag.py)

1. Embed the question with the same local model.
2. One Cypher query: top-3 nodes by meaning, then each one's relationships turned into
   plain-English facts.
3. Send the facts to the LLM as context with the plain OpenAI SDK (`config.llm_client()`,
   any OpenAI-compatible endpoint set by `LLM_*` in `.env`; otherwise the script prints the
   context and stops).

```bash
.venv/bin/python graphrag.py "Who could help us build a recommendation engine?"
```
```
=== Retrieved graph context ===
[Person] Eve (match 0.84)
  Data scientist training ranking models and running A/B experiments on search quality.
  - Eve HAS_ROLE Data Scientist
  - Eve WORKS_ON Search Revamp
  - Eve HAS_SKILL Python (level 4)
  - Eve HAS_SKILL ML (level 5)
  - Cara KNOWS Eve

[Project] Search Revamp (match 0.84)
  Rebuilding product search with semantic ranking, query understanding and machine-learning relevance models.
  - Search Revamp REQUIRES ML
  - Search Revamp REQUIRES Python
  - Eve WORKS_ON Search Revamp
  - Alice WORKS_ON Search Revamp

[Person] Finn (match 0.79)
  Senior engineer exploring Rust for high-performance, low-latency services.
  - Finn HAS_ROLE Senior Engineer
  - Finn HAS_SKILL Rust (level 2)
  - Dev MANAGES Finn
  - Finn KNOWS Alice
```
"Recommendation" appears nowhere in the data, yet the vector step found Eve and Search Revamp
and the graph step added *why* (ML level 5, the project's requirements, Alice as teammate).

```bash
.venv/bin/python graphrag.py "Who should own our new Rust performance work?"
```
```
[Person] Finn (match 0.87)
  Senior engineer exploring Rust for high-performance, low-latency services.
  - Finn HAS_ROLE Senior Engineer
  - Finn HAS_SKILL Rust (level 2)
  - Dev MANAGES Finn
  - Finn KNOWS Alice

[Person] Gia (match 0.86)
  Systems programmer with a background in compilers and Rust developer tooling.
  - Gia HAS_SKILL Python (level 3)
  - Gia HAS_SKILL Rust (level 4)

[Person] Cara (match 0.81)
  ...
```
The bios score Finn and Gia almost the same; the graph supplies the tie-breaker — **Gia's Rust
is level 4, Finn's is level 2**. (The generated-answer step was not run in the original
session because no API key was set.)

---

## Concepts and design choices

*Where this fits:* Steps 5 and 8: embed text, then search by vector and expand along the graph. The whole pipeline, step by step, is in [design decisions → the pipeline as concepts](../docs/design-decisions.md#the-pipeline-as-concepts).

**Terms with a specific meaning here** (every sense is in the [glossary](../docs/glossary.md#words-with-several-meanings)):

| Term | Means here | Not to be confused with |
|---|---|---|
| [semantic](../docs/glossary.md#semantic) | sense 2: *semantic search* = vector similarity of embeddings | semantic *memory* (general facts, Lesson 8) |
| [embedding](../docs/glossary.md#embedding) | sense 1: a text embedding from `bge-small` (384 numbers) | a graph embedding (Lesson 15) |
| [score](../docs/glossary.md#similarity-and-score) | the vector index's 0–1 score, (1 + cosine) / 2 | raw cosine (−1 to 1) |
| [hybrid](../docs/glossary.md#hybrid) | sense 2: vector + graph | vector + keyword (the usual meaning in RAG products) |
| GraphRAG | the *local* style: vector entry points, then traversal | Microsoft GraphRAG's *global* community summaries |

**Design choices** (with alternatives in [design-decisions.md](../docs/design-decisions.md)):

| We chose | Because | It costs | Instead, you could | |
|---|---|---|---|---|
| Vector index inside Neo4j | one query goes from a vector hit to connected facts | fewer vector features than a specialist vector DB | Qdrant, Pinecone, pgvector next to the graph | [D1](../docs/design-decisions.md#d1) |
| A small local embedding model | no key, no cost, offline | weaker separation of related and unrelated text (measured in Lesson 19) | a remote model such as Qwen3-Embedding-8B | [D5](../docs/design-decisions.md#d5) |
| Vector + graph, no keyword search | embeddings find where to start, the graph reasons from there | rare exact words can be missed | add a full-text (BM25) index and fuse the ranks | [D11](../docs/design-decisions.md#d11) |

## Test it — [`tests/test_05_vectors_and_graphrag.py`](../tests/test_05_vectors_and_graphrag.py)

**Retrieval is the part of RAG you can measure without an LLM.** Label a few questions with the
node that *should* come back, and score hit@3:

```
hit@3 = 1.00
HIT  'Who could help us build a recommendation engine?' -> ['Eve', 'Search Revamp', 'Finn']
HIT  'Who knows Rust and low-latency systems?' -> ['Finn', 'Gia', 'Cara']
...
```

Re-run it whenever you change the embedding model, the descriptions or `k`, and compare the
number rather than your impression. `HIT_RATE_MIN` is the bar. Lower it only on purpose.

The **LLM evals** (`pytest -m llm`) grade the generated answer with code:
- the recommendation question must name Eve;
- "Who speaks Japanese?" must name **nobody**.

They come as a pair because of a real regression. Adding "say so instead of guessing" to the
prompt made the model answer *"The provided context does not answer this question"* even
though the context listed Eve's ranking models. One eval alone would have passed either
prompt. The pair catches both failure modes.

## Cheat sheet

| Step | How |
|---|---|
| Store a vector | `CALL db.create.setNodeVectorProperty(n, 'embedding', $vec)` |
| Index it | `CREATE VECTOR INDEX ... OPTIONS {indexConfig: {`vector.dimensions`: N, `vector.similarity_function`: 'cosine'}}` |
| Search | `CALL db.index.vector.queryNodes('index', k, $vec) YIELD node, score` |
| GraphRAG | vector search → `MATCH (node)-[r]-(other)` → facts → LLM |
