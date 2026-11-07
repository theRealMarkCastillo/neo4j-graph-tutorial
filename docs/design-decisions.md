# Design decisions and tradeoffs — agent memory on Neo4j

This page is for people designing memory for AI agents. Every lesson builds one piece of a
companion that remembers a person's life. Here is **why each piece is built the way it is,
what it costs, and what else would work**, so you can make a different choice where your
constraints differ.

How to read it:

- Each decision has an id (**D1**–**D22**). The lessons link here from their *Design choices*
  sections.
- **Chose / Why / Costs / Alternatives / Revisit when**: the last item is the signal that our
  choice has stopped fitting.
- Words with more than one meaning (*semantic*, *ontology*, *recall*, *profile*...) link to
  [the glossary's list of overloaded terms](glossary.md#words-with-several-meanings). This page
  always says which sense it means.

**Jump to:** [The pipeline as concepts](#the-pipeline-as-concepts) ·
[Storage](#storage-and-data-model) (D1–D5) · [Writing](#writing-memories) (D6–D9) ·
[Reading](#reading-memories) (D10–D13) · [Time](#time) (D14–D15) ·
[Operations](#operations-privacy-and-quality) (D16–D20) ·
[Growing the graph](#growing-the-graph) (D21–D22) · [Layers](glossary.md#layers) · [Gaps](#gaps-what-this-design-does-not-cover) ·
[Other systems](#how-other-memory-systems-choose)

---

## The pipeline as concepts

One chat turn, step by step, with the field each idea comes from. Where each step's data sits
is in the glossary's [layers](glossary.md#layers): storage, schema, ontology (TBox), facts
(ABox), memory, retrieval, context. The vocabulary of agent
memory is borrowed from at least six fields, and much of the confusion comes from mixing them.

| Step | What the code does | Concept | Field it comes from | Lesson |
|---|---|---|---|---|
| 1 | Classify what to keep: fact, event, plan, feeling, boundary... | memory kinds: [semantic](glossary.md#semantic) / episodic / prospective / procedural | cognitive psychology (Tulving; CoALA for agents) | 8 |
| 2 | Turn the message into structured records | information extraction, structured output | NLP | 9, 11 |
| 3 | Decide "Maya" = "my sister" = an existing node | entity resolution / record linkage | databases, data integration | 9 |
| 4 | Map "girlfriend" to `partner`, refuse "places" as a relation | controlled vocabulary, canonicalization, [ontology](glossary.md#ontology) | knowledge representation | 12 |
| 5 | Embed the memory text as a vector | text embedding; enables [semantic](glossary.md#semantic) (vector) search | information retrieval | 5, 19 |
| 6 | Reinforce a reworded copy, retire a replaced fact | deduplication, supersession, bi-temporal data, edge provenance | databases (temporal DBs) | 9, 11, 20 |
| 7 | Store nodes and relationships | labeled property graph, reification, knowledge graph | graph databases | 3, 8 |
| 8 | On the next message: vector search + graph hop + rules | hybrid retrieval, local GraphRAG, bounded multi-hop traversal | IR + graph traversal | 5, 10, 20 |
| 9 | Rank by match × importance × decay | reranking, forgetting curve | IR + psychology (Ebbinghaus; Generative Agents) | 10, 13 |
| 10 | Infer "brother's daughter" = niece | inference / entailment | logic, ontologies | 12 |
| 11 | Hide what this persona may not see | visibility rule, access control | security | 16 |
| 12 | Render the memory block into the prompt | context engineering, RAG | LLM applications | 10 |
| 13 | Nightly: group repeats into habits, prune trivia | consolidation, episodic → semantic memory | neuroscience / psychology | 13, 20 |
| 14 | Check it all with scripted conversations | evals, scenario tests | software testing + ML evaluation | 11, 18 |

---

## Storage and data model

<a id="d1"></a>
### D1. One database for graph and vectors: Neo4j

**Chose:** Neo4j 5.26 LTS (Community) holds entities, relationships, memories, episodes
**and** their vector indexes. Retrieval mixes both in one Cypher query. (Releases after 5.26
use calendar versions, 2025.01 onwards, and add Cypher 25. 5.26 is the long-term-support line
of the 5.x series.)

**Why:** biographical memory is relational. "How's my sister?" is a hop to Maya, then to every
memory `ABOUT` her. "Anyone in my family?" is a walk over typed relationships. "What did I
believe before?" follows `SUPERSEDES` chains. A vector store can only find text that *looks*
similar. Keeping vectors in the same database means one transaction, one consistency model,
and one query that can start from a vector hit and continue along edges. Lesson 8 tabulates
the needs that vectors alone don't meet.

**Costs:**
- A second query language (Cypher) and a less common operational skill set.
- Neo4j's vector index is good but not a specialist: no BM25 + vector fusion inside the
  database. Recall runs both indexes in one query and fuses their ranks in Python (D11).
- On 5.26, filters run *after* the approximate search. That's why recall asks for the top 50
  and filters by `user` afterwards (see D4). Neo4j 2026.01 added filtering *inside* the vector
  index (Cypher 25's `SEARCH` clause, with filter properties declared on the index) as a
  preview. That removes this cost on newer versions.
- Community Edition has one user database, no fine-grained access control and no clustering.

**Alternatives:**

| Option | Wins when | Loses |
|---|---|---|
| **Vector DB only** (Pinecone, Qdrant, Chroma, pgvector alone) | memory is "find similar past text"; no entities or history | multi-hop, "everything about X", contradiction handling, time |
| **Postgres + pgvector** (+ recursive CTEs, or Apache AGE for openCypher) | you already run Postgres; joins are shallow; you want one familiar DB | deep or variable-length traversals get awkward and slow; graph modeling is by convention |
| **Another property-graph DB** (Memgraph, FalkorDB, Amazon Neptune; embedded: LadybugDB) | Memgraph: in-memory speed and streaming; FalkorDB: Redis-based, low latency, used by Graphiti and Cognee; Neptune: managed on AWS with LPG and RDF; embedded: no server at all | most speak Cypher or openCypher, so the ideas here carry over, but GDS, APOC and vector-index details differ. Kuzu, the best-known embedded option, was archived in October 2025; LadybugDB is its maintained fork |
| **Separate graph DB + vector DB** | each side at large scale, with a specialist team | two writes per memory, no shared transaction, ids to keep in sync |
| **RDF triple store** (GraphDB, Stardog, Neptune RDF) + OWL | formal semantics, standard vocabularies, reasoning across organizations | heavier modeling, SPARQL, weaker vector support; overkill for one person's life |
| **Document/KV store** (LangGraph `BaseStore`, Redis, Mongo) | a "user profile" document is enough | no relationships; every query is a scan or a custom index |
| **Memory framework or service** (Mem0, Zep, Letta, Cognee, LangMem, Neo4j Labs `neo4j-agent-memory`) | you want memory as a product, not a project | less control over identity, time, privacy rules and evaluation; see [Other systems](#how-other-memory-systems-choose) |
| **Cloud platform memory** (Amazon Bedrock AgentCore Memory, Vertex AI Agent Engine Memory Bank) | you already build agents on that cloud; extraction and consolidation are run for you | the provider's extraction rules and scopes; facts are a flat list searched by similarity, with no graph to traverse |
| **Files the agent edits** (Anthropic's memory tool, Letta's git-backed MemFS) | the agent should curate its own notes; humans can read and diff them | no index or graph: the agent finds things with file search, and quality depends on the model's discipline |

**Revisit when:** you need horizontal scale beyond one machine, hard multi-tenant isolation
(D16), or your memory has no entities worth connecting.

<a id="d2"></a>
### D2. A memory is a sentence node that points at entities, not a triple

**Chose:** `(:Memory {text: "Alex's sister Maya is a nurse in Denver"})-[:ABOUT]->(:Entity)`,
plus a separate **life graph** of typed relationships `(:User)-[:RELATED {type:'sister'}]->(Maya)`.
Lesson 3's *reification*: a memory needs its own properties (importance, dates, sensitivity)
and its own links (`SUPERSEDES`, `FROM`, `SUMMARIZES`).

**Why:** pure triples (`Maya -works_as-> nurse`) lose nuance ("a nurse, but thinking of
quitting"), and an LLM reads sentences better than triples. Pure text chunks lose structure.
The hybrid keeps a readable sentence for the prompt and edges for traversal.

**Costs:** two representations of some facts (a memory sentence and a `RELATED` edge) that
must be kept consistent. When a relationship ends, both the edge (`until`) and the memory
(`invalidated_at`) change.

**Alternatives:** triples only (classic knowledge graph; Graphiti stores facts *on* edges
between entity nodes); raw transcript chunks only (simple, but no updates); one profile
document per user (LangMem's *profile* style; easy to read, hard to version).

**Revisit when:** you need to answer analytical questions over facts ("how many of my friends
are nurses?"). Then promote the attributes inside sentences to typed properties or edges.

<a id="d3"></a>
### D3. Labeled property graph, not RDF

**Chose:** Neo4j's labeled property graph (LPG). Identity is our own `id` strings, scoped by
`user`.

**Why:** properties on relationships (`since`, `until`, `said`, `private`) are natural in an
LPG. In RDF they need reification or RDF-star. One person's life doesn't need globally unique
URIs or cross-organization vocabularies.

**Costs:** no standard semantics. Our ontology's meaning lives in our code (D9), and another
system can't import it as-is.

**Alternatives:** RDF + OWL/SHACL for interoperability and standard reasoners. `neosemantics`
(n10s) can import RDF/OWL into Neo4j if you need both.

<a id="d4"></a>
### D4. Every node carries `user`, in one database

**Chose:** multi-tenancy by property. Every query filters `{user: $user}`, and vector searches
filter after the index returns its top 50.

**Why:** it works on Community Edition, and one Cypher query can serve every user.

**Costs:**
- One missing filter leaks data between users. Tests cover each read path, but the database
  doesn't enforce it.
- Post-filtering vector results means a user with few memories in a large shared index can
  get fewer than k results.

**Alternatives:**
- A database per user (Enterprise multi-database): real isolation, but many databases.
- A label per user: index-friendly, but labels are schema, not data.
- Row-level security in another database.
- Neo4j Enterprise fine-grained access control.

**Revisit when:** there are thousands of users, or there's a compliance requirement for
isolation.

<a id="d5"></a>
### D5. Vectors versioned by embedding model

**Chose:** each embedding model writes its own property and indexes (`embedding` for the local
default, `embedding_<model>` otherwise), and has its own measured thresholds: a **profile**
(Lesson 19; *profile* here is not the "core profile" or LangMem's profile memory, see the
[glossary](glossary.md#profile)).

**Why:** a vector index has one dimension size. Thresholds tuned on one model silently break on
another. Tests need to run offline.

**Costs:** extra storage while two models coexist, and a backfill step (`companion.reembed`)
after every switch.

**Alternatives:** overwrite in place and re-index (simple, no rollback); a separate vector DB
per model; Matryoshka-truncated vectors so one model serves several sizes.

---

## Writing memories

<a id="d6"></a>
### D6. An always-on extractor after the reply, not a memory tool in the chat model

**Chose:** after each reply, a separate LLM call with structured output extracts memories,
entities and relations (Lesson 9). It runs in a background thread (Lesson 14), and an
end-of-episode pass catches what only became clear later (Lesson 11).

**Why:** chat models under-save when saving is optional, small models use tools
inconsistently, and the persona prompt stays focused on talking. Lesson 11 compares the
options in a table.

**Costs:** one extra LLM call per turn (a regex skips small talk), and a memory isn't available
until extraction finishes, usually by the next turn.

**Alternatives:**
- *Hot-path* memory tools (MemGPT/Letta, LangMem): the agent decides what to save and can
  confirm "I'll remember that".
- Storing raw transcripts and extracting at read time: cheap writes, expensive and slow reads.
- Batch extraction nightly only: cheapest, but the companion forgets until tomorrow.

**Revisit when:** the user often says "remember this", where an explicit tool gives better UX.
Combine both styles.

<a id="d7"></a>
### D7. Entity resolution as a cascade of cheap checks before expensive ones

**Chose:** the extractor's `ref` to a known id → exact name/alias → embedding similarity of the
same *kind*, **and** similar names → a new entity. Role names ("my son") are placeholders,
renamed when a real name arrives (Lesson 9).

**Why:** every wrong merge corrupts two people's histories, and every missed merge splits one
person in two. Exact matches are certain and cheap. Embeddings catch "Mom" ~ "my mother"
but also merged Seattle into Boston (Lesson 12), hence the second, name-only check.

**Costs:** thresholds per embedding model (D5), and a role-naming rule that is
English-specific.

**Alternatives:**
- Ask the LLM to decide every match: accurate but slow and costly per entity.
- Pure string matching: misses "Bob" ~ "Robert".
- Human-in-the-loop review of merges.
- A dedicated entity-resolution library or service.

<a id="d8"></a>
### D8. Update by superseding, and keep history (bi-temporal)

**Chose:** a new version links `(new)-[:SUPERSEDES]->(old)`. The old memory gets
`invalidated_at` (system time) and stays queryable. Dates in the world are separate
(`occurred_on`, `due_on`, `valid_until`). Near-copies **reinforce** the existing memory instead
of being stored again.

**Why:** "Where did I live before Chicago?" needs the past. Debugging "why did it say that?"
needs to know what the system believed when.

**Costs:** every read must filter current versions. The graph grows. "Forget" must really
delete (D18), because retiring isn't forgetting.

**Alternatives:**
- Overwrite in place (simplest, no history).
- An append-only event log with projections (event sourcing: full audit, more machinery).
- Mem0's styles:
  - 1.x: the LLM chose ADD/UPDATE/DELETE/NOOP against similar existing memories.
  - 2.0 (2026): extraction is a single ADD-only pass, and conflicting versions are resolved
    when memories are retrieved. That's write-cheap and read-heavy, the opposite of
    supersession at write time.
- Graphiti's style: validity intervals on fact edges.

<a id="d9"></a>
### D9. A lightweight ontology in code, mirrored into the graph

**Chose:** a Python dict of relation types with synonyms, hierarchy (`SUBTYPE_OF`), symmetry,
inverses, cardinality and range rules, plus a few kinship compositions. It is copied to
`(:RelType)` nodes so Cypher can use it (Lesson 12). Unknown types are kept and *proposed* for
review. *Ontology* here means sense 2 in the [glossary](glossary.md#ontology): typed relations
with rules, not OWL. It is the TBox, the *ontology layer*; the life graph it types is the ABox
([layers](glossary.md#layers)). Entity kinds belong to the TBox too, but stay in code
(`models.py`) and on each node's `kind`, not as nodes.

**Why:** LLMs write `girlfriend` one day and `partner` the next. Without a vocabulary, "anyone
in my family?" and "a person lives in one place at a time" can't be answered or enforced.
Embedding similarity turned out to be bad at mapping relation words (Lesson 12, experiment 1),
so synonyms are explicit.

**Costs:** someone must maintain the vocabulary. Rules are in Python, not a standard language.

**Alternatives:**
- No ontology: free-form relation labels (fine for a demo, breaks aggregation).
- OWL with a reasoner (formal, heavy).
- SHACL shapes (validation, not inference).
- Let the LLM normalize at write time with the vocabulary in the prompt (we do this too, and
  the code checks it).
- Not an alternative: a BI *semantic layer* (dbt, LookML, Snowflake semantic views). It makes
  *metrics* agree across tools; it doesn't define what relations mean or infer anything. See
  [semantic layer vs ontology layer](glossary.md#semantic-layer-vs-ontology-layer).

---

## Reading memories

<a id="d10"></a>
### D10. Recall every turn, from several retrievers, not on demand

**Chose:** before each reply, `recall()` runs nine **retrievers** in code: core profile, vector
search, entity hop, open plans, yearly dates, recently ended, history, people and past
episodes. It renders one memory block for the prompt (Lessons 10, 18).

**Why:** a companion should notice things the user didn't ask about: the plan due tomorrow,
the boundary, the sister's name. Retrieval in code is predictable and testable, and it costs
no extra LLM round-trip.

**Costs:** every turn pays for recall (Lesson 18 measures it). Retrievers can return
irrelevant items, because vector search always returns *something*; Lesson 18 labels them.
The block uses tokens even for "hi".

**Alternatives:**
- *Agentic retrieval*: give the model a `search_memory` tool (Lesson 7's style). It's cheaper
  when memory is rarely needed, but the model must know to look.
- Long context: put everything in the prompt. That's fine for small memories but expensive,
  and recall inside very long contexts degrades.
- Microsoft GraphRAG's *global* search (community summaries), DRIFT search (local search
  primed by community reports) and LazyGraphRAG (summaries deferred to query time) target
  questions over a whole document corpus, not one person's life.

<a id="d11"></a>
### D11. Hybrid retrieval: vectors find where to start, the graph reasons from there

**Chose:** *hybrid* in both senses of the [glossary](glossary.md#hybrid). The main one is (b),
vector + graph: entities named or *meant* in the message are expanded along `ABOUT` and
`RELATED` edges. Inside the semantic retriever, (a) vector + keyword: a full-text index
(`memory_text`, Lucene, BM25) runs beside the vector index, its hits join the candidates, and
the two rankings are fused with reciprocal-rank fusion (RRF, `1 / (60 + position)` per list).

**Why:** Lesson 6's rule of thumb. Embeddings are good at fuzzy entry points and bad at exact,
counted or multi-hop questions. They also blur rare exact words (a medication name, a product
code), which is what keyword search is good at. RRF combines *ranks*, so BM25 and cosine
scores, which aren't on the same scale, never have to be compared.

Two details matter:
- **The keyword vote only reorders.** Weak matches (Lesson 18) still go last, so a hit on a
  common word ("my", "today") can't lift a memory that isn't about the message.
- **The full-text index filters by user inside Lucene** (`user` is indexed beside `text`),
  so another user's memories can't fill its top hits the way they can fill the vector index's
  top 50 on 5.26 (D1, D4). It's a pre-filter over analyzed tokens, not an exact match:
  `user:mark` also matches `user:mark-2`. The query re-checks `m.user` exactly.

**Costs:** one more index and query per recall (a few milliseconds). A message's common words
also produce keyword hits; they're harmless because of the rule above, but they show up in the
trace's keyword column.

**Alternatives:** Neo4j's `neo4j-graphrag` package ships a `HybridRetriever` (vector +
full-text). It's not used here, for the same reason as the other wrappers: it hides the
Cypher, and it fuses *scores* after normalizing them, where this repo fuses *ranks* and then
applies its own relevance labels and decay. Graphiti fuses embedding and BM25 results with
RRF too. Weighted RRF, or a learned fusion, would be the next step if the keyword vote turns
out too strong or too weak.

<a id="d12"></a>
### D12. Rank with a transparent formula; use a learned model only to label

**Chose:** `rank = 0.6 × similarity + 0.4 × importance/10 × decay` (Lesson 10, the *Generative
Agents* recipe, with decay from Lesson 13). Weak matches are **labelled, not dropped**, by a
cross-encoder or by the vector score when the embedding model separates well (Lessons 18 and
19).

**Why:** every number can be shown in the trace and unit-tested. Dropping weak matches hid an
inference the chat model could make ("my meds don't help" → the ADHD memory). Labelling them
in the prompt misled the model, so the label appears only in the trace.

**Costs:** the weights are hand-picked.

**Alternatives:**
- A cross-encoder that reorders everything: better relevance, opaque, slower.
- A learned-to-rank model from feedback (needs data).
- LLM reranking (costly per turn).

<a id="d13"></a>
### D13. The prompt gets readable sentences grouped by purpose

**Chose:** `format_context()` writes labelled sections ("Core profile", "People and things in
their life", "Open plans to follow up on", "Relevant memories", "Recently, no longer true",
"No longer true (history, not current)"...) in plain sentences, with notes on stale and
sensitive items.

**Why:** models use well-labelled natural language better than raw JSON. Section headings say
*why* an item is there, which steers behaviour (ask about plans, respect boundaries).

**Costs:** formatting choices change model behaviour. Lesson 18 found that a "loosely
related" label made the model ignore a relevant memory. Every change needs the scenarios
re-run.

**Alternatives:** JSON or XML blocks (precise, more tokens); tool results (on demand); a
memory summary written by an LLM (compact, lossy, another call).

---

## Time

<a id="d14"></a>
### D14. Three questions about time, three mechanisms

**Chose:** **validity** (`valid_until`, filtered at read time), **decay** (a forgetting curve
with half-lives per kind, which affects ranking only) and **consolidation** (repeats become a
habit memory; old trivia is pruned). One injectable clock drives all three (Lesson 13).

**Why:** "Is it still true?", "is it worth mentioning?" and "should it be stored this way?"
are different questions. A single database TTL would delete data that history, follow-ups
and consolidation still need.

**Costs:** the half-lives are guesses, checked by scenarios rather than learned. Pruning really
deletes.

**Alternatives:**
- A TTL (APOC `apoc.ttl`): right when *not keeping* data is the goal.
- Recency-only ranking.
- LLM-written "reflections" (Generative Agents) on a schedule.

<a id="d15"></a>
### D15. Episodes as the unit of provenance and summary

**Chose:** each conversation is an `Episode` with ordered `Turn`s, a summary and a mood.
Memories link `FROM` their episode.

**Why:** provenance ("when did I say that?"), a diary view, continuity between sessions, and a
natural point for end-of-episode extraction and redaction.

**Costs:** "the end of an episode" is fuzzy in a chat that never closes. Here, `/new`, `/quit`
or the web UI's *End & save* close it.

**Alternatives:** time windows (a new episode after N idle minutes), or no episodes (a flat
message log).

---

## Operations, privacy and quality

<a id="d16"></a>
### D16. One write transaction per extraction, serialized per user

**Chose:** `apply()` runs as one transaction and locks the user node first, so two writers for
the same user take turns while other users don't wait (Lesson 14).

**Why:** a half-applied extraction leaves contradictions. Lesson 14 measured a check-then-act
race that produced four copies of one memory.

**Costs:** LLM and embedding work must happen outside the transaction, so the lock is held
only for milliseconds. Writes for one user are serialized.

**Alternatives:**
- A queue per user (no database lock, but more infrastructure).
- Uniqueness constraints on a content hash (only catches exact copies).
- Optimistic concurrency with retries.

<a id="d17"></a>
### D17. Privacy rules live in the queries

**Chose:**
- Boundaries, sensitivity and per-persona visibility are Cypher predicates
  (`visible()`, `entity_visible()`), pasted into every read, including derived data like
  summaries and patterns (Lesson 16).
- Sensitive memories are recalled only when the message is clearly about them.

**Why:** a rule in the prompt is a suggestion. A rule in the query is a guarantee for that
query.

**Costs:** each new read path must include the rule. Mutation checks in the tests prove that
removing it breaks something.

**Alternatives:**
- Database-level access control (Enterprise fine-grained privileges): enforced by the server,
  but coarser than "this persona heard it privately".
- A policy engine in front of the store (OPA and similar).
- Separate stores per persona (no sharing at all).

<a id="d18"></a>
### D18. Forgetting means deleting, with redaction

**Chose:** `/forget Jordan` deletes the entity, every memory about or naming them, and redacts
the name from turns and summaries in one transaction. A tombstone (a vector, no text) stops the
end-of-episode pass from re-learning it (Lessons 11 and 19).

**Why:** the right to erasure (GDPR Art. 17) and user trust. Retiring a memory isn't
forgetting it.

**Costs:** provenance is lost on purpose. Backups and logs outside the graph still need their
own policy (see [Gaps](#gaps-what-this-design-does-not-cover)).

<a id="d19"></a>
### D19. A workflow, not an autonomous agent

**Chose:** the companion is a fixed LangGraph: recall → respond → log → (maybe) remember.
Lesson 7's team agent shows the tool-choosing agent style for contrast.

**Why:** memory behaviour should be predictable and testable. The model's freedom is in what
it *says*, not in what it reads or writes.

**Costs:** no "let me check my notes" behaviour, and recall can't adapt mid-turn.

**Alternatives:** a ReAct agent with memory tools (D6, D10); a planner that decides which
retrievers to run.

<a id="d20"></a>
### D20. Evaluate with code-graded scenarios, measure every threshold

**Chose:** three test layers (unit, database, LLM evals). Multi-session scenarios check the
graph and the replies with code (Cypher, regex, ids), never an LLM judge. Every similarity
threshold has labelled pairs (`companion.calibrate`).

**Why:** memory bugs are silent. A duplicate or a wrong merge doesn't crash anything. Code
checks are cheap, deterministic in grading and explainable.

**Costs:** scenarios cover what someone thought to script. LLM output varies, so pass *rates*
need several runs.

**Alternatives:**
- LLM-as-judge for open-ended reply quality.
- Public long-term memory benchmarks, to compare with other systems:
  - LoCoMo (2024): very long multi-session dialogue; doesn't score knowledge updates.
  - LongMemEval (2024): adds knowledge updates and abstention, which matter for
    supersession (D8).
  - BEAM (2026): memory at 1M–10M tokens.

  Most published scores are self-reported by vendors.
- Online metrics from real users (corrections, "you already told me").


---

## Growing the graph

<a id="d21"></a>
### D21. Grow the graph past the user, with bounded walks and edge provenance

**Chose:** named things people like or do are entities, and a preference's edge starts at
the person who has it (`Logan -likes-> Minecraft`). Recall walks at most two links from each
matched entity, never through the user or a protected entity, capped at 15 links. Edges list
the memories that support them (`r.evidence`, only memories about both ends) and end when all
of those are retired. Edges without `evidence` are never ended this way.
([Lesson 20](../lessons/20-growing-the-life-graph.md))

**Why:** a graph that only fans out from the user can't answer "who likes Minecraft?" or
"how is my son's friend?". Unbounded walks turn "related" into "everything", and an edge with
no link to its source outlives the fact that created it.

**Costs:** more nodes for the extractor to resolve; one more property to keep in step on every
retirement; older edges stay unprovenanced until a re-extraction touches them.

**Alternatives:** keep everything past one hop as memory text only (simpler, but no graph
queries); reify each edge as a node with `FROM` links (full provenance, double the nodes);
an agent with a Cypher tool (D19 explains why the companion has none).

<a id="d22"></a>
### D22. Decide identity in the graph; compare prose only where meaning is fuzzy

**Chose:** decisions about *identity* read structure and typed fields, not sentence wording.
A known day lives in `occurred_on` / `due_on` / `valid_until`, not in the text (shown from
the field in prompts and the UI). An activity someone does ("gym", "running") is an `activity`
entity every memory about it is linked to. A habit is memories `ABOUT` one activity on 3+
distinct days, counted in Cypher. Pruning reads "temporary" from `valid_until`, the timeline
reads "dated" from the date fields. Each embedding profile records the extraction model its
thresholds were measured with (`Profile.extraction_model`), and every scenario run checks it.
([Lessons 9](../lessons/09-memory-write-path.md) and [13](../lessons/13-memory-over-time.md);
design notes in `docs/superpowers/specs/2026-11-06-structure-over-prose-design.md`)

**Why:** the extractor is an LLM that can be swapped. When `gemma-4-31B-it` was replaced by
`gemini-3.8-flash`, the gym habit stopped forming. The new model wrote dated, longer sentences,
so no pair of gym memories reached the clustering cut (0.82, measured on undated hand-written
sentences), and a dated rock-climbing memory scored closer to a gym memory than gym memories
did to each other. It also filed one visit as a `plan`, a kind the clustering skipped. Every
threshold and every `kind` switch had the same weakness: it silently depended on how one
model words and labels things. Lowering the cut would have passed the test until the next
model change.

**Costs:** the extractor must link activities (an eval checks it); two names for one activity
that never get merged ("gym", "working out") count separately; repeated feelings with no
activity ("tired" three times) no longer roll up. Similarity thresholds remain where meaning
really is fuzzy (recall, relevance, entity matching), and those still need re-measuring after
a model change: the scenario check makes that impossible to miss.

**Alternatives:** keep clustering and re-tune the cut per extraction model (cheap now, breaks
again later); strip dates before embedding (fixes one cause of drift, not labels or wording);
let the LLM group memories into habits directly (no threshold, but no guarantee across runs).

---

## Gaps: what this design does not cover

Honest limits, and what would fill each one:

| Gap | Risk | What would fill it |
|---|---|---|
| **Memory poisoning / prompt injection** (OWASP's 2026 Top 10 for agentic applications lists it as ASI06, memory and context poisoning) | Text the user (or a pasted document) says is stored, then returned in future prompts as trusted memory, and it outlives the session | Store provenance and source type; treat memory as data in the prompt; filter instruction-like text at write time; review high-impact memories |
| **Scale** | One `User` node becomes a supernode; one Neo4j instance; post-filtered vector search | Partition by user (D4); bucket edges by time; dedicated vector index per tenant; read replicas (Enterprise) |
| **Reflection / insights** | No higher-level conclusions ("Alex is happier since moving") | Scheduled LLM reflection writing new `insight` memories with evidence links |
| **Procedural memory learning** | The companion doesn't learn *how* to talk to this person beyond boundaries | Store style feedback as memories that edit the persona prompt; review before applying |
| **Multimodal memory** | Photos, voice, documents | Embeddings for images/audio, `Media` nodes linked `ABOUT` entities |
| **Shared memory across people** | Two users who know the same Maya have two Mayas | Global entity layer with consent; cross-user resolution |
| **Erasure outside the graph** | Backups, logs, traces, LLM provider retention | Retention policies, log redaction, provider no-retention settings |
| **Cost control** | One extraction call per turn, one remote embedding call per recall | Cheaper extraction model (`LLM_EXTRACTION_MODEL`), batching, caching, skipping more small talk |
| **Benchmarks** | Our scenarios aren't comparable with other systems | Run LoCoMo / LongMemEval / BEAM-style suites |
| **Learned thresholds** | Hand-measured cut-offs per model, on hand-written pairs rather than the extractor's real sentences | Collect labelled pairs from the trace UI and scenario runs, and refit |
| **Structural dedupe** | A reworded copy of a memory is caught only by cosine ≥ 0.975 (bge) / 0.966 (Qwen), which depends on wording (D22) | Same entities + kind + day as the primary test, similarity as a fallback |
| **Reasoning memory** | The agent doesn't remember *how* it handled similar requests (which tools, what worked) | Store decision traces as nodes linked to the turn, as Neo4j Labs' `neo4j-agent-memory` does |
| **Non-English** | Role-name rules and regexes are English | Language-aware rules; multilingual embedding model (Qwen3 is multilingual) |

---

## How other memory systems choose

A compressed comparison, to place this design. Products change quickly: treat this as the
shape of each approach, not a feature list.

| System | Storage | Who writes memory | Update model | Retrieval |
|---|---|---|---|---|
| **This tutorial** | Neo4j: graph + vectors | background extractor + end-of-episode pass | supersede, bi-temporal, reinforce | 9 retrievers in code, every turn |
| **MemGPT / Letta** | in-context *core* memory blocks + *archival* store + *recall* (conversation history) store; since 2026 optionally a git-backed file tree (MemFS) | the agent itself via tools, and optionally a *sleep-time* agent in the background | agents edit blocks | agent searches when it chooses |
| **Mem0** | vector store, optional graph store (Neo4j and others) | LLM extraction pipeline | 1.x: LLM picks ADD / UPDATE / DELETE / NOOP; 2.0 (2026): ADD-only, conflicts resolved at retrieval | vector (+ graph) search |
| **Zep / Graphiti** | temporal knowledge graph (Neo4j recommended, FalkorDB and others) | ingestion of *episodes* into entities and fact edges | bi-temporal validity on edges; contradictions invalidate old edges | embeddings + BM25 + graph distance, fused |
| **LangMem** | LangGraph `BaseStore` | hot-path tools and/or a background memory manager | *profile* (one schema) or *collection* (many items) | store search |
| **Cognee** | graph + vector + relational, pluggable backends (Neo4j, FalkorDB, Kuzu/LadybugDB…) | an ECL pipeline: *add*, *cognify* (extract a graph), optional *memify* enrichment | re-run the pipeline; enrichment infers new links | graph and vector search combined |
| **Neo4j Labs `neo4j-agent-memory`** | Neo4j: short-term (messages), long-term (a POLE+O entity graph), *reasoning* memory (decision traces) | extraction pipeline with deduplication | entity merge | Cypher tools, also as an MCP server; a Labs project, so no support guarantees |
| **Cloud memory services** (Bedrock AgentCore Memory, Vertex AI Memory Bank) | managed store of extracted memory records per scope (user, session) | the service, asynchronously, from stored conversation events | the service consolidates (Memory Bank asks Gemini to merge and resolve contradictions) | similarity search within one scope |
| **Files the agent edits** (Anthropic memory tool, Letta MemFS, OpenAI Agents SDK sessions for history only) | Markdown files, versioned with git in Letta's case | the agent, with file tools | edit in place, history in git | file search and reads, no vector index by default |
| **Microsoft GraphRAG** | entity graph + community summaries over a corpus | offline indexing | re-index | local, global, DRIFT; LazyGraphRAG defers summarization to query time |

Graphiti is the closest relative: also a temporal knowledge graph on Neo4j, with episodes as
provenance. The main differences here are:
- a memory is a sentence node pointing at entities, rather than a fact on an edge;
- the explicit relation ontology with rules;
- per-persona visibility;
- keyword hits fused into the vector retriever rather than kept as a separate list (D11).

Letta's sleep-time agents and our background extractor plus `consolidate()` solve the same
problem: memory work that shouldn't slow down the reply.

Neo4j's own Labs package is the closest *on the same database*. Its three layers map onto
ours: short-term messages ≈ the checkpointer plus `Episode` nodes, long-term entity graph ≈
the life graph, reasoning memory ≈ nothing here (we don't store the agent's own decision
traces; [Lesson 18](../lessons/18-seeing-memory-work.md)'s trace is shown, not stored).

The 2026 trend is in two directions at once: cloud platforms hide memory behind an API with
scopes and similarity search, while coding agents move memory into plain files the model edits.
Both drop the graph. Keep one when your questions are about *relationships and time*, which a
companion's are.

*Checked against public documentation in November 2026. Sources are listed at the end of the
[glossary](glossary.md#sources-checked).*
