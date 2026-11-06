# Glossary — overlapping terms in graph, memory and agent work

Many of these terms come from different fields (psychology, databases, the Semantic Web,
NLP, information retrieval, LLM tooling) and get used loosely. Each entry says what it means,
where it shows up in this repo, and what it is often confused with. Words with several
meanings (*semantic*, *ontology*, *recall*, *profile*...) have their own section at the end,
one sense per row. The reasons behind each design choice, and the alternatives, are in
[design-decisions.md](design-decisions.md).

**Jump to:** [Kinds of memory](#kinds-of-memory) · [Where memory lives](#where-memory-lives) ·
[Graph modeling](#graph-modeling) · [Ontology terms](#ontology-terms) ·
[Extraction](#extraction-text--structure) · [Retrieval](#retrieval) ·
[Time and change](#time-and-change) · [Memory upkeep](#memory-upkeep-and-quality) ·
[Privacy](#privacy-and-safety) · [Agents and LangGraph](#agents-and-langgraph) ·
[Evaluation](#evaluation) · [Concurrency](#concurrency-and-transactions) ·
[Graph data science](#graph-data-science) · [Words with several meanings](#words-with-several-meanings)

---

## Kinds of memory

From cognitive psychology (Tulving's episodic/semantic split, 1972), borrowed by agent
frameworks such as the CoALA paper and LangMem.

| Term | Meaning | In this repo | Often confused with |
|---|---|---|---|
| **Working memory** | What's "in mind" for the current step | LangGraph state for one turn: messages + the recalled memory block | the context window (the *capacity*, not the contents) |
| **Short-term memory** | Recent conversation, kept for a session | The checkpointer's messages per `thread_id` (last 16 sent to the LLM) | working memory (often used interchangeably) |
| **Long-term memory** | Persists across sessions | Everything in Neo4j | — |
| **Episodic memory** | Experiences: what happened, when, in what context | `Episode`/`Turn` nodes, `event` memories, diary summaries | events in a *timeline* (episodic memory also keeps the context and source) |
| **Semantic memory** | General facts, true independent of when they were learned | `fact`/`preference` memories, the life graph | "semantic search" (see [semantic](#semantic)) |
| **Procedural memory** | How to do things / how to behave | Persona prompt, `boundary` memories, extraction rules | preferences (what they like vs how to act) |
| **Autobiographical / biographical memory** | Episodic + semantic memory about one person's life | The whole companion design ([Lesson 8](../lessons/08-memory-design-for-companions.md)) | a user profile (a flat list; biography has history and relationships) |
| **Prospective memory** | Remembering to do something in the future | `plan` memories with `due_on`, follow-ups, yearly dates | reminders/alarms (prospective memory also chooses *when* it's relevant) |
| **Parametric memory** | Knowledge stored in model weights | What the LLM "knows" without tools | fine-tuning on user data (not done here) |
| **Non-parametric (external) memory** | Knowledge stored outside the model and retrieved | Neo4j + vector indexes | RAG (the retrieval *technique* over such memory) |

## Where memory lives

| Term | Meaning | In this repo | Often confused with |
|---|---|---|---|
| **Context window** | Max tokens the LLM sees per call | Persona + memory block + recent messages | memory (it's only what's loaded *now*) |
| **Context engineering** | Choosing what goes into the context window | `recall()` + `format_context()` | prompt engineering (wording vs selection) |
| **Checkpointer / thread** | LangGraph's per-conversation state saver | `InMemorySaver`, `thread_id` = episode id | long-term memory |
| **LangGraph Store** | LangGraph's own long-term key-value/document store (`BaseStore`, namespaces) | Not used; Neo4j plays this role | the checkpointer |
| **Vector store / vector database** | Stores embeddings for similarity search | Neo4j vector indexes | a knowledge graph |
| **Graph database** | Database whose model is nodes + relationships, optimized for traversal | Neo4j | a knowledge graph (a graph DB is the engine; a KG is content) |
| **MemGPT / Letta memory tiers** | *Core* memory (always in context), *recall* memory (conversation history), *archival* memory (searchable store) | Our "core profile" ≈ core memory; our `recall()` is **not** MemGPT's recall memory | our `recall()` function |
| **Hot path vs background memory** | Writing memories during the reply (agent calls a tool) vs after/outside it | Our extractor runs after the reply; [Lesson 11](../lessons/11-harder-scenarios-and-evals.md) compares both | synchronous vs async (a background writer can still be synchronous) |
| **Profile vs collection memory** (LangMem) | One document updated in place vs many small memories | We use a collection (Memory nodes) + a derived core profile | — |

## Graph modeling

| Term | Meaning | In this repo | Often confused with |
|---|---|---|---|
| **Labeled property graph (LPG)** | Nodes and relationships with labels/types and key-value properties | Neo4j's model | RDF |
| **RDF / triple store** | Data as subject–predicate–object triples, identified by URIs | Not used (`neosemantics` can import RDF/OWL into Neo4j) | LPG |
| **Triple (SPO)** | One subject–predicate–object statement | `user -sister-> Maya` is a triple-shaped fact | a memory (ours are sentences that *point at* triples) |
| **Knowledge graph (KG)** | A graph of entities and facts about them, usually with an ontology | The life graph + memories + RelType ontology | a graph database; GraphRAG |
| **Temporal knowledge graph** | A KG whose facts carry validity times | `r.until`, `invalidated_at`, `occurred_on` | event logs |
| **Schema** | The structural rules: labels, properties, constraints, indexes | `companion/schema.py` | ontology (schema = shape; ontology = meaning) |
| **Reification** | Turning a relationship into a node so it can have its own links | `Memory` nodes ([Lesson 3](../lessons/03-modeling-and-performance.md)) | — |
| **Traversal / hop** | Following relationships from node to node | Entity hop, `SUPERSEDES*1..3` | a join (traversal follows stored pointers) |
| **Supernode** | Node with a huge number of relationships | `User` grows into one over time | — |
| **Cypher / GQL / SPARQL / Gremlin** | Graph query languages: Neo4j's / the ISO standard (ISO/IEC 39075, 2024, close to Cypher) / RDF's / TinkerPop's | Cypher everywhere | — |

## Ontology terms

See [Lesson 12](../lessons/12-ontology.md).

| Term | Meaning | In this repo | Often confused with |
|---|---|---|---|
| **Ontology** | Formal description of the kinds of things and relations in a domain, and rules about them | `companion/ontology.py`, mirrored as `(:RelType)` nodes | schema; taxonomy |
| **Taxonomy** | A hierarchy only (is-a / subtype tree) | The `SUBTYPE_OF` part of the ontology | an ontology (which also has rules) |
| **Controlled vocabulary** | An agreed list of allowed terms | The canonical relation types shown to the extractor | free text |
| **Thesaurus / SKOS** | Vocabulary + synonyms and broader/narrower terms (SKOS is the W3C standard) | `synonyms` + `SUBTYPE_OF` (≈ SKOS altLabel + broader) | an ontology with logic (OWL) |
| **OWL / RDFS** | W3C languages for ontologies with formal logic / basic class hierarchies | Not used; ours is a lightweight Python + Neo4j version | — |
| **TBox / ABox** | Terminology (types and rules) vs assertions (facts about individuals) | RelType nodes vs RELATED edges | — |
| **Class / instance** | A type vs a member of it | `kind: 'person'` vs Maya | label vs node |
| **Domain / range** | Allowed source / target types of a relation | `range`: `pet` → a pet, `lives_in` → a place | data types |
| **Cardinality** | How many targets a relation may have | `lives_in`, `partner`: one at a time | uniqueness constraints |
| **Symmetric / inverse / transitive property** | A→B implies B→A / B→A under another name / A→B→C implies A→C | `sibling` symmetric; `parent`↔`child`; SUBTYPE_OF is transitive | — |
| **Inference / reasoning / entailment** | Deriving facts not stated | Kinship composition: brother's daughter → niece | LLM "reasoning" (statistical, not rule-based) |
| **Open- vs closed-world assumption** | Missing fact = unknown vs missing fact = false | Our recall is closed-world in practice; the persona says "ask instead of guessing" to stay open-world in conversation | — |
| **Canonicalization / normalization** | Mapping variants to one standard form | `girlfriend` → `partner`, `daughter_of` → flipped `daughter` | entity resolution (that's for *things*, this is for *types*) |

## Extraction (text → structure)

| Term | Meaning | In this repo | Often confused with |
|---|---|---|---|
| **Information extraction (IE)** | Turning text into structured records | `extract_memories()` → `Extraction` | summarization |
| **Named-entity recognition (NER)** | Finding names of people, places, orgs in text | `entities` in the extraction | entity resolution |
| **Relation extraction** | Finding typed links between entities | `relations` in the extraction | — |
| **Coreference resolution** | Working out what pronouns refer to ("she" → Priya) | Earlier turns given as context ([Lesson 11](../lessons/11-harder-scenarios-and-evals.md)) | entity resolution |
| **Entity resolution / linking / record linkage** | Deciding two mentions are the same real-world thing | `resolve_entity()`: ref, name/alias, embedding + name similarity | deduplication of *memories* |
| **Deduplication** | Not storing the same fact twice | index score ≥ the model's `duplicate` threshold (0.975 for bge-small) + same dates → reinforce | supersession (a *change*, not a copy) |
| **Structured output / function calling** | Making the LLM return data matching a schema | `with_structured_output(Extraction, method="function_calling")` | JSON mode (no schema guarantee) |

## Retrieval

| Term | Meaning | In this repo | Often confused with |
|---|---|---|---|
| **Embedding** | A vector representing meaning | `bge-small` (384 dims, local) or Qwen3-Embedding-8B (4096, remote) | a graph embedding (structure, e.g. `gds.fastRP`) |
| **Embedding profile** | An embedding model together with the thresholds measured for it | `companion/embeddings.py`: bge-small (local, 384 dims) or Qwen3-Embedding-8B (remote, 4096); each writes its own property and indexes (Lesson 19) | a model name alone (its scores mean nothing without its thresholds) |
| **Query instruction** | Text some models expect before a *query* but not before stored text | Qwen: "Instruct: Given a message from a user, retrieve memories... Query: " (Lesson 19) | a prompt to an LLM |
| **Calibration** | Picking a threshold from labelled yes/no pairs | `python -m companion.calibrate` (Lesson 19) | probability calibration in classifiers |
| **Cosine similarity** | Angle-based vector similarity | Neo4j's vector score is cosine *rescaled* to 0–1: (1 + cos) / 2 | distance |
| **ANN / HNSW** | Approximate nearest-neighbour search / the graph-based index Neo4j's vector index uses | `db.index.vector.queryNodes` | exact k-NN |
| **Lexical / keyword search, BM25** | Matching words, scored by term frequency | Not used (Neo4j full-text indexes provide it) | semantic search |
| **Hybrid search** | (a) vector + keyword, or (b) vector + graph | We do (b): vector entry points + graph hops | — |
| **Reranking** | Re-scoring retrieved items with another signal | similarity × importance × recency | retrieval |
| **RAG** | Retrieval-augmented generation: fetch, then put it in the prompt | `recall()` → memory block | fine-tuning |
| **GraphRAG** | RAG that uses a graph. Microsoft's GraphRAG builds community summaries over a whole corpus; "local" GraphRAG traverses from entry points | [Lesson 5](../lessons/05-vectors-and-graphrag.md) and the companion use the local style | — |

## Time and change

| Term | Meaning | In this repo | Often confused with |
|---|---|---|---|
| **Valid time** | When something is true in the world | `occurred_on`, `due_on`, `valid_until`, `r.since` / `r.until` | when it was recorded |
| **Transaction / system time** | When the system recorded or retired it | `created_at`, `invalidated_at` | valid time |
| **Bi-temporal** | Tracking both times | Memories ([Lesson 8](../lessons/08-memory-design-for-companions.md)) | versioning |
| **Supersession / versioning** | A new version replaces an old one, which is kept | `(new)-[:SUPERSEDES]->(old)` | deletion |
| **Soft vs hard delete** | Mark as gone vs actually remove | Retire (`invalidated_at`) vs `/forget` (DETACH DELETE) | — |
| **Decay / forgetting curve** | Relevance fading with time (Ebbinghaus) | `companion/decay.py`: strength halves every half-life (status 1 day, feeling 7, event 90) ([Lesson 13](../lessons/13-memory-over-time.md)) | expiry (decay is about relevance, not truth) |
| **Half-life** | Time for strength to halve | `decay.HALF_LIFE_DAYS`; facts have none | TTL |
| **Dormant** | Too weak to recall, but still stored | strength < 0.05 and importance < 8 | deleted |
| **Stale** | Possibly out of date: old and never re-confirmed | facts not confirmed for a year get a "check first" marker | expired (known to be over) |
| **Expiry / TTL** | A time after which something stops counting. A database TTL *deletes* it | `valid_until`, filtered at read time; APOC's `apoc.ttl` would delete instead | decay |
| **Status (situational memory)** | Short-lived current state: "at the gym" | memory kind `status`, always with `valid_until` | feeling, event |
| **As-of query** | "What did we believe at time T?": filter on transaction time | `memories(today=T)` requires `created_at <= T` | a valid-time filter |

## Memory upkeep and quality

| Term | Meaning | In this repo | Often confused with |
|---|---|---|---|
| **Consolidation** | Re-processing memories offline: merge, fix, summarize | End-of-episode pass; `consolidate()` rolls repeats into patterns, lapses old plans, prunes trivia ([Lesson 13](../lessons/13-memory-over-time.md)) | summarization alone |
| **Pattern / roll-up** | One lasting memory standing for many similar ones: episodic → semantic | "works out at the gym regularly", `(pattern)-[:SUMMARIZES]->(member)`, `evidence` count | a summary of one conversation |
| **Complete linkage** | Clustering where every member of a group must be similar to *every* other member (this repo finds the largest such group first) | `_clusters()` in store.py, threshold per embedding model (0.82 for bge-small) | single linkage (similar to *any* member; chains) |
| **Pruning** | Deleting memories on purpose (data minimization) | dormant statuses/feelings with importance ≤ 3, after consolidation | decay (keeps them) |
| **Reflection** | Generating higher-level insights from many memories (Generative Agents) | Not yet (a "where to go next" idea) | consolidation |
| **Conflict / contradiction resolution** | Deciding which of two incompatible facts is current | `supersedes`, `invalidate`, cardinality rules | dedup |
| **Provenance / lineage** | Where a fact came from | `(m)-[:FROM]->(Episode)`, `r.said` | history (versions) |
| **Grounding** | Answers based on supplied facts | Persona: "only rely on memories shown here" | citing |
| **Hallucination** | Confident, unsupported output | Extractor rule: "never invent details" | a wrong memory (could be a true statement that changed) |
| **Echo** (our term) | Storing the assistant's own recall as a new fact | Anti-echo rule, "context only" labels ([Lesson 11](../lessons/11-harder-scenarios-and-evals.md)) | dedup |

## Privacy and safety

| Term | Meaning | In this repo | Often confused with |
|---|---|---|---|
| **PII** | Personally identifiable information | Most of the companion's memory | sensitive data (a stricter subset) |
| **Sensitive / special-category data** | Health, sexuality, religion, etc. (GDPR Article 9) | `sensitive` flag, recalled only when on-topic | secrets/credentials |
| **Right to erasure** | A person can have their data deleted (GDPR Article 17) | `/forget`, `forget_entity()` with redaction | retiring a fact |
| **Redaction** | Removing identifying text while keeping the record | `[forgotten]` in turns and summaries | deletion |
| **Data minimization** | Store only what's needed | "Record only what's worth remembering", small-talk gate | — |
| **Boundary** (our term) | A topic or person the user asked not to raise | `boundary` memories, filtered in Cypher | a forget request (boundary keeps the memory, hides the topic) |
| **Persona** (our term) | One of several companions sharing a user's memory | `agent` on memories, relations and episodes; `--persona Coach` (Lesson 16) | a user (personas are the *other* side of the conversation) |
| **Private memory** | Visible only to the persona that heard it | `private` flag: "just between us", or sensitive; never boundaries | sensitive (sensitive memories are private, not every private one is sensitive) |
| **Visibility rule** | The filter deciding what a reader may see | `visible()` and `entity_visible()` in store.py, pasted into every read query | access control on the database (this is in the queries, not in Neo4j roles) |
| **Derived data** | Data computed from other data: summaries, patterns, entities, embeddings | must be hidden along with the private memory it came from | the source data |

## Agents and LangGraph

| Term | Meaning | In this repo | Often confused with |
|---|---|---|---|
| **Agent** | An LLM that chooses actions (tools) in a loop | `team_agent.py` | a workflow (fixed steps; the companion is closer to this) |
| **Tool calling** | The LLM asks to run a function with arguments | `@tool` functions, `ToolNode` | structured output (same mechanism, different purpose) |
| **ReAct** | Reason → act → observe loop | The agent ⇄ tools loop | — |
| **State graph / node / edge** | LangGraph's model: typed state, steps, transitions | `StateGraph(CompanionState)` | the Neo4j graph (unrelated graph!) |
| **Conditional edge** | A transition chosen by a function | `tools_condition`, `worth_remembering` | a tool call |
| **Background worker** | Code that runs a job on another thread so the caller doesn't wait | `companion/worker.py`: one thread, jobs in order (Lesson 14) | async/await (one thread taking turns; here it's a real second thread) |
| **Fake** (test double) | A simple working stand-in for a real dependency | `FakeListChatModel`, fake extractors and pattern writers | a mock (which checks *how* it was called) |
| **Trace** | A record of what each step of one request did, with timings | `build_companion(..., trace=[])`: recall, prompt, extraction decisions per turn (Lesson 18) | a log (lines in time order, not grouped by request) |
| **Retriever** | One way of finding candidates for the prompt | recall runs nine: core, vector, graph hop, plans, dates, recent, history, people, episodes | the retrieval *result* (what they return together) |

## Evaluation

| Term | Meaning | In this repo | Often confused with |
|---|---|---|---|
| **Unit test** | Checks one function with no external services | `tests/test_12_ontology.py` (`normalize`, `infer_kin`) | integration test |
| **Integration test** | Checks code against a real dependency | `db` tests against Neo4j, writes rolled back | end-to-end test |
| **Fixture** | Setup/teardown shared by tests | `driver`, `tx` (rolled back), `store` (throwaway user) in `tests/conftest.py` | test data (fixtures *provide* it) |
| **Smoke test** | Cheapest check that the system is up at all | `tests/test_00_setup.py` | sanity check (same idea) |
| **Eval** | A test of non-deterministic (LLM) behaviour, judged by a pass *rate* | `pytest -m llm`, `EVAL_TRIALS=3` | benchmark (a fixed public dataset) |
| **Golden case / golden set** | A fixed input with known-good expectations | the 10 cases in `test_11_extraction_evals.py`, `RETRIEVAL_CASES` | "golden file" (exact expected output) |
| **Trajectory eval** | Grading the *steps* an agent took, not only its answer | `test_agent_picks_the_right_tool` checks which tool was called | outcome eval |
| **Scenario test** | Scripted multi-turn conversation with assertions | `companion/scenarios.py`, `test_11_scenarios.py` | unit test (deterministic, one function) |
| **Regression test** | A test added for a bug, so it can't come back | `test_seattle_is_not_merged_into_boston`, the boundary-about-user test | re-running the suite (that's how they're used) |
| **LLM-as-judge** | An LLM grades outputs against a rubric | Not used: every check is code (fields, ids, regex, Cypher) | — |
| **hit@k** | Share of questions whose right answer is in the top k | `test_retrieval_hit_rate` (Lesson 5) | recall@k (share of *all* relevant items found) |
| **Precision / recall@k** | Share of retrieved items that are relevant / share of relevant items retrieved | Not measured: each question has one right answer, so hit@k is used | our `recall()` function |
| **Flakiness** | Same test, different results across runs | Why evals run several trials | a bug (it may be one) |
| **Fault injection** | Making a step fail on purpose to test recovery | `monkeypatch` raising mid-`apply()` (Lesson 14) | chaos engineering (the same idea, in production) |
| **Mutation check** | Breaking the code on purpose to see that a test fails | `visible()` → `"true"`: the four privacy tests failed (Lesson 16) | mutation testing tools (automated versions) |

## Concurrency and transactions

| Term | Meaning | In this repo | Often confused with |
|---|---|---|---|
| **Transaction** | A group of writes that commit or roll back together | `MemoryStore._write()`: one per `apply()` (Lesson 14) | a session (a connection that can run several transactions) |
| **Managed transaction** | The driver runs your function and retries it on transient errors | `session.execute_write(unit)` | an auto-commit query (`execute_query`, one statement) |
| **Rollback** | Undoing every write of an unfinished transaction | what happens when `apply()` raises | deleting data afterwards |
| **Isolation level** | What one transaction sees of others | Neo4j: read committed. You never see others' uncommitted writes | serializable (would prevent the duplicate race, Neo4j's default doesn't) |
| **Race condition** | The result depends on timing | check-then-act duplicate check: 4 writers → 4 copies without a lock | a deadlock |
| **Write lock** | Held on a node you write to, until commit | `SET u.write_lock = datetime()` on the user node serializes writers per user | a Python `threading.Lock` (one process only) |
| **Deadlock** | Two transactions each wait for a lock the other holds | Neo4j detects it and fails one, and `execute_write` retries it | a slow query |
| **Thread-local** | A value each thread has its own copy of | the open transaction (`self._local.tx`), so recall never joins the worker's write | a global |

## Graph data science

| Term | Meaning | In this repo | Often confused with |
|---|---|---|---|
| **Projection** | An in-memory copy of part of the graph for GDS algorithms | `gds.graph.project(...)` per user (Lesson 15), always dropped | a Cypher projection of properties (`m {.id, .text}`) |
| **Graph embedding** | A vector for a node, from its position in the graph | FastRP in `companion/insights.py` | a text embedding (from words) |
| **FastRP** | Fast Random Projection: averages random neighbour vectors over 1, 2, … hops | `iterationWeights: [0.0, 1.0, 1.0]` | node2vec (random walks, slower) |
| **kNN** | k-nearest neighbours of each node's vector | `gds.knn.stream`, cut at 0.6 (0.5 = unrelated) | a vector index search (one query vector) |
| **Node similarity** | Jaccard over direct neighbours | `method="jaccard"`: exact, explainable, one hop | cosine similarity of embeddings |
| **Random seed** | Fixes a randomized algorithm's choices | `randomSeed: 42, concurrency: 1` for repeatable tests | — |

---

## Words with several meanings

The same word names different things in psychology, databases, information retrieval and LLM
tooling. Each entry lists the senses, says **which ones this repo uses and where**, and gives a
quick test for telling them apart. The lessons link here when they use one of these words, and
[design-decisions.md](design-decisions.md) says which sense it means each time.

Quick index: [semantic](#semantic) · [ontology](#ontology) · [knowledge graph](#knowledge-graph) ·
[graph](#graph) · [memory](#memory) · [recall](#recall) · [profile](#profile) ·
[episode](#episode) · [entity](#entity) · [schema](#schema) · [embedding](#embedding) ·
[similarity and score](#similarity-and-score) · [index](#index) · [hybrid](#hybrid) ·
[rerank and relevance](#rerank-and-relevance) · [inference](#inference) ·
[context](#context) · [agent](#agent) · [forget](#forget) · [instruction](#instruction)

### semantic

| # | Sense | Field | In this repo |
|---|---|---|---|
| 1 | **semantic memory**: general facts, true independent of when they were learned (vs *episodic*: experiences) | cognitive psychology; CoALA, LangMem | `fact`/`preference` memories, the life graph, consolidated habits (Lessons 8, 13) |
| 2 | **semantic search / semantic similarity**: comparing meaning through embeddings, not words | information retrieval | vector search over `memory_vec`; the retriever called *semantic* in `recall()` (Lessons 5, 10) |
| 3 | **Semantic Web**: RDF, OWL, SPARQL and linked data standards | W3C | not used; our ontology is a lightweight cousin (Lesson 12) |
| 4 | **semantic layer / semantic model**: business definitions of metrics and entities over a data warehouse | BI and analytics | not used. The closest thing here is the ontology: an agreed meaning for relation types |
| 5 | **semantic** as in "what it means" in general: semantic deduplication, semantic chunking, semantic cache | everyday ML usage | dedupe by meaning, not exact text (Lesson 9) |

**Watch out:** `recall()` returns a list called `semantic`. That's sense 2 (it came from vector
search), and it can contain episodic memories (events, feelings). Semantic *memory* (sense 1)
is a kind of knowledge. Semantic *search* (sense 2) is a way of finding things. They're
independent: you can vector-search episodes, and look up facts by exact name.

**Test:** if you can replace the word with "vector-based", it's sense 2. If you can replace it
with "general-fact", it's sense 1.

### ontology

| # | Sense | In this repo |
|---|---|---|
| 1 | **formal ontology**: classes, properties and logical axioms in OWL/RDFS, used by a reasoner | not used |
| 2 | **lightweight ontology**: a controlled vocabulary of types with a hierarchy and a few rules (symmetry, inverse, cardinality, range) and some inference | `companion/ontology.py` → `(:RelType)` nodes (Lesson 12) |
| 3 | loosely, **any schema or data model** ("our ontology has Person and Company") | avoided; we say *schema* for structure (Lesson 8) |
| 4 | **philosophy**: the study of what exists | — |

**Watch out:** schema = *shape* (labels, properties, constraints). Ontology = *meaning* (what
`sister` is, that it implies `sibling`, that `lives_in` holds one place at a time). An
ontology also differs from a *taxonomy* (only the is-a tree) and from a *knowledge graph*
(the facts the ontology describes). In TBox/ABox terms: the ontology is the TBox, the life
graph the ABox.

### knowledge graph

| # | Sense | In this repo |
|---|---|---|
| 1 | a graph of **entities and facts**, usually typed by an ontology | the life graph + memories + RelType nodes |
| 2 | Google's **Knowledge Graph** product (the info panels in search) | — |
| 3 | marketing for **any graph database** | avoided: the database is Neo4j, the KG is the content |
| 4 | a **temporal knowledge graph**: facts with validity times | ours is one (`since`/`until`, `invalidated_at`); so is Graphiti |

### graph

(1) the **Neo4j data graph**: nodes and relationships; (2) a **LangGraph state graph**: the
control flow of the companion (recall → respond → remember), whose "nodes" are Python
functions; (3) a **chart**; (4) in GDS, a **projected graph**: an in-memory copy for
algorithms (Lessons 4, 15). Sentences like "the graph calls recall" mean sense 2. "Recall
walks the graph" means sense 1.

### memory

(1) **Any stored knowledge** the companion has; (2) one **`Memory` node** (a sentence with a
kind and dates); (3) the **kinds of memory** from psychology (episodic, semantic, procedural,
prospective); (4) the checkpointer's **short-term thread state**; (5) **RAM**: Neo4j's page
cache and heap, GDS's in-memory projections. "Memory decays" is sense 2 (a node's strength).
"GDS needs memory" is sense 5.

### recall

(1) our **`recall()`**: fetch what's relevant for one turn; (2) the **retrieval metric**: the
share of relevant items found (recall@k), not measured here (we use hit@k); (3) MemGPT/Letta's
**recall memory**: the searchable conversation history; (4) **human remembering**. Lesson 18's
*Try recall* tab is sense 1.

### profile

(1) the **core profile**: boundaries plus the most important facts, always in the prompt
(Lesson 10); (2) an **embedding profile**: a model plus its measured thresholds (Lesson 19);
(3) LangMem's **profile memory**: one document per user updated in place, as opposed to a
*collection* of small memories; (4) a **user profile** in product terms; (5) Cypher's
**`PROFILE`**: an executed query plan (Lesson 3).

### episode

(1) one **conversation**: an `Episode` node with turns, summary and mood; (2) **episodic
memory** in general: experiences with time and context; (3) an episode in **reinforcement
learning**; (4) Graphiti's *episode*: one ingested chunk of input data. Sense 1 is the
provenance unit for sense 2.

### entity

(1) an **`Entity` node**: a person, pet, place, organization, activity or thing in the user's
life; (2) a **named entity** in NER: a span of text such as "Maya"; (3) a **database
row/object** (an ORM entity). The extractor turns sense 2 into sense 1 through *entity
resolution*.

### schema

(1) **Neo4j constraints and indexes** (`companion/schema.py`); (2) a **Pydantic/JSON schema**
that structured output must match (`Extraction`); (3) loosely, an **ontology** (see above).

### embedding

(1) a **text embedding**: a vector from words, about meaning (Lessons 5, 19); (2) a **graph
embedding**: a vector from a node's position in the graph (FastRP, Lesson 15); (3) embedding a
widget in a page. Senses 1 and 2 can disagree on purpose: text says what things are *about*,
structure says what they're *connected to*.

### similarity and score

(1) **cosine similarity**: −1 to 1, computed in Python; (2) Neo4j's **vector index score**:
cosine rescaled to 0–1 as (1 + cos) / 2, so 0.5 means unrelated; (3) GDS **kNN similarity**:
also 0–1; (4) **Jaccard** node similarity: shared neighbours / all neighbours; (5) a
**cross-encoder relevance** logit: unbounded, about −11 to +5 (Lesson 18). Thresholds only
make sense on one scale *and* one model: see Lesson 19's calibration table, which labels each
"score" or "cosine".

### index

(1) a **database index** for lookups (`memory_user_kind`); (2) a **vector index** (HNSW) for
nearest-neighbour search (`memory_vec`); (3) a **full-text index** (Lucene, BM25), available
but not used; (4) "indexing" in RAG: the offline step that chunks and embeds documents.

### hybrid

(1) **vector + keyword** search (the common meaning in RAG products); (2) **vector + graph**
retrieval: what this repo means by hybrid recall; (3) **LLM + rules**, such as the boundary
tripwire or the ontology checks on extracted relations.

### rerank and relevance

(1) our **ranking formula**: similarity × importance × decay (`decay.rank`, Lessons 10, 13);
(2) a **reranker model** (cross-encoder) that reads query and text together (Lesson 18);
(3) **relevance** as a human judgement: is this memory about this message? Lesson 18 measures
(3) with (2). Lesson 19 measures it with Qwen's vector score.

### inference

(1) **logical inference / entailment**: deriving facts from rules (a brother's daughter is a
niece; `sister` implies `sibling`) (Lesson 12); (2) **model inference**: running an LLM or
embedding model ("inference endpoint", "inference cost"); (3) the LLM **inferring** an
unstated link ("my meds don't help" → the ADHD memory), which is statistical, not rule-based.
"Inference provider" is sense 2. "Inferred relatives" is sense 1.

### context

(1) the **context window**: what the model sees in one call; (2) **context engineering**:
choosing what goes in it (`recall()` + `format_context()`); (3) earlier turns given to the
extractor **as context only**, not as a source of new facts (Lesson 11); (4) a driver
**session** or transaction context.

### agent

(1) an **LLM that chooses actions** (tools) in a loop: `team_agent.py` (Lesson 7); (2) the
**`agent` property** on memories: which *persona* heard it (Lesson 16); (3) loosely, any LLM
application. The companion is a workflow (fixed steps), not an agent in sense 1: see
[D19](design-decisions.md#d19).

### forget

(1) **hard delete** with redaction (`/forget`, Lesson 11); (2) **retire / supersede**: no longer
current, still stored (Lesson 9); (3) **decay**: still stored and current, just ranked lower
or dormant (Lesson 13); (4) **pruning**: deleting low-value memories on purpose (Lesson 13).
Only (1) and (4) remove data.

### instruction

(1) a **system prompt** or persona instruction to the chat model; (2) an embedding model's
**query instruction**: text put before a query so the model embeds it for retrieval
(Lesson 19); (3) instruction-tuned models in general. Sense 2 is never seen by the chat model.
