# Lesson 6 — Graphs in AI and AI agents (concepts)

Graph databases give AI systems something LLMs lack on their own: **explicit, connected facts**
they can look up, reason over and cite.

## How AI systems use graphs

1. **GraphRAG** — vector search finds starting nodes, traversal gathers connected facts
   (Lesson 5). A variant runs community detection (Louvain, Lesson 4) and has an LLM summarize
   each cluster to answer broad "what are the main themes?" questions.
2. **Long-term agent memory** — extract entities and facts from conversations into nodes and
   edges. Add `valid_from` / `valid_to` on relationships so changed facts are expired, not
   deleted ("where is Alice now?" vs "where was she in March?"). Use `MERGE` so one entity
   stays one node across many conversations.
3. **The graph as a tool** — either text-to-Cypher (flexible; use a **read-only user** and a
   timeout) or fixed tools like `find_experts(skill)` (predictable; start here for production).
4. **Building the graph with an LLM** — extract `(entity)-[rel]->(entity)` triples from docs,
   tickets and chats once; then cheap, deterministic Cypher does the work.
5. **Planning and coordination** — task dependency graphs, code graphs (calls / imports),
   permission graphs (`(agent)-[:CAN_ACCESS]->(doc)`).
6. **Traceable answers** — every fact maps to a node or edge you can show as the source.

## Do embeddings help?

**Yes for finding where to start; no for exact or multi-hop reasoning.**

| Task | Embeddings | Graph |
|---|---|---|
| Fuzzy matching ("recommendation engine" → Eve) | ✅ | ❌ needs exact names |
| Exact lookups ("Alice's manager") | ❌ | ✅ one hop |
| Counting / grouping | ❌ | ✅ `count()` |
| Multi-hop | ❌ | ✅ |
| Entity resolution ("A. Smith" = "Alice Smith"?) | ✅ | + `MERGE` once resolved |
| "Like this one" | ✅ text similarity | ✅ structural similarity |

- **Text embeddings** (what `embed.py` makes) capture what a description *means*.
- **Graph embeddings** (`gds.fastRP`, node2vec) capture *where a node sits* in the graph —
  useful for link prediction and structure-based recommendations.

Pitfalls: embed descriptions/chunks, not bare names; use the same model for storing and
querying; re-embed when text changes; don't embed things better reached by traversal.

> **Rule of thumb:** embeddings find where to start, the graph reasons from there, the LLM
> explains the result.

## Ideas for next lessons

- Agent memory: record facts from conversation with `MERGE` and time-stamped edges, recall later.
- Graph tools for an agent: `find_experts`, `who_manages`, plus a read-only Cypher tool.
- Graph embeddings: run `gds.fastRP` and compare with Lesson 5's text similarity.
