"""Lesson 15 — graph embeddings on the life graph: who and what belong together?

Two ways to ask "what is related to Cy?" with Graph Data Science (GDS):

- node similarity   Jaccard over direct neighbours: Cy and Ana are similar if the SAME
                    memories mention both. Exact and explainable, but blind to anything
                    two hops away.
- FastRP + kNN      every node gets a vector built from random projections of its
                    neighbourhood (1 hop, then 2 hops...), and kNN finds the nearest
                    vectors. Cy and Ana come out close if they connect to the same THINGS,
                    even if no single memory mentions both.

Both run on an in-memory projection of ONE user's graph: their entities (not the user node,
which touches everything) and their current memories, joined by ABOUT and RELATED edges.
Memories that are sensitive, retired, expired, private to another persona (Lesson 16), or about
a boundary-protected entity are left out, as are the protected entities themselves: an
"insight" must not route around a boundary.
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager

from companion.store import MemoryStore, entity_visible, visible

EMBEDDING_DIM = 128    # measured: 64 left the real links (Cy~Ana 0.64) close to the cut
SEED = 42              # FastRP and kNN are randomized: fix the seed (and concurrency 1) to repeat results
# kNN fills every node's top-k, related or not. Unrelated pairs measured ~0.5 at every
# embedding size: kNN reports cosine rescaled from [-1, 1] to [0, 1], so 0.5 = "unrelated".
MIN_SCORE = {"fastrp": 0.6, "jaccard": 0.0}

PROJECT = """
MATCH (s:Entity|Memory {user: $user}) WHERE NOT s:User
  AND (s:Entity OR (s.invalidated_at IS NULL AND (s.valid_until IS NULL OR s.valid_until > $now)
                    AND NOT coalesce(s.sensitive, false) AND s.kind <> 'boundary'
                    AND """ + visible("s") + """))
  AND (s:Memory OR """ + entity_visible("s") + """)
  AND NOT EXISTS { MATCH (s)-[:ABOUT]->(p:Entity)<-[:ABOUT]-(b:Memory {kind: 'boundary'})
                   WHERE b.invalidated_at IS NULL AND NOT p:User }
  AND NOT EXISTS { MATCH (s)<-[:ABOUT]-(b:Memory {kind: 'boundary'}) WHERE b.invalidated_at IS NULL }
OPTIONAL MATCH (s)-[r:ABOUT|RELATED]->(t:Entity {user: $user})
  WHERE NOT t:User AND (r:ABOUT OR (r.until IS NULL AND """ + visible("r") + """))
  AND """ + entity_visible("t") + """
  AND NOT EXISTS { MATCH (t)<-[:ABOUT]-(b:Memory {kind: 'boundary'}) WHERE b.invalidated_at IS NULL }
WITH gds.graph.project($graph, s, t,
       {sourceNodeLabels: labels(s), targetNodeLabels: labels(t)},
       {undirectedRelationshipTypes: ['*']}) AS g
RETURN g.nodeCount AS nodes, g.relationshipCount AS relationships
"""


@contextmanager
def life_graph(store: MemoryStore):
    """Project the user's life graph into GDS memory, and always drop it afterwards."""
    name = f"life_{uuid.uuid4().hex[:8]}"
    size = store._q(PROJECT, graph=name)[0]
    try:
        yield name, size
    finally:
        store._q("CALL gds.graph.drop($graph, false) YIELD graphName RETURN graphName", graph=name)


def related(store: MemoryStore, name: str | None = None, k: int = 3, method: str = "fastrp") -> list[dict]:
    """Entities most related to `name` (or the top pairs overall, without a name).

    method="fastrp": FastRP embeddings, then kNN over the entities.
    method="jaccard": node similarity (shared direct neighbours)."""
    with life_graph(store) as (graph, size):
        if size["relationships"] == 0:
            return []
        if method == "fastrp":
            store._q(
                """
                CALL gds.fastRP.mutate($graph, {
                  embeddingDimension: $dim, mutateProperty: 'emb', randomSeed: $seed, concurrency: 1,
                  iterationWeights: [0.0, 1.0, 1.0]   // ignore the node's own random start; 1 and 2 hops count equally
                }) YIELD nodePropertiesWritten RETURN nodePropertiesWritten
                """, graph=graph, dim=EMBEDDING_DIM, seed=SEED)
            rows = store._q(
                """
                CALL gds.knn.stream($graph, {
                  nodeLabels: ['Entity'], nodeProperties: ['emb'], topK: $k,
                  randomSeed: $seed, concurrency: 1, sampleRate: 1.0, deltaThreshold: 0.0
                }) YIELD node1, node2, similarity
                RETURN gds.util.asNode(node1).name AS name, gds.util.asNode(node2).name AS other,
                       gds.util.asNode(node2).kind AS kind, round(similarity, 3) AS score
                """, graph=graph, k=k, seed=SEED)
        elif method == "jaccard":
            rows = store._q(
                """
                CALL gds.nodeSimilarity.stream($graph, {topK: $k, similarityCutoff: 0.0, concurrency: 1})
                YIELD node1, node2, similarity
                WITH gds.util.asNode(node1) AS a, gds.util.asNode(node2) AS b, similarity
                WHERE a:Entity AND b:Entity
                RETURN a.name AS name, b.name AS other, b.kind AS kind, round(similarity, 3) AS score
                """, graph=graph, k=k + 10)  # memories take some of each node's top-k slots
        else:
            raise ValueError(f"unknown method {method!r}")
    rows = [r for r in rows if r["score"] > MIN_SCORE[method]]
    if name:
        rows = [r for r in rows if r["name"].lower() == name.lower()]
    rows.sort(key=lambda r: -r["score"])
    return rows[:k] if name else _unique_pairs(rows)


def _unique_pairs(rows: list[dict]) -> list[dict]:
    """kNN returns a->b and b->a: keep each pair once, strongest first."""
    seen, out = set(), []
    for r in rows:
        key = frozenset((r["name"], r["other"]))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
