"""Lesson 8 — the memory graph's schema. Safe to run repeatedly (IF NOT EXISTS everywhere).

    (:User:Entity {id, name, kind:'self'})
    (:Entity {id, user, name, kind, aliases, summary, embedding})   embedding: see Lesson 19
        kind: person | pet | place | organization | activity | thing
    (:User|Entity)-[:RELATED {type, said, since, until, updated_at}]->(:Entity)   life graph
        type: canonical ontology type (Lesson 12); said: the word used; until: when it ended
    (:Forgotten {user, episode, embedding})                     tombstone for /forget (Lesson 11)
    (:Memory {id, user, text, kind, importance, embedding,
              occurred_on, due_on, status,                     domain time (when it happened)
              created_at, invalidated_at, last_accessed, access_count})  system time
        kind: fact | preference | event | plan | feeling | boundary
    (:Memory)-[:ABOUT]->(:Entity)                               who/what it concerns
    (:Memory)-[:SUPERSEDES]->(:Memory)                          newer version of a fact
    (:Memory)-[:FROM]->(:Episode)                               provenance
    (:Episode {id, user, started_at, ended_at, summary, mood, embedding})
    (:Episode)-[:NEXT]->(:Episode), (:Episode)-[:HAS_TURN]->(:Turn {idx, role, text, at})
"""
from companion.embeddings import PROFILE

STATEMENTS = [
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
    "CREATE CONSTRAINT episode_id IF NOT EXISTS FOR (e:Episode) REQUIRE e.id IS UNIQUE",
    "CREATE INDEX memory_user_kind IF NOT EXISTS FOR (m:Memory) ON (m.user, m.kind)",
    "CREATE INDEX entity_user IF NOT EXISTS FOR (e:Entity) ON (e.user)",
    "CREATE INDEX episode_user IF NOT EXISTS FOR (e:Episode) ON (e.user, e.started_at)",
]


def vector_indexes(profile=PROFILE) -> list[str]:
    """One set of vector indexes per embedding model (Lesson 19): memory_vec on n.embedding for
    the default model, memory_vec_<model> on n.embedding_<model> for any other."""
    return [
        f"CREATE VECTOR INDEX {profile.index(name)} IF NOT EXISTS FOR (n:{label}) ON (n.{profile.prop}) "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {profile.dims}, "
        f"`vector.similarity_function`: 'cosine'}}}}"
        for name, label in [("memory_vec", "Memory"), ("entity_vec", "Entity"), ("episode_vec", "Episode")]
    ]


def ensure_schema(driver) -> None:
    for stmt in STATEMENTS + vector_indexes():
        driver.execute_query(stmt)
    driver.execute_query("CALL db.awaitIndexes(120)")
