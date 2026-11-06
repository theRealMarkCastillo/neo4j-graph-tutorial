"""Lesson 19 — give every stored memory, entity and episode a vector from the active model.

    python -m companion.reembed              nodes that don't have one yet (after a switch)
    python -m companion.reembed --all        recompute every vector (e.g. a changed instruction)
    python -m companion.reembed --user Mark  one user only

Run it after changing EMBED_MODEL / EMBED_BASE_URL. Each model writes its own property
(embedding / embedding_<model>), so the old vectors stay and switching back costs nothing.
The text embedded is the same text the write path embeds (keep them in step: a vector made
from different text matches differently). Tombstones (:Forgotten) keep no text, so they
can't be re-embedded; they only matter while their episode is open, so that's harmless.
"""
from __future__ import annotations

import sys
import time

from neo4j import GraphDatabase

from config import NEO4J_AUTH, NEO4J_URI
from companion.embeddings import PROFILE, embed, remote
from companion.schema import ensure_schema

# label -> Cypher for the text the write path embeds for it (store.py)
TEXTS = {
    "User": "n.name + ', the user'",
    "Entity": "n.name + ' (' + n.kind + '): ' + coalesce(n.summary, '')",
    "Memory": "n.text",
    "Episode": "n.summary",
}
BATCH = 64


def reembed(driver, user: str | None = None, everything: bool = False) -> dict[str, int]:
    """`user`: a user's name ("Mark"), as MemoryStore takes it; None for everyone."""
    user = f"user:{user.lower()}" if user else None  # the id MemoryStore gives that user
    ensure_schema(driver)  # this model's vector indexes
    counts = {}
    for label, text in TEXTS.items():
        where = ["n[$prop] IS NULL"] if not everything else []
        where += [f"{text} IS NOT NULL"]
        if label == "Entity":
            where.append("NOT n:User")
        if user:
            where.append("n.user = $user")
        rows, _, _ = driver.execute_query(
            f"MATCH (n:{label}) WHERE {' AND '.join(where)} RETURN elementId(n) AS id, {text} AS text",
            prop=PROFILE.prop, user=user)
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            vecs = embed([r["text"] for r in chunk])
            driver.execute_query(
                """
                UNWIND $rows AS row MATCH (n) WHERE elementId(n) = row.id
                CALL db.create.setNodeVectorProperty(n, $prop, row.vec)
                """,
                rows=[{"id": r["id"], "vec": v} for r, v in zip(chunk, vecs)], prop=PROFILE.prop)
        counts[label] = len(rows)
    return counts


def main() -> None:
    args = sys.argv[1:]
    user = args[args.index("--user") + 1] if "--user" in args else None
    where = f"{PROFILE.model} via {'the endpoint in EMBED_BASE_URL' if remote() else 'fastembed (local)'}"
    print(f"Embedding with {where} -> property {PROFILE.prop}")
    start = time.time()
    with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        counts = reembed(driver, user, everything="--all" in args)
    print(", ".join(f"{n} {label.lower()}" for label, n in counts.items()), f"in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
