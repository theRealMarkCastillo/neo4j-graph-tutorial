"""Lessons 9-10 — MemoryStore: writes memories into the graph and recalls them for a turn.

Write path (Lesson 9):  resolve entities -> link relations -> dedupe -> supersede -> store
Read path  (Lesson 10): core profile + semantic (vector) + entity hop (graph) + follow-ups
                        + recent episodes, ranked by similarity, importance and recency
"""
from __future__ import annotations

import datetime as dt
import uuid

from fastembed import TextEmbedding

from config import EMBED_MODEL
from companion.models import Extraction
from companion.schema import ensure_schema

_embedder: TextEmbedding | None = None


def embed(texts: list[str]) -> list[list[float]]:
    global _embedder
    _embedder = _embedder or TextEmbedding(EMBED_MODEL)
    return [v.tolist() for v in _embedder.embed(texts)]


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


def _to_date(value: str | None) -> dt.date | None:
    try:
        return dt.date.fromisoformat(value) if value else None
    except ValueError:  # "2019" or "next spring" stay in the memory text instead
        return None


# Duplicate threshold on Neo4j's 0-1 rescaled cosine (0.975 ≈ raw cosine 0.95).
DUPLICATE_SCORE = 0.975
# Entity match threshold when names differ ("Mom" vs "my mother").
ENTITY_MATCH_SCORE = 0.93


class MemoryStore:
    def __init__(self, driver, user_name: str):
        self.driver = driver
        self.user_name = user_name
        self.user = f"user:{user_name.lower()}"
        ensure_schema(driver)
        self._q(
            """
            MERGE (u:Entity:User {id: $user})
            ON CREATE SET u.created_at = datetime()
            SET u.name = $name, u.user = $user, u.kind = 'self'
            WITH u CALL db.create.setNodeVectorProperty(u, 'embedding', $vec)
            """,
            name=user_name, vec=embed([f"{user_name}, the user"])[0],
        )

    def _q(self, query: str, **params) -> list[dict]:
        records, _, _ = self.driver.execute_query(query, user=self.user, **params)
        return [r.data() for r in records]

    # ------------------------------------------------------------------ episodes
    def start_episode(self, on: dt.date | None = None) -> str:
        """Open a conversation episode. `on` backdates it (used by the scripted demo)."""
        episode_id = _new_id("ep")
        self._q(
            """
            OPTIONAL MATCH (prev:Episode {user: $user}) WHERE prev.ended_at IS NOT NULL
            WITH prev ORDER BY prev.started_at DESC LIMIT 1
            CREATE (ep:Episode {id: $id, user: $user,
                                started_at: CASE WHEN $on IS NULL THEN datetime()
                                                 ELSE datetime({date: $on, hour: 12}) END})
            FOREACH (_ IN CASE WHEN prev IS NULL THEN [] ELSE [1] END | MERGE (prev)-[:NEXT]->(ep))
            """,
            id=episode_id, on=on,
        )
        return episode_id

    def log_turn(self, episode_id: str, role: str, text: str) -> None:
        self._q(
            """
            MATCH (ep:Episode {id: $ep})
            CREATE (ep)-[:HAS_TURN]->(:Turn {idx: COUNT { (ep)-[:HAS_TURN]->() },
                                             role: $role, text: $text, at: datetime()})
            """,
            ep=episode_id, role=role, text=text,
        )

    def episode_transcript(self, episode_id: str) -> str:
        rows = self._q(
            "MATCH (:Episode {id: $ep})-[:HAS_TURN]->(t) RETURN t.role AS role, t.text AS text ORDER BY t.idx",
            ep=episode_id,
        )
        return "\n".join(f"{r['role']}: {r['text']}" for r in rows)

    def end_episode(self, episode_id: str, summary: str | None, mood: str | None) -> None:
        if not summary:  # nothing was said: drop the empty episode
            self._q("MATCH (ep:Episode {id: $ep}) WHERE NOT (ep)-[:HAS_TURN]->() DETACH DELETE ep", ep=episode_id)
            return
        self._q(
            """
            MATCH (ep:Episode {id: $ep})
            SET ep.ended_at = datetime(), ep.summary = $summary, ep.mood = $mood
            WITH ep CALL db.create.setNodeVectorProperty(ep, 'embedding', $vec)
            """,
            ep=episode_id, summary=summary, mood=mood, vec=embed([summary])[0],
        )

    # ------------------------------------------------------------------ entities
    def resolve_entity(self, name: str, kind: str = "thing", aliases=(), summary: str = "") -> str:
        """Find-or-create an entity: exact name/alias, then embedding similarity, then new."""
        if name.strip().lower() in {"user", "me", "i", self.user_name.lower()}:
            return self.user
        names = [n.lower() for n in [name, *aliases]]
        hit = self._q(
            """
            MATCH (e:Entity {user: $user})
            WHERE toLower(e.name) IN $names OR any(a IN coalesce(e.aliases, []) WHERE toLower(a) IN $names)
            RETURN e.id AS id LIMIT 1
            """,
            names=names,
        )
        vec = embed([f"{name} ({kind}): {summary}"])[0]
        if not hit:
            hit = self._q(
                """
                CALL db.index.vector.queryNodes('entity_vec', 5, $vec) YIELD node, score
                WHERE node.user = $user AND node.kind = $kind AND score >= $min
                RETURN node.id AS id ORDER BY score DESC LIMIT 1
                """,
                vec=vec, kind=kind, min=ENTITY_MATCH_SCORE,
            )
        entity_id = hit[0]["id"] if hit else _new_id("e")
        self._q(
            """
            MERGE (e:Entity {id: $id})
            ON CREATE SET e.user = $user, e.name = $name, e.kind = $kind, e.created_at = datetime()
            SET e.aliases = [a IN apoc.coll.toSet(coalesce(e.aliases, []) + $aliases) WHERE toLower(a) <> toLower(e.name)],
                e.summary = CASE WHEN $summary <> '' THEN $summary ELSE e.summary END
            WITH e CALL db.create.setNodeVectorProperty(e, 'embedding', $vec)
            """,
            id=entity_id, name=name, kind=kind, aliases=list(aliases), summary=summary, vec=vec,
        )
        return entity_id

    def _lookup(self, name: str, known: dict[str, str]) -> str | None:
        """Map a name used in a memory/relation to an entity id without creating junk nodes."""
        key = name.strip().lower()
        if key in {"user", "me", "i", self.user_name.lower()}:
            return self.user
        if key in known:
            return known[key]
        hit = self._q(
            """
            MATCH (e:Entity {user: $user})
            WHERE toLower(e.name) = $n OR any(a IN coalesce(e.aliases, []) WHERE toLower(a) = $n)
            RETURN e.id AS id LIMIT 1
            """,
            n=key,
        )
        return hit[0]["id"] if hit else None

    # ------------------------------------------------------------------ write path
    def apply(self, ex: Extraction, episode_id: str | None = None) -> list[str]:
        """Store one extraction. Returns a human-readable log of what changed."""
        log: list[str] = []
        known: dict[str, str] = {}
        for ent in ex.entities:
            eid = self.resolve_entity(ent.name, ent.kind, ent.aliases, ent.summary)
            for n in [ent.name, *ent.aliases]:
                known[n.lower()] = eid

        for rel in ex.relations:
            a, b = self._lookup(rel.source, known), self._lookup(rel.target, known)
            if a and b and a != b:
                self._q(
                    """
                    MATCH (a:Entity {id: $a}), (b:Entity {id: $b})
                    MERGE (a)-[r:RELATED {type: $type}]->(b)
                    SET r.since = coalesce($since, r.since), r.updated_at = datetime()
                    """,
                    a=a, b=b, type=rel.type, since=rel.since,
                )
                log.append(f"linked {rel.source} -{rel.type}-> {rel.target}")

        for mid in ex.invalidate:
            if self._q(
                """
                MATCH (m:Memory {id: $id, user: $user}) WHERE m.invalidated_at IS NULL
                SET m.invalidated_at = datetime() RETURN m.text AS text
                """, id=mid,
            ):
                log.append(f"retired {mid}")
        for mid in ex.completed_plans:
            if self._q("MATCH (m:Memory {id: $id, user: $user, kind: 'plan'}) SET m.status = 'done' RETURN m.id", id=mid):
                log.append(f"plan done {mid}")

        vectors = embed([m.text for m in ex.memories]) if ex.memories else []
        for mem, vec in zip(ex.memories, vectors):
            dup = self._q(
                """
                CALL db.index.vector.queryNodes('memory_vec', 5, $vec) YIELD node, score
                WHERE node.user = $user AND node.invalidated_at IS NULL AND score >= $min
                  AND NOT node.id IN $supersedes
                RETURN node.id AS id LIMIT 1
                """,
                vec=vec, min=DUPLICATE_SCORE, supersedes=mem.supersedes,
            )
            if dup:  # already known: reinforce instead of storing a near-copy
                self._q(
                    """
                    MATCH (m:Memory {id: $id})
                    SET m.access_count = coalesce(m.access_count, 0) + 1, m.last_confirmed = datetime(),
                        m.importance = CASE WHEN $imp > m.importance THEN $imp ELSE m.importance END
                    """,
                    id=dup[0]["id"], imp=mem.importance,
                )
                log.append(f"reinforced {dup[0]['id']}")
                continue

            mid = _new_id("m")
            about = [eid for n in (mem.about or ["user"]) if (eid := self._lookup(n, known))]
            retired = self._q(
                """
                CREATE (m:Memory {id: $id, user: $user, text: $text, kind: $kind, importance: $imp,
                                  occurred_on: $occurred, due_on: $due,
                                  status: CASE WHEN $kind = 'plan' THEN 'open' ELSE null END,
                                  created_at: datetime(), access_count: 0})
                WITH m CALL db.create.setNodeVectorProperty(m, 'embedding', $vec)
                WITH m
                CALL (m) { UNWIND $about AS eid MATCH (e:Entity {id: eid}) MERGE (m)-[:ABOUT]->(e) }
                CALL (m) { MATCH (ep:Episode {id: $ep}) MERGE (m)-[:FROM]->(ep) }
                CALL (m) {
                  UNWIND $supersedes AS old_id
                  MATCH (old:Memory {id: old_id, user: $user}) WHERE old.invalidated_at IS NULL
                  SET old.invalidated_at = datetime()
                  MERGE (m)-[:SUPERSEDES]->(old)
                  RETURN collect(old.id) AS retired
                }
                RETURN retired
                """,
                id=mid, text=mem.text, kind=mem.kind, imp=mem.importance,
                occurred=_to_date(mem.occurred_on), due=_to_date(mem.due_on),
                vec=vec, about=about, ep=episode_id or "", supersedes=mem.supersedes,
            )[0]["retired"]
            log.append(f"+ [{mem.kind}] {mem.text}" + (f"  (replaces {', '.join(retired)})" if retired else ""))
        return log

    # ------------------------------------------------------------------ read path
    def extraction_candidates(self, text: str, k: int = 10) -> list[dict]:
        """Existing current memories the extractor may update or retire, with their ids."""
        return self._q(
            """
            CALL db.index.vector.queryNodes('memory_vec', $k, $vec) YIELD node AS m, score
            WHERE m.user = $user AND m.invalidated_at IS NULL
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.status AS status
            UNION
            MATCH (m:Memory {user: $user, kind: 'plan', status: 'open'}) WHERE m.invalidated_at IS NULL
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.status AS status
            """,
            k=k, vec=embed([text])[0],
        )

    def known_entities(self) -> list[str]:
        rows = self._q(
            "MATCH (e:Entity {user: $user}) WHERE e.kind <> 'self' RETURN e.name AS name, e.kind AS kind ORDER BY e.name"
        )
        return [f"{r['name']} ({r['kind']})" for r in rows]

    def recall(self, query: str, k: int = 8, today: dt.date | None = None) -> dict:
        today = today or dt.date.today()
        vec = embed([query])[0]
        # Core profile: every boundary + the most important stable facts and preferences.
        core = self._q(
            """
            CALL () {
              MATCH (m:Memory {user: $user, kind: 'boundary'}) WHERE m.invalidated_at IS NULL
              RETURN m
              UNION
              MATCH (m:Memory {user: $user}) WHERE m.invalidated_at IS NULL
                AND m.kind IN ['fact', 'preference'] AND m.importance >= 6
              WITH m ORDER BY m.importance DESC, m.created_at DESC LIMIT 8
              RETURN m
            }
            RETURN m.id AS id, m.kind AS kind, m.text AS text
            ORDER BY CASE m.kind WHEN 'boundary' THEN 0 ELSE 1 END, m.importance DESC
            """
        )
        # Semantic: vector similarity re-ranked by importance and recency (feelings fade fast).
        semantic = self._q(
            """
            CALL db.index.vector.queryNodes('memory_vec', 50, $vec) YIELD node AS m, score
            WHERE m.user = $user AND m.invalidated_at IS NULL AND coalesce(m.status, '') <> 'done'
            WITH m, score,
                 duration.inDays(coalesce(m.last_accessed, m.created_at), datetime()).days AS age
            WITH m, score, 0.6 * score + 0.25 * m.importance / 10.0
                 + 0.15 * exp(-age / CASE m.kind WHEN 'feeling' THEN 7.0 ELSE 90.0 END) AS rank
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.occurred_on AS occurred_on,
                   round(score, 3) AS score, round(rank, 3) AS rank
            ORDER BY rank DESC LIMIT $k
            """,
            vec=vec, k=k,
        )
        # Graph hop: entities named (or meant) in the message -> their memories and relations.
        entities = self._q(
            """
            CALL () {
              MATCH (e:Entity {user: $user}) WHERE e.kind <> 'self'
                AND any(n IN [e.name] + coalesce(e.aliases, []) WHERE size(n) > 2 AND toLower($q) CONTAINS toLower(n))
              RETURN e
              UNION
              CALL db.index.vector.queryNodes('entity_vec', 3, $vec) YIELD node AS e, score
              WITH e, score WHERE e.user = $user AND e.kind <> 'self' AND score >= 0.85
              RETURN e
            }
            WITH e WHERE NOT EXISTS {  // entities protected by a boundary are never surfaced
              MATCH (:Memory {kind: 'boundary'})-[:ABOUT]->(e) }
            OPTIONAL MATCH (:User {id: $user})-[r:RELATED]-(e)
            OPTIONAL MATCH (m:Memory)-[:ABOUT]->(e)
              WHERE m.invalidated_at IS NULL AND coalesce(m.status, '') <> 'done'
            WITH e, r, m ORDER BY m.importance DESC
            WITH e, head(collect(r.type)) AS relation, collect(m {.id, .kind, .text})[..5] AS memories
            RETURN e.name AS name, e.kind AS kind, e.summary AS summary, relation, memories
            """,
            q=query, vec=vec,
        )
        people = self._q(
            """
            MATCH (:User {id: $user})-[r:RELATED]->(e:Entity)
            WHERE NOT EXISTS { MATCH (b:Memory {kind: 'boundary'})-[:ABOUT]->(e) WHERE b.invalidated_at IS NULL }
            RETURN e.name AS name, e.kind AS kind, r.type AS relation ORDER BY e.kind, e.name LIMIT 25
            """
        )
        followups = self._q(
            """
            MATCH (m:Memory {user: $user, kind: 'plan', status: 'open'})
            WHERE m.invalidated_at IS NULL AND m.due_on IS NOT NULL
              AND m.due_on >= $today - duration({days: 14}) AND m.due_on <= $today + duration({days: 14})
            RETURN m.id AS id, m.text AS text, m.due_on AS due_on ORDER BY m.due_on
            """,
            today=today,
        )
        episodes = self._q(
            """
            MATCH (ep:Episode {user: $user}) WHERE ep.summary IS NOT NULL
            RETURN ep.started_at AS at, ep.summary AS summary, ep.mood AS mood
            ORDER BY ep.started_at DESC LIMIT 3
            """
        )
        touched = [m["id"] for m in semantic] + [m["id"] for e in entities for m in e["memories"]]
        self._q(
            """
            MATCH (m:Memory) WHERE m.id IN $ids
            SET m.last_accessed = datetime(), m.access_count = coalesce(m.access_count, 0) + 1
            """,
            ids=touched,
        )
        return {"core": core, "semantic": semantic, "entities": entities, "people": people,
                "followups": followups, "episodes": episodes}

    # ------------------------------------------------------------------ inspection
    def memories(self, kind: str | None = None, include_retired: bool = False) -> list[dict]:
        return self._q(
            """
            MATCH (m:Memory {user: $user})
            WHERE ($kind IS NULL OR m.kind = $kind) AND ($all OR m.invalidated_at IS NULL)
            OPTIONAL MATCH (m)-[:ABOUT]->(e)
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.importance AS importance,
                   m.status AS status, m.invalidated_at IS NOT NULL AS retired,
                   m.created_at AS created_at, collect(e.name) AS about
            ORDER BY created_at
            """,
            kind=kind, all=include_retired,
        )

    def timeline(self) -> list[dict]:
        return self._q(
            """
            MATCH (m:Memory {user: $user})
            WHERE m.kind IN ['event', 'plan'] AND coalesce(m.occurred_on, m.due_on) IS NOT NULL
            RETURN coalesce(m.occurred_on, m.due_on) AS date, m.kind AS kind, m.text AS text,
                   m.status AS status, m.invalidated_at IS NOT NULL AS retired
            ORDER BY date
            """
        )

    def history(self, memory_id: str) -> list[dict]:
        """A memory and every version it replaced, newest first."""
        return self._q(
            """
            MATCH path = (m:Memory {id: $id, user: $user})-[:SUPERSEDES*0..]->(old)
            RETURN old.id AS id, old.text AS text, old.created_at AS created_at,
                   old.invalidated_at AS retired_at ORDER BY length(path)
            """,
            id=memory_id,
        )

    def forget(self, memory_id: str) -> bool:
        """User-requested forgetting: hard delete (privacy beats provenance here)."""
        return bool(self._q(
            "MATCH (m:Memory {id: $id, user: $user}) WITH m, m.id AS id DETACH DELETE m RETURN id",
            id=memory_id,
        ))

    def wipe(self) -> None:
        """Delete everything stored for this user (used by the demo to start clean)."""
        self._q(
            """
            MATCH (n) WHERE (n:Memory OR n:Entity OR n:Episode) AND n.user = $user
            OPTIONAL MATCH (n)-[:HAS_TURN]->(t:Turn)
            DETACH DELETE n, t
            """
        )
        self.__init__(self.driver, self.user_name)


def format_context(r: dict, today: dt.date | None = None) -> str:
    """Render a recall() result as the memory block placed in the system prompt."""
    today = today or dt.date.today()
    out: list[str] = []
    if r["core"]:
        out.append("Core profile:\n" + "\n".join(f"- {m['text']}" for m in r["core"]))
    if r["people"]:
        out.append("People and things in their life:\n" + "\n".join(
            f"- {p['name']} ({p['relation']})" for p in r["people"]))
    if r["followups"]:
        out.append("Open plans to follow up on (today is " + today.isoformat() + "):\n" + "\n".join(
            f"- {m['text']} (due {m['due_on']})" for m in r["followups"]))
    seen = {m["text"] for m in r["core"] + r["followups"]}
    relevant = [m for m in r["semantic"] if m["text"] not in seen]
    for e in r["entities"]:
        relevant += [m for m in e["memories"] if m["text"] not in seen | {x["text"] for x in relevant}]
    if relevant:
        out.append("Relevant memories:\n" + "\n".join(f"- [{m['kind']}] {m['text']}" for m in relevant))
    if r["episodes"]:
        out.append("Recent conversations:\n" + "\n".join(
            f"- {str(e['at'])[:10]} (mood: {e['mood']}): {e['summary']}" for e in r["episodes"]))
    return "\n\n".join(out) or "(No memories yet — this is a new relationship.)"
