"""Lessons 9-10 — MemoryStore: writes memories into the graph and recalls them for a turn.

Write path (Lesson 9):  resolve entities -> link relations -> dedupe -> supersede -> store
Read path  (Lesson 10): core profile + semantic (vector) + entity hop (graph) + follow-ups
                        + recent episodes, ranked by similarity, importance and recency
"""
from __future__ import annotations

import datetime as dt
import re
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
# Sensitive memories (health, finances...) are recalled only when the message is clearly about
# them: (near) the best vector match for this message, not just above a fixed score. Small
# embedding models squeeze all scores into a narrow band (~0.63-0.80 here), so relative wins.
SENSITIVE_MARGIN = 0.01
SENSITIVE_FLOOR = 0.72

# A memory about someone a boundary protects ("don't bring up Dana") is never recalled.
NOT_PROTECTED = """NOT EXISTS {
    MATCH (m)-[:ABOUT]->(:Entity)<-[:ABOUT]-(b:Memory {kind: 'boundary'})
    WHERE b.invalidated_at IS NULL AND b <> m }"""

# "Is anyone in my family...?" -> relation types that belong to each group.
RELATION_GROUPS = {
    "family": {"mother", "father", "mom", "dad", "parent", "sister", "brother", "sibling", "son",
               "daughter", "child", "aunt", "uncle", "cousin", "grandmother", "grandfather",
               "grandparent", "niece", "nephew", "wife", "husband", "spouse", "partner"},
    "friends": {"friend", "best_friend", "close_friend", "roommate"},
    "work": {"coworker", "colleague", "boss", "manager", "report", "mentor", "works_at"},
    "pets": {"pet", "owns_pet"},
}
GROUP_WORDS = {
    "family": r"family|relatives?|siblings?|parents",
    "friends": r"friends?",
    "work": r"co-?workers?|colleagues?|work|team|job",
    "pets": r"pets?|animals?",
}


def _system_time(today: dt.date) -> dt.datetime:
    """System-time stamp for writes: now, or noon of a simulated `today` (demo/scenarios)."""
    real = dt.datetime.now(dt.timezone.utc)
    return real if today == real.date() else dt.datetime(today.year, today.month, today.day, 12, tzinfo=dt.timezone.utc)


def _annual(md: str | None) -> str | None:
    return md if md and re.fullmatch(r"\d{2}-\d{2}", md) else None


def _next_occurrence(md: str, today: dt.date) -> dt.date | None:
    """Next date (from 3 days ago onward) on which a yearly 'MM-DD' falls."""
    month, day = map(int, md.split("-"))
    for year in (today.year - 1, today.year, today.year + 1):
        try:
            d = dt.date(year, month, day)
        except ValueError:  # Feb 29 in a non-leap year
            d = dt.date(year, month, 28)
        if d >= today - dt.timedelta(days=3):
            return d
    return None


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

    def episode_turns(self, episode_id: str) -> list[dict]:
        return self._q(
            "MATCH (:Episode {id: $ep})-[:HAS_TURN]->(t) RETURN t.role AS role, t.text AS text ORDER BY t.idx",
            ep=episode_id,
        )

    def episode_transcript(self, episode_id: str) -> str:
        return "\n".join(f"{r['role']}: {r['text']}" for r in self.episode_turns(episode_id))

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
    def _is_user(self, name: str) -> bool:
        return name.strip().lower() in {"user", "me", "i", self.user, self.user_name.lower()}

    def resolve_entity(self, name: str, kind: str = "thing", aliases=(), summary: str = "",
                       ref: str | None = None, new: bool = False) -> str:
        """Find-or-create an entity.

        1. `ref` — the extractor pointed at a known entity id
        2. `new` — the extractor says it's a different entity with a known name (a second Sam)
        3. exact name/alias; if several share the name, the one whose summary is most similar
        4. embedding similarity (same kind) above ENTITY_MATCH_SCORE
        5. otherwise create it
        """
        if self._is_user(name):
            return self.user
        vec = embed([f"{name} ({kind}): {summary}"])[0]
        hit = self._q("MATCH (e:Entity {id: $id, user: $user}) RETURN e.id AS id", id=ref) if ref else []
        if not hit and not new:
            names = [n.lower() for n in [name, *aliases]]
            hit = self._q(
                """
                MATCH (e:Entity {user: $user}) WHERE e.kind <> 'self'
                  AND (toLower(e.name) IN $names OR any(a IN coalesce(e.aliases, []) WHERE toLower(a) IN $names))
                RETURN e.id AS id ORDER BY vector.similarity.cosine(e.embedding, $vec) DESC LIMIT 1
                """,
                names=names, vec=vec,
            )
        if not hit and not new:
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
            SET e.updated_at = datetime(), e.aliases = [a IN apoc.coll.toSet(coalesce(e.aliases, []) + $aliases) WHERE toLower(a) <> toLower(e.name)],
                e.summary = CASE WHEN $summary <> '' THEN $summary ELSE e.summary END
            WITH e CALL db.create.setNodeVectorProperty(e, 'embedding', $vec)
            """,
            id=entity_id, name=name, kind=kind, aliases=list(aliases), summary=summary, vec=vec,
        )
        return entity_id

    def _lookup(self, name: str, known: dict[str, str]) -> str | None:
        """Map a name or id used in a memory/relation to an entity id without creating junk nodes.

        Names resolved in this same extraction win; otherwise an id, then the most recently
        updated entity with that name or alias."""
        key = name.strip().lower()
        if self._is_user(key):
            return self.user
        if key in known:
            return known[key]
        hit = self._q(
            """
            MATCH (e:Entity {user: $user})
            WHERE e.id = $n OR toLower(e.name) = $n OR any(a IN coalesce(e.aliases, []) WHERE toLower(a) = $n)
            RETURN e.id AS id ORDER BY CASE WHEN e.id = $n THEN 0 ELSE 1 END, e.updated_at DESC LIMIT 1
            """,
            n=key,
        )
        return hit[0]["id"] if hit else None

    # ------------------------------------------------------------------ write path
    def apply(self, ex: Extraction, episode_id: str | None = None, today: dt.date | None = None) -> list[str]:
        """Store one extraction. Returns a human-readable log of what changed."""
        today = today or dt.date.today()
        now = _system_time(today)
        log: list[str] = []
        known: dict[str, str] = {}
        for ent in ex.entities:
            eid = self.resolve_entity(ent.name, ent.kind, ent.aliases, ent.summary, ent.ref, ent.new)
            for n in [ent.name, *ent.aliases, *([ent.ref] if ent.ref else [])]:
                known[n.lower()] = eid
            if ent.new:
                log.append(f"new entity {ent.name} ({eid}), distinct from the known {ent.name}")

        linking = {(self._lookup(r.source, known), self._lookup(r.target, known), r.type)
                   for r in ex.relations if not r.ended}
        for rel in ex.relations:
            a, b = self._lookup(rel.source, known), self._lookup(rel.target, known)
            if not (a and b and a != b) or (rel.ended and (a, b, rel.type) in linking):
                continue  # unresolved, or "ended X" + "linked X" in one go: keep it linked
            names = self._q("MATCH (a:Entity {id: $a}), (b:Entity {id: $b}) RETURN a.name AS a, b.name AS b", a=a, b=b)[0]
            src = "user" if a == self.user else names["a"]
            if rel.ended:  # breakup, moved away, left a job: keep the edge as history
                ended = self._q(
                    """
                    MATCH (a:Entity {id: $a})-[r:RELATED]->(b:Entity {id: $b}) WHERE r.until IS NULL
                    WITH r, r.type = $type AS exact
                    WITH collect(r) AS active, [x IN collect(CASE WHEN exact THEN r END) WHERE x IS NOT NULL] AS same
                    UNWIND CASE WHEN size(same) > 0 THEN same ELSE active END AS r
                    SET r.until = $today
                    RETURN r.type AS type
                    """,
                    a=a, b=b, type=rel.type, today=today,
                )
                if ended:
                    log.append(f"ended {src} -{'/'.join(e['type'] for e in ended)}-> {names['b']}")
                continue
            changed = self._q(
                """
                MATCH (a:Entity {id: $a}), (b:Entity {id: $b})
                MERGE (a)-[r:RELATED {type: $type}]->(b)
                WITH r, r.updated_at IS NULL OR r.until IS NOT NULL AS changed
                SET r.since = coalesce($since, r.since), r.updated_at = $now, r.until = null
                RETURN changed
                """,
                a=a, b=b, type=rel.type, since=rel.since, now=now,
            )[0]["changed"]
            if changed:  # only log new or re-activated links
                log.append(f"linked {src} -{rel.type}-> {names['b']}")

        for mid in ex.invalidate:
            if self._q(
                """
                MATCH (m:Memory {id: $id, user: $user}) WHERE m.invalidated_at IS NULL
                SET m.invalidated_at = $now RETURN m.text AS text
                """, id=mid, now=now,
            ):
                log.append(f"retired {mid}")
        for mid in ex.completed_plans:
            if self._q("MATCH (m:Memory {id: $id, user: $user, kind: 'plan', status: 'open'}) SET m.status = 'done' RETURN m.id", id=mid):
                log.append(f"plan done {mid}")

        vectors = embed([m.text for m in ex.memories]) if ex.memories else []
        for mem, vec in zip(ex.memories, vectors):
            dup = self._q(
                """
                CALL db.index.vector.queryNodes('memory_vec', 5, $vec) YIELD node, score
                WHERE node.user = $user AND node.invalidated_at IS NULL AND score >= $min
                  // same dates too: "dentist on 2026-11-12" -> "...on 2026-11-17" reads almost the
                  // same but is a real correction; an identical "replacement" is just a duplicate
                  AND coalesce(node.due_on, date('1900-01-01')) = coalesce($due, date('1900-01-01'))
                  AND coalesce(node.occurred_on, date('1900-01-01')) = coalesce($occurred, date('1900-01-01'))
                RETURN node.id AS id LIMIT 1
                """,
                vec=vec, min=DUPLICATE_SCORE, due=_to_date(mem.due_on), occurred=_to_date(mem.occurred_on),
            )
            if dup:  # already known: reinforce instead of storing a near-copy
                self._q(
                    """
                    MATCH (m:Memory {id: $id})
                    SET m.access_count = coalesce(m.access_count, 0) + 1, m.last_confirmed = $now,
                        m.importance = CASE WHEN $imp > m.importance THEN $imp ELSE m.importance END
                    """,
                    id=dup[0]["id"], imp=mem.importance, now=now,
                )
                log.append(f"reinforced {dup[0]['id']}")
                continue

            mid = _new_id("m")
            about = [eid for n in (mem.about or ["user"]) if (eid := self._lookup(n, known))]
            retired = self._q(
                """
                CREATE (m:Memory {id: $id, user: $user, text: $text, kind: $kind, importance: $imp,
                                  occurred_on: $occurred, due_on: $due, annual_date: $annual,
                                  sensitive: $sensitive,
                                  status: CASE WHEN $kind = 'plan' THEN 'open' ELSE null END,
                                  created_at: $now, access_count: 0})
                WITH m CALL db.create.setNodeVectorProperty(m, 'embedding', $vec)
                WITH m
                CALL (m) { UNWIND $about AS eid MATCH (e:Entity {id: eid}) MERGE (m)-[:ABOUT]->(e) }
                CALL (m) { MATCH (ep:Episode {id: $ep}) MERGE (m)-[:FROM]->(ep) }
                CALL (m) {
                  UNWIND $supersedes AS old_id
                  MATCH (old:Memory {id: old_id, user: $user}) WHERE old.invalidated_at IS NULL
                  SET old.invalidated_at = $now
                  MERGE (m)-[:SUPERSEDES]->(old)
                  RETURN collect(old.id) AS retired
                }
                RETURN retired
                """,
                id=mid, text=mem.text, kind=mem.kind, imp=mem.importance,
                occurred=_to_date(mem.occurred_on), due=_to_date(mem.due_on),
                annual=_annual(mem.annual_date), sensitive=mem.sensitive,
                vec=vec, about=about, ep=episode_id or "", supersedes=mem.supersedes, now=now,
            )[0]["retired"]
            flags = (" (sensitive)" if mem.sensitive else "") + (f" (yearly {mem.annual_date})" if _annual(mem.annual_date) else "")
            log.append(f"+ [{mem.kind}] {mem.text}{flags}" + (f"  (replaces {', '.join(retired)})" if retired else ""))

        for name in ex.forget_entities:  # last, so memories made this turn about them go too
            gone = self.forget_entity(name, known)
            if gone:
                log.append(f"forgot {gone['name']}: {gone['memories']} memories deleted, "
                           f"{gone['turns']} turns and {gone['episodes']} summaries redacted")
        return log

    # ------------------------------------------------------------------ read path
    def extraction_candidates(self, text: str, k: int = 10, episode_id: str | None = None) -> list[dict]:
        """Existing current memories the extractor may update or retire, with their ids:
        vector matches + memories about entities named in the text + open plans
        (+ everything already taken from this episode, for the end-of-episode pass)."""
        return self._q(
            """
            CALL db.index.vector.queryNodes('memory_vec', $k, $vec) YIELD node AS m, score
            WHERE m.user = $user AND m.invalidated_at IS NULL AND coalesce(m.status, '') <> 'done'
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.status AS status
            UNION
            MATCH (m:Memory)-[:ABOUT]->(e:Entity {user: $user}) WHERE e.kind <> 'self' AND m.invalidated_at IS NULL
              AND coalesce(m.status, '') <> 'done'
              AND any(n IN [e.name] + coalesce(e.aliases, []) WHERE size(n) > 2 AND toLower($text) CONTAINS toLower(n))
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.status AS status
            UNION
            MATCH (m:Memory {user: $user, kind: 'plan', status: 'open'}) WHERE m.invalidated_at IS NULL
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.status AS status
            UNION
            MATCH (m:Memory)-[:FROM]->(:Episode {id: $ep}) WHERE m.invalidated_at IS NULL
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.status AS status
            """,
            k=k, vec=embed([text])[0], text=text, ep=episode_id or "",
        )

    def known_entities(self) -> list[str]:
        """Entities shown to the extractor, with ids so it can tell two Sams apart."""
        rows = self._q(
            """
            MATCH (e:Entity {user: $user}) WHERE e.kind <> 'self'
            OPTIONAL MATCH (:User {id: $user})-[r:RELATED]->(e) WHERE r.until IS NULL
            RETURN e.id AS id, e.name AS name, e.kind AS kind, e.summary AS summary,
                   collect(r.type) AS rel ORDER BY e.name
            """
        )
        return [f"{r['id']}: {r['name']} ({r['kind']}{', ' + '/'.join(r['rel']) if r['rel'] else ''})"
                + (f" — {r['summary']}" if r["summary"] else "") for r in rows]

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
                AND NOT coalesce(m.sensitive, false) AND """ + NOT_PROTECTED + """
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
              AND (m.kind = 'boundary' OR """ + NOT_PROTECTED + """)
            WITH collect({m: m, score: score}) AS hits
            WITH hits, reduce(top = 0.0, h IN hits | CASE WHEN h.score > top THEN h.score ELSE top END) AS top
            UNWIND hits AS h
            WITH h.m AS m, h.score AS score, top
            WHERE NOT coalesce(m.sensitive, false)  // sensitive: only as (near) the best match
               OR (score >= top - $margin AND score >= $floor)
            WITH m, score,
                 duration.inDays(coalesce(m.last_accessed, m.created_at), datetime()).days AS age
            WITH m, score, 0.6 * score + 0.25 * m.importance / 10.0
                 + 0.15 * exp(-age / CASE m.kind WHEN 'feeling' THEN 7.0 ELSE 90.0 END) AS rank
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.occurred_on AS occurred_on,
                   coalesce(m.sensitive, false) AS sensitive, round(score, 3) AS score, round(rank, 3) AS rank
            ORDER BY rank DESC LIMIT $k
            """,
            vec=vec, k=k, margin=SENSITIVE_MARGIN, floor=SENSITIVE_FLOOR,
        )
        # History ("where did I live before?"): embeddings can't express "before", so walk the
        # graph instead — older versions of the best-matching current memories.
        top = max((m["score"] for m in semantic), default=0)
        best = [m["id"] for m in sorted(semantic, key=lambda m: -m["score"])[:3] if m["score"] >= top - 0.02]
        history = self._q(
            """
            MATCH (cur:Memory)-[:SUPERSEDES*1..3]->(old:Memory) WHERE cur.id IN $ids
            RETURN old.id AS id, old.text AS text, date(old.invalidated_at) AS until, cur.text AS replaced_by
            """,
            ids=best,
        )
        groups = [g for g, pattern in GROUP_WORDS.items() if re.search(rf"\b({pattern})\b", query, re.I)]
        group_types = sorted(set().union(*(RELATION_GROUPS[g] for g in groups))) if groups else []
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
              UNION  // "anyone in my family / at work...": everyone in that group of relations
              MATCH (:User {id: $user})-[r:RELATED]->(e:Entity) WHERE r.type IN $types AND r.until IS NULL
              RETURN e
            }
            WITH e WHERE NOT EXISTS {  // entities protected by a boundary are never surfaced
              MATCH (b:Memory {kind: 'boundary'})-[:ABOUT]->(e) WHERE b.invalidated_at IS NULL }
            OPTIONAL MATCH (:User {id: $user})-[r:RELATED]->(e) WHERE r.until IS NULL
            OPTIONAL MATCH (m:Memory)-[:ABOUT]->(e)
              WHERE m.invalidated_at IS NULL AND coalesce(m.status, '') <> 'done' AND NOT coalesce(m.sensitive, false)
            WITH e, r, m ORDER BY m.importance DESC
            WITH e, head(collect(r.type)) AS relation, collect(DISTINCT m {.id, .kind, .text})[..5] AS memories
            RETURN e.name AS name, e.kind AS kind, e.summary AS summary, relation, memories
            LIMIT 12
            """,
            q=query, vec=vec, types=group_types,
        )
        people = self._q(
            """
            MATCH (:User {id: $user})-[r:RELATED]->(e:Entity) WHERE r.until IS NULL
              AND NOT EXISTS { MATCH (b:Memory {kind: 'boundary'})-[:ABOUT]->(e) WHERE b.invalidated_at IS NULL }
            RETURN e.id AS id, e.name AS name, e.kind AS kind, r.type AS relation, e.summary AS summary
            ORDER BY e.kind, e.name LIMIT 25
            """
        )
        followups = self._q(
            """
            MATCH (m:Memory {user: $user, kind: 'plan', status: 'open'})
            WHERE m.invalidated_at IS NULL AND m.due_on IS NOT NULL
              AND m.due_on >= $today - duration({days: 14}) AND m.due_on <= $today + duration({days: 14})
              AND """ + NOT_PROTECTED + """
            RETURN m.id AS id, m.text AS text, m.due_on AS due_on ORDER BY m.due_on
            """,
            today=today,
        )
        # Yearly dates (birthdays, anniversaries) coming up in the next two weeks.
        upcoming = []
        for m in self._q(
            """
            MATCH (m:Memory {user: $user}) WHERE m.annual_date IS NOT NULL AND m.invalidated_at IS NULL
              AND """ + NOT_PROTECTED + """
            RETURN m.id AS id, m.text AS text, m.annual_date AS annual_date
            """
        ):
            when = _next_occurrence(m["annual_date"], today)
            if when and when <= today + dt.timedelta(days=14):
                upcoming.append({**m, "on": when})
        upcoming.sort(key=lambda m: m["on"])
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
                "followups": followups, "upcoming": upcoming, "history": history, "episodes": episodes}

    # ------------------------------------------------------------------ inspection
    def memories(self, kind: str | None = None, include_retired: bool = False) -> list[dict]:
        return self._q(
            """
            MATCH (m:Memory {user: $user})
            WHERE ($kind IS NULL OR m.kind = $kind) AND ($all OR m.invalidated_at IS NULL)
            OPTIONAL MATCH (m)-[:ABOUT]->(e)
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.importance AS importance,
                   m.status AS status, m.invalidated_at IS NOT NULL AS retired,
                   coalesce(m.sensitive, false) AS sensitive, m.annual_date AS annual_date,
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

    def forget_entity(self, name: str, known: dict[str, str] | None = None) -> dict | None:
        """"Forget everything about Jordan": delete the entity and every memory about or naming
        it, and redact the name from stored turns and episode summaries."""
        eid = self._lookup(name, known or {})
        if not eid or eid == self.user:
            return None
        ent = self._q("MATCH (e:Entity {id: $id}) RETURN e.name AS name, coalesce(e.aliases, []) AS aliases", id=eid)[0]
        terms = sorted({t for t in [ent["name"], *ent["aliases"]] if len(t) > 2}, key=len, reverse=True)
        pattern = re.compile(r"\b(" + "|".join(map(re.escape, terms)) + r")\b", re.I)
        lowered = [t.lower() for t in terms]
        memories = self._q(
            """
            MATCH (m:Memory {user: $user})
            WHERE EXISTS { (m)-[:ABOUT]->(:Entity {id: $id}) } OR any(t IN $terms WHERE toLower(m.text) CONTAINS t)
            DETACH DELETE m RETURN count(*) AS n
            """,
            id=eid, terms=lowered,
        )[0]["n"]
        turns = self._q(
            """
            MATCH (:Episode {user: $user})-[:HAS_TURN]->(t:Turn) WHERE any(x IN $terms WHERE toLower(t.text) CONTAINS x)
            RETURN elementId(t) AS id, t.text AS text
            """,
            terms=lowered,
        )
        for t in turns:
            self._q("MATCH (t) WHERE elementId(t) = $id SET t.text = $text",
                    id=t["id"], text=pattern.sub("[forgotten]", t["text"]))
        episodes = self._q(
            """
            MATCH (ep:Episode {user: $user}) WHERE any(x IN $terms WHERE toLower(coalesce(ep.summary, '')) CONTAINS x)
            RETURN ep.id AS id, ep.summary AS summary
            """,
            terms=lowered,
        )
        for ep in episodes:
            text = pattern.sub("[forgotten]", ep["summary"])
            self._q(
                """
                MATCH (ep:Episode {id: $id}) SET ep.summary = $text
                WITH ep CALL db.create.setNodeVectorProperty(ep, 'embedding', $vec)
                """,
                id=ep["id"], text=text, vec=embed([text])[0],
            )
        self._q("MATCH (e:Entity {id: $id}) DETACH DELETE e", id=eid)
        return {"name": ent["name"], "memories": memories, "turns": len(turns), "episodes": len(episodes)}

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
            f"- {p['name']} ({p['relation']})" + (f" — {p['summary']}" if p.get("summary") else "")
            for p in r["people"]))
    if r["followups"]:
        out.append("Open plans to follow up on (today is " + today.isoformat() + "):\n" + "\n".join(
            f"- {m['text']} (due {m['due_on']})" for m in r["followups"]))
    if r.get("upcoming"):
        out.append("Yearly dates coming up:\n" + "\n".join(
            f"- {m['on'].isoformat()}: {m['text']}" for m in r["upcoming"]))
    seen = {m["text"] for m in r["core"] + r["followups"]}
    relevant = [m for m in r["semantic"] if m["text"] not in seen]
    for e in r["entities"]:
        relevant += [m for m in e["memories"] if m["text"] not in seen | {x["text"] for x in relevant}]
    if relevant:
        out.append("Relevant memories:\n" + "\n".join(
            f"- [{m['kind']}] {m['text']}" + (" (sensitive: only discuss if they raise it)" if m.get("sensitive") else "")
            for m in relevant))
    if r.get("history"):
        out.append("No longer true (history, not current):\n" + "\n".join(
            f"- until {h['until']}: {h['text']}" + (f" → now: {h['replaced_by']}" if h["replaced_by"] else "")
            for h in r["history"]))
    if r["episodes"]:
        out.append("Recent conversations:\n" + "\n".join(
            f"- {str(e['at'])[:10]} (mood: {e['mood']}): {e['summary']}" for e in r["episodes"]))
    return "\n\n".join(out) or "(No memories yet — this is a new relationship.)"
