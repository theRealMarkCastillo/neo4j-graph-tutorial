"""Lessons 9-10 — MemoryStore: writes memories into the graph and recalls them for a turn.

Write path (Lesson 9):  resolve entities -> link relations -> dedupe -> supersede -> store
Read path  (Lesson 10): core profile + semantic (vector) + entity hop (graph) + follow-ups
                        + recent episodes, ranked by similarity, importance and recency
"""
from __future__ import annotations

import datetime as dt
import re
import uuid
from functools import partial

import numpy as np

from fastembed import TextEmbedding

from config import EMBED_MODEL
from companion import clock, decay, ontology
from companion.models import Extraction
from companion.schema import ensure_schema

_embedder: TextEmbedding | None = None
_ontology_loaded = False


def embed(texts: list[str]) -> list[list[float]]:
    global _embedder
    _embedder = _embedder or TextEmbedding(EMBED_MODEL)
    return [v.tolist() for v in _embedder.embed(texts)]


def _name_similarity(a: list[str], b: list[str]) -> float:
    """Best cosine similarity between any name/alias in `a` and any in `b`."""
    va, vb = np.array(embed(a)), np.array(embed(b))
    va /= np.linalg.norm(va, axis=1, keepdims=True)
    vb /= np.linalg.norm(vb, axis=1, keepdims=True)
    return float((va @ vb.T).max())


def _clusters(rows: list[dict], min_sim: float, min_size: int) -> list[list[dict]]:
    """Greedy complete-linkage clustering: a row joins a cluster only if it is at least
    `min_sim` similar to EVERY member, so "gym" can't chain via "climbing" to "hiking"."""
    clusters: list[list[dict]] = []
    for r in rows:
        v = np.asarray(r["vec"])
        for c in clusters:
            if all(float(np.dot(v, np.asarray(m["vec"]))) >= min_sim for m in c):
                c.append(r)
                break
        else:
            clusters.append([r])
    return [c for c in clusters if len(c) >= min_size]


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


def _to_date(value: str | None) -> dt.date | None:
    try:
        return dt.date.fromisoformat(value) if value else None
    except ValueError:  # "2019" or "next spring" stay in the memory text instead
        return None


# Duplicate threshold on Neo4j's 0-1 rescaled cosine (0.975 ≈ raw cosine 0.95).
DUPLICATE_SCORE = 0.975
# A memory this close to one the user deleted in the same episode is not stored again.
FORGOTTEN_SCORE = 0.90
# Expired situational memories stay visible as "Recently ..." for this long.
RECENT_HOURS = 24
# Consolidation (Lesson 13)
LAPSE_AFTER_DAYS = 14        # an open plan this far past due stops being followed up
PATTERN_WINDOW_DAYS = 60     # look for repeats among memories this recent
PATTERN_SIMILARITY = 0.82    # EVERY pair in a cluster at least this similar. Measured: gym sentences
                             # 0.83-0.91; rock climbing 0.78-0.82 to them, so it stays out. The
                             # writer LLM can still decline a cluster that isn't a habit.
PATTERN_MIN = 3              # this many similar memories make a pattern
PRUNE_MAX_IMPORTANCE = 3     # only trivia is ever deleted
# Entity match threshold when names differ ("Mom" vs "my mother").
ENTITY_MATCH_SCORE = 0.93
# ...and the NAMES must also be similar (raw cosine of name-only embeddings). Without this,
# "Seattle: the city where Riley now lives" matched "Boston: the city where Riley lives" —
# the summaries dominate the embedding. Mom/mother 0.87, NYC/New York 0.93; Seattle/Boston 0.64.
NAME_MATCH_MIN = 0.80
# Sensitive memories (health, finances...) are recalled only when the message is clearly about
# them: (near) the best vector match for this message, not just above a fixed score. Small
# embedding models squeeze all scores into a narrow band (~0.63-0.80 here), so relative wins.
SENSITIVE_MARGIN = 0.01
SENSITIVE_FLOOR = 0.72

# A memory about someone a boundary protects ("don't bring up Dana") is never recalled.
# The user node is excluded: a topic boundary ("don't ask about my job") is ABOUT the user,
# and must not hide everything else about them (found by tests/test_10_recall.py).
NOT_PROTECTED = """NOT EXISTS {
    MATCH (m)-[:ABOUT]->(protected:Entity)<-[:ABOUT]-(b:Memory {kind: 'boundary'})
    WHERE b.invalidated_at IS NULL AND (b.valid_until IS NULL OR b.valid_until > $now) AND b <> m AND NOT protected:User }"""
# (Its variable names must not clash with the query it's pasted into: an outer `e` would
# silently turn "any entity" into "this entity" inside the EXISTS.)

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
        global _ontology_loaded
        if not _ontology_loaded:
            ontology.ensure_ontology(driver)
            _ontology_loaded = True
        self._q(
            """
            MERGE (u:Entity:User {id: $user})
            ON CREATE SET u.created_at = datetime()
            SET u.name = $name, u.user = $user, u.kind = 'self'
            WITH u CALL db.create.setNodeVectorProperty(u, 'embedding', $vec)
            """,
            name=user_name, vec=embed([f"{user_name}, the user"])[0],
        )

    def _q(self, query: str, now: dt.datetime | None = None, **params) -> list[dict]:
        """Run a query with $user and $now always set ($now: the conversation's clock)."""
        records, _, _ = self.driver.execute_query(
            query, user=self.user, now=now or clock.now_for(dt.date.today()), **params)
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
        if not summary:  # drop an empty episode; close one whose summary failed, so NEXT links it
            self._q(
                """
                MATCH (ep:Episode {id: $ep})
                CALL (ep) { WITH ep WHERE NOT (ep)-[:HAS_TURN]->() DETACH DELETE ep }
                CALL (ep) { WITH ep WHERE (ep)-[:HAS_TURN]->() SET ep.ended_at = datetime() }
                """,
                ep=episode_id,
            )
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
            candidates = self._q(
                """
                CALL db.index.vector.queryNodes('entity_vec', 50, $vec) YIELD node, score
                WHERE node.user = $user AND node.kind = $kind AND score >= $min
                RETURN node.id AS id, [node.name] + coalesce(node.aliases, []) AS names ORDER BY score DESC
                """,
                vec=vec, kind=kind, min=ENTITY_MATCH_SCORE,
            )
            hit = [c for c in candidates if _name_similarity([name, *aliases], c["names"]) >= NAME_MATCH_MIN][:1]
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
        now = clock.now_for(today)
        q = partial(self._q, now=now)
        log: list[str] = []
        known: dict[str, str] = {}
        self.forgotten_terms: list[str] = []  # names forgotten by this call (the graph redacts its thread)
        for ent in ex.entities:
            eid = self.resolve_entity(ent.name, ent.kind, ent.aliases, ent.summary, ent.ref, ent.new)
            for n in [ent.name, *ent.aliases, *([ent.ref] if ent.ref else [])]:
                if known.get(n.lower(), eid) != eid:  # two different Sams in one extraction
                    log.append(f"warning: '{n}' names two entities here; references to it go to the last one")
                known[n.lower()] = eid
            if ent.new and not ent.ref:
                log.append(f"new entity {ent.name} ({eid}), distinct from the known {ent.name}")

        # Relations go through the ontology: normalize the type, check its range, then apply
        # its rules (cardinality "one" ends the previous edge; "ended" ends the whole lineage).
        resolved = []
        for rel in ex.relations:
            a, b = self._lookup(rel.source, known), self._lookup(rel.target, known)
            if a and b and a != b:
                canon, how, flip = ontology.normalize(rel.type)
                if how == "group":
                    log.append(f"rejected {rel.source} -{rel.type}-> {rel.target}: '{rel.type}' is a "
                               "group in the ontology, not a relation type")
                    continue
                resolved.append((rel, *((b, a) if flip else (a, b)), canon, "flipped" if flip else how))
        # Guard (Lesson 13): a one-at-a-time relation (lives_in) to something that only a
        # temporary status mentions is a trip, not a move. Applying it would end the real home.
        temporary = {n.lower() for m in ex.memories if m.kind == "status" for n in m.about}
        lasting = {n.lower() for m in ex.memories if m.kind != "status" for n in m.about}
        kept = []
        for item in resolved:
            rel, canon = item[0], item[3]
            target = rel.target.lower()
            if (not rel.ended and ontology.rule(canon, "cardinality") == "one"
                    and target in temporary and target not in lasting):
                log.append(f"ignored {rel.source} -{canon}-> {rel.target}: only a temporary status "
                           "mentions it (a trip, not a move)")
                continue
            kept.append(item)
        resolved = kept
        linking = {(a, b, canon) for rel, a, b, canon, _ in resolved if not rel.ended}
        for rel, a, b, canon, how in resolved:
            if rel.ended and (a, b, canon) in linking:
                continue  # "ended X" + "linked X" in one go: keep it linked
            info = q("MATCH (a:Entity {id: $a}), (b:Entity {id: $b}) RETURN a.name AS a, b.name AS b, b.kind AS kind",
                           a=a, b=b)[0]
            src, dst = ("user" if a == self.user else info["a"]), info["b"]
            said = f" (said '{rel.type}')" if how in ("synonym", "flipped") else ""
            if rel.ended:  # breakup, moved away, left a job: keep the edge as history
                ended = q(
                    """
                    MATCH (a:Entity {id: $a})-[r:RELATED]->(b:Entity {id: $b})
                    WHERE r.until IS NULL AND r.type IN $types
                    SET r.until = $today RETURN r.type AS type
                    """,
                    a=a, b=b, types=sorted(ontology.lineage(canon) | {canon}), today=today,
                )
                if ended:
                    log.append(f"ended {src} -{'/'.join(e['type'] for e in ended)}-> {dst}{said}")
                continue
            if not ontology.range_ok(canon, info["kind"]):
                log.append(f"rejected {src} -{canon}-> {dst}: {canon} needs a {'/'.join(ontology.rule(canon, 'range'))}, "
                           f"{dst} is a {info['kind']}")
                continue
            if how == "new":
                hint = ontology.propose_type(self.driver, canon, f"{src} -{rel.type}-> {dst}")
                log.append(f"new relation type '{canon}' proposed for the ontology (closest: {hint})")
            if ontology.rule(canon, "cardinality") == "one":  # e.g. one lives_in at a time
                moved = q(
                    """
                    MATCH (a:Entity {id: $a})-[r:RELATED]->(other:Entity)
                    WHERE r.until IS NULL AND r.type IN $types AND (other.id <> $b OR r.type <> $type)
                    SET r.until = $today RETURN other.name AS name, r.type AS type
                    """,
                    a=a, b=b, type=canon, types=sorted(ontology.lineage(canon) | {canon}), today=today,
                )
                for m in moved:
                    log.append(f"ended {src} -{m['type']}-> {m['name']} (one {canon} at a time)")
            changed = q(
                """
                MATCH (a:Entity {id: $a}), (b:Entity {id: $b})
                MERGE (a)-[r:RELATED {type: $type}]->(b)
                WITH r, r.updated_at IS NULL OR r.until IS NOT NULL AS changed
                SET r.since = coalesce($since, r.since), r.updated_at = $now, r.until = null,
                    r.said = $said_as
                RETURN changed
                """,
                a=a, b=b, type=canon, since=rel.since, now=now, said_as=rel.type,
            )[0]["changed"]
            if changed:  # only log new or re-activated links
                log.append(f"linked {src} -{canon}-> {dst}{said}")

        for mid in ex.invalidate:
            if q(
                """
                MATCH (m:Memory {id: $id, user: $user}) WHERE m.invalidated_at IS NULL
                SET m.invalidated_at = $now RETURN m.text AS text
                """, id=mid, now=now,
            ):
                log.append(f"retired {mid}")
        for mid in ex.completed_plans:
            if q("MATCH (m:Memory {id: $id, user: $user, kind: 'plan', status: 'open'}) SET m.status = 'done' RETURN m.id", id=mid):
                log.append(f"plan done {mid}")

        vectors = embed([m.text for m in ex.memories]) if ex.memories else []
        tombstones = self._tombstones(episode_id)
        for mem, vec in zip(ex.memories, vectors):
            until = clock.parse_until(mem.valid_until, now, mem.kind)
            if tombstones and max(float(np.dot(vec, t)) for t in tombstones) >= FORGOTTEN_SCORE:
                log.append("skipped a memory the user asked to forget earlier in this conversation")
                continue
            dup = q(
                """
                CALL db.index.vector.queryNodes('memory_vec', 50, $vec) YIELD node, score
                WHERE node.user = $user AND node.invalidated_at IS NULL AND score >= $min
                  AND (node.valid_until IS NULL OR node.valid_until > $now)  // expired: say it again = new
                  // same dates too: "dentist on 2026-11-12" -> "...on 2026-11-17" reads almost the
                  // same but is a real correction; an identical "replacement" is just a duplicate
                  AND coalesce(node.due_on, date('1900-01-01')) = coalesce($due, date('1900-01-01'))
                  AND coalesce(node.occurred_on, date('1900-01-01')) = coalesce($occurred, date('1900-01-01'))
                RETURN node.id AS id LIMIT 1
                """,
                vec=vec, min=DUPLICATE_SCORE, due=_to_date(mem.due_on), occurred=_to_date(mem.occurred_on),
            )
            if dup:  # already known: reinforce instead of storing a near-copy
                q(
                    """
                    MATCH (m:Memory {id: $id})
                    SET m.access_count = coalesce(m.access_count, 0) + 1, m.last_confirmed = $now,
                        m.valid_until = CASE WHEN m.valid_until IS NULL OR $until IS NULL THEN m.valid_until
                                             WHEN $until > m.valid_until THEN $until ELSE m.valid_until END,
                        m.importance = CASE WHEN $imp > m.importance THEN $imp ELSE m.importance END
                    """,
                    id=dup[0]["id"], imp=mem.importance, until=until,
                )
                log.append(f"reinforced {dup[0]['id']}")
                # The near-copy may still be a correction ("brother is Leo" superseding "...Theo"):
                # retire what it replaces, or the wrong memory stays current.
                for old in q(
                    """
                    MATCH (keep:Memory {id: $keep}), (old:Memory {user: $user})
                    WHERE old.id IN $ids AND old.id <> $keep AND old.invalidated_at IS NULL
                    SET old.invalidated_at = $now MERGE (keep)-[:SUPERSEDES]->(old) RETURN old.id AS id
                    """,
                    keep=dup[0]["id"], ids=mem.supersedes, now=now,
                ):
                    log.append(f"retired {old['id']} (replaced by {dup[0]['id']})")
                continue

            mid = _new_id("m")
            about = [eid for n in (mem.about or ["user"]) if (eid := self._lookup(n, known))]
            retired = q(
                """
                CREATE (m:Memory {id: $id, user: $user, text: $text, kind: $kind, importance: $imp,
                                  occurred_on: $occurred, due_on: $due, annual_date: $annual,
                                  sensitive: $sensitive, valid_until: $until,
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
                annual=_annual(mem.annual_date), sensitive=mem.sensitive, until=until,
                vec=vec, about=about, ep=episode_id or "", supersedes=mem.supersedes, now=now,
            )[0]["retired"]
            flags = (" (sensitive)" if mem.sensitive else "") + (f" (yearly {mem.annual_date})" if _annual(mem.annual_date) else "")
            if until:
                flags += f" (until {until:%Y-%m-%d %H:%M})"
            log.append(f"+ [{mem.kind}] {mem.text}{flags}" + (f"  (replaces {', '.join(retired)})" if retired else ""))

        for name in ex.forget_entities:  # last, so memories made this turn about them go too
            gone = self.forget_entity(name, known)
            if gone:
                self.forgotten_terms += gone["terms"]
                log.append(f"forgot {gone['name']}: {gone['memories']} memories deleted, "
                           f"{gone['turns']} turns and {gone['episodes']} summaries redacted")
        return log

    # ------------------------------------------------------------------ read path
    def extraction_candidates(self, text: str, k: int = 10, episode_id: str | None = None,
                              today: dt.date | None = None) -> list[dict]:
        """Existing current memories the extractor may update or retire, with their ids:
        vector matches + memories about entities named in the text + open plans
        (+ everything already taken from this episode, for the end-of-episode pass)."""
        return self._q(
            """
            CALL db.index.vector.queryNodes('memory_vec', 50, $vec) YIELD node AS m, score
            WHERE m.user = $user AND m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now) AND coalesce(m.status, '') <> 'done'
            WITH m ORDER BY score DESC LIMIT $k  // filter first, then keep the top k
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.status AS status
            UNION
            MATCH (m:Memory)-[:ABOUT]->(e:Entity {user: $user}) WHERE e.kind <> 'self' AND m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now)
              AND coalesce(m.status, '') <> 'done'
              AND any(n IN [e.name] + coalesce(e.aliases, []) WHERE size(n) > 2 AND toLower($text) CONTAINS toLower(n))
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.status AS status
            UNION
            MATCH (m:Memory {user: $user, kind: 'plan', status: 'open'}) WHERE m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now)
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.status AS status
            UNION
            MATCH (m:Memory)-[:FROM]->(:Episode {id: $ep}) WHERE m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now)
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.status AS status
            """,
            k=k, vec=embed([text])[0], text=text, ep=episode_id or "",
            now=clock.now_for(today or dt.date.today()),
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

    def inferred_relations(self, now: dt.datetime | None = None) -> list[dict]:
        """Relatives the user never linked directly, inferred with the ontology's kinship rules
        (e.g. user -brother-> Leo -daughter-> Ana  =>  Ana is the user's niece_or_nephew).
        Entities protected by a boundary are left out."""
        edges = self._q(
            """
            MATCH (a:Entity {user: $user})-[r:RELATED]->(b:Entity {user: $user}) WHERE r.until IS NULL
            RETURN a.id AS src, b.id AS dst, r.type AS type
            """
        )
        found = ontology.infer_kin(edges, self.user)
        if not found:
            return []
        rows = {r["id"]: r for r in self._q(
            """
            MATCH (e:Entity) WHERE e.id IN $ids
              AND NOT EXISTS { MATCH (b:Memory {kind: 'boundary'})-[:ABOUT]->(e) WHERE b.invalidated_at IS NULL AND (b.valid_until IS NULL OR b.valid_until > $now) }
            RETURN e.id AS id, e.name AS name, e.kind AS kind, e.summary AS summary
            """,
            ids=[f["id"] for f in found] + [f["via"] for f in found], now=now,
        )}
        return [{**f, **rows[f["id"]], "via_name": rows[f["via"]]["name"]}
                for f in found if f["id"] in rows and f["via"] in rows]

    def recall(self, query: str, k: int = 8, today: dt.date | None = None) -> dict:
        today = today or dt.date.today()
        now = clock.now_for(today)
        q = partial(self._q, now=now)  # every query below sees the conversation's clock
        vec = embed([query])[0]
        # Core profile: every boundary + the most important stable facts and preferences.
        core = q(
            """
            CALL () {
              MATCH (m:Memory {user: $user, kind: 'boundary'}) WHERE m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now)
              RETURN m
              UNION
              MATCH (m:Memory {user: $user}) WHERE m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now)
                AND m.kind IN ['fact', 'preference'] AND m.importance >= 6
                AND NOT coalesce(m.sensitive, false) AND """ + NOT_PROTECTED + """
              WITH m ORDER BY m.importance DESC, m.created_at DESC LIMIT 8
              RETURN m
            }
            RETURN m.id AS id, m.kind AS kind, m.text AS text,
                   duration.inSeconds(coalesce(m.last_confirmed, m.created_at), $now).seconds / 86400.0 AS age
            ORDER BY CASE m.kind WHEN 'boundary' THEN 0 ELSE 1 END, m.importance DESC
            """
        )
        for m in core:
            m["stale"] = decay.is_stale(m["kind"], m.pop("age"))
        # Semantic: vector similarity re-ranked by importance and recency (feelings fade fast).
        semantic = q(
            """
            CALL db.index.vector.queryNodes('memory_vec', 50, $vec) YIELD node AS m, score
            WHERE m.user = $user AND m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now) AND coalesce(m.status, '') <> 'done'
              AND (m.kind = 'boundary' OR """ + NOT_PROTECTED + """)
            WITH collect({m: m, score: score}) AS hits
            WITH hits, reduce(top = 0.0, h IN hits | CASE WHEN h.score > top THEN h.score ELSE top END) AS top
            UNWIND hits AS h
            WITH h.m AS m, h.score AS score, top
            WHERE NOT coalesce(m.sensitive, false)  // sensitive: only as (near) the best match
               OR (score >= top - $margin AND score >= $floor)
            // Age since it was stored or last *confirmed* by the user, on the conversation's clock.
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.occurred_on AS occurred_on,
                   m.importance AS importance, coalesce(m.sensitive, false) AS sensitive,
                   round(score, 3) AS score, duration.inSeconds(coalesce(m.last_confirmed, m.created_at), $now).seconds / 86400.0 AS age
            """,
            vec=vec, margin=SENSITIVE_MARGIN, floor=SENSITIVE_FLOOR,
        )
        # Decay (Lesson 13) in Python, where the curve is unit-tested: drop dormant memories,
        # then order by match + strength.
        semantic = sorted(
            ({**m, "rank": round(decay.rank(m["score"], m["importance"], m["kind"], m["age"]), 3),
              "stale": decay.is_stale(m["kind"], m["age"])}
             for m in semantic if not decay.is_dormant(m["importance"], m["kind"], m["age"])),
            key=lambda m: -m["rank"])[:k]
        # History ("where did I live before?"): embeddings can't express "before", so walk the
        # graph instead — older versions of the best-matching current memories.
        top = max((m["score"] for m in semantic), default=0)
        best = [m["id"] for m in sorted(semantic, key=lambda m: -m["score"])[:3] if m["score"] >= top - 0.02]
        history = q(
            """
            MATCH (cur:Memory)-[:SUPERSEDES*1..3]->(m:Memory) WHERE cur.id IN $ids AND """ + NOT_PROTECTED + """
            WITH cur, m AS old
            RETURN old.id AS id, old.text AS text, date(old.invalidated_at) AS until, cur.text AS replaced_by
            """,
            ids=best,
        )
        # "Anyone in my family / at work...": the ontology says which relation types each group
        # covers, and inference adds relatives nobody linked directly (a brother's daughter).
        groups = ontology.groups_in(query)
        group_types = sorted(set().union(*(ontology.types_in_group(g) for g in groups))) if groups else []
        inferred = self.inferred_relations(now)
        inferred_ids = [i["id"] for i in inferred if i["relation"] in group_types]
        # Graph hop: entities named (or meant) in the message -> their memories and relations.
        entities = q(
            """
            CALL () {
              MATCH (e:Entity {user: $user}) WHERE e.kind <> 'self'
                AND any(n IN [e.name] + coalesce(e.aliases, []) WHERE size(n) > 2 AND toLower($q) CONTAINS toLower(n))
              RETURN e
              UNION
              CALL db.index.vector.queryNodes('entity_vec', 50, $vec) YIELD node AS e, score
              WITH e, score WHERE e.user = $user AND e.kind <> 'self' AND score >= 0.85
              RETURN e ORDER BY score DESC LIMIT 3
              UNION  // "anyone in my family / at work...": everyone in that group of relations
              MATCH (:User {id: $user})-[r:RELATED]->(e:Entity) WHERE r.type IN $types AND r.until IS NULL
              RETURN e
              UNION  // ...including inferred relatives
              MATCH (e:Entity {user: $user}) WHERE e.id IN $inferred
              RETURN e
            }
            WITH e WHERE NOT EXISTS {  // entities protected by a boundary are never surfaced
              MATCH (b:Memory {kind: 'boundary'})-[:ABOUT]->(e) WHERE b.invalidated_at IS NULL AND (b.valid_until IS NULL OR b.valid_until > $now) }
            OPTIONAL MATCH (:User {id: $user})-[r:RELATED]->(e) WHERE r.until IS NULL
            OPTIONAL MATCH (m:Memory)-[:ABOUT]->(e)
              WHERE m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now) AND coalesce(m.status, '') <> 'done' AND NOT coalesce(m.sensitive, false)
                AND """ + NOT_PROTECTED + """
            WITH e, r, m ORDER BY m.importance DESC
            WITH e, head(collect(r.type)) AS relation,
                 collect(DISTINCT m {.id, .kind, .text, .importance,
                                     age: duration.inSeconds(coalesce(m.last_confirmed, m.created_at), $now).seconds / 86400.0})[..20] AS memories
            RETURN e.id AS id, e.name AS name, e.kind AS kind, e.summary AS summary, relation, memories
            LIMIT 12
            """,
            q=query, vec=vec, types=group_types, inferred=inferred_ids,
        )
        for e in entities:  # same decay rule as semantic recall
            e["memories"] = [m for m in e["memories"]
                             if not decay.is_dormant(m["importance"], m["kind"], m["age"])][:5]
        by_id = {i["id"]: i for i in inferred}
        for e in entities:
            if not e["relation"] and e["id"] in by_id:
                e["relation"] = f"{by_id[e['id']]['relation']}, inferred via {by_id[e['id']]['via_name']}"
        people = q(
            """
            MATCH (:User {id: $user})-[r:RELATED]->(e:Entity) WHERE r.until IS NULL
              AND NOT EXISTS { MATCH (b:Memory {kind: 'boundary'})-[:ABOUT]->(e) WHERE b.invalidated_at IS NULL AND (b.valid_until IS NULL OR b.valid_until > $now) }
            RETURN e.id AS id, e.name AS name, e.kind AS kind, r.type AS relation, e.summary AS summary
            ORDER BY e.kind, e.name LIMIT 25
            """
        )
        people += [{"id": i["id"], "name": i["name"], "kind": i["kind"], "summary": i["summary"],
                    "relation": f"{i['relation']}, inferred via {i['via_name']}"} for i in inferred]
        followups = q(
            """
            MATCH (m:Memory {user: $user, kind: 'plan', status: 'open'})
            WHERE m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now) AND m.due_on IS NOT NULL
              AND m.due_on >= $today - duration({days: 14}) AND m.due_on <= $today + duration({days: 14})
              AND """ + NOT_PROTECTED + """
            RETURN m.id AS id, m.text AS text, m.due_on AS due_on,
                   coalesce(m.sensitive, false) AS sensitive ORDER BY m.due_on
            """,
            today=today,
        )
        # Yearly dates (birthdays, anniversaries) coming up in the next two weeks.
        upcoming = []
        for m in q(
            """
            MATCH (m:Memory {user: $user}) WHERE m.annual_date IS NOT NULL AND m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now)
              AND """ + NOT_PROTECTED + """
            RETURN m.id AS id, m.text AS text, m.annual_date AS annual_date,
                   coalesce(m.sensitive, false) AS sensitive
            """
        ):
            when = _next_occurrence(m["annual_date"], today)
            if when and when <= today + dt.timedelta(days=14):
                upcoming.append({**m, "on": when})
        upcoming.sort(key=lambda m: m["on"])
        # Situational memories that ended in the last day ("at the gym" until 14:00): no longer
        # true, but a friend would ask how it went.
        recent = q(
            """
            MATCH (m:Memory {user: $user}) WHERE m.invalidated_at IS NULL
              AND m.valid_until <= $now AND m.valid_until > $now - duration({hours: $hours})
              AND """ + NOT_PROTECTED + """
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.valid_until AS until,
                   coalesce(m.sensitive, false) AS sensitive ORDER BY m.valid_until DESC LIMIT 5
            """,
            hours=RECENT_HOURS,
        )
        episodes = q(
            """
            MATCH (ep:Episode {user: $user}) WHERE ep.summary IS NOT NULL
            RETURN ep.started_at AS at, ep.summary AS summary, ep.mood AS mood
            ORDER BY ep.started_at DESC LIMIT 3
            """
        )
        touched = [m["id"] for m in semantic] + [m["id"] for e in entities for m in e["memories"]]
        q(
            """
            MATCH (m:Memory) WHERE m.id IN $ids
            SET m.last_accessed = datetime(), m.access_count = coalesce(m.access_count, 0) + 1
            """,
            ids=touched,
        )
        return {"core": core, "semantic": semantic, "entities": entities, "people": people,
                "followups": followups, "upcoming": upcoming, "history": history, "episodes": episodes,
                "recent": recent}

    # ------------------------------------------------------------------ consolidation
    def consolidate(self, today: dt.date | None = None, writer=None, prune: bool = True) -> list[str]:
        """Background upkeep (Lesson 13). Run after a conversation, or nightly:

        1. lapse   open plans long past due stop being followed up
        2. roll up repeated situational memories ("at the gym" x3) into one lasting pattern,
                   written by `writer(user, texts, today) -> MemoryOut | None` (an LLM call,
                   injected so tests can use a fake); the members are marked consolidated
        3. prune   delete dormant, low-importance statuses and feelings (data minimization),
                   but only once they're rolled up or older than the pattern window: pruning
                   earlier would delete the repeats before they could ever form a pattern

        Everything else is kept: events stay as history even when dormant."""
        today = today or dt.date.today()
        now = clock.now_for(today)
        q = partial(self._q, now=now)
        log = [f"plan lapsed: {r['text']}" for r in q(
            """
            MATCH (m:Memory {user: $user, kind: 'plan', status: 'open'})
            WHERE m.invalidated_at IS NULL AND m.due_on < date($now) - duration({days: $days})
            SET m.status = 'lapsed' RETURN m.text AS text
            """, days=LAPSE_AFTER_DAYS)]

        if writer is not None:
            rows = q(
                """
                MATCH (m:Memory {user: $user}) WHERE m.invalidated_at IS NULL AND m.consolidated_at IS NULL
                  AND m.kind IN ['event', 'status', 'feeling'] AND NOT coalesce(m.sensitive, false)
                  AND m.created_at > $now - duration({days: $days})
                RETURN m.id AS id, m.text AS text, m.embedding AS vec ORDER BY m.created_at
                """, days=PATTERN_WINDOW_DAYS)
            for group in _clusters(rows, PATTERN_SIMILARITY, PATTERN_MIN):
                pattern = writer(self.user_name, [r["text"] for r in group], today)
                if pattern is None:
                    continue
                log += [f"pattern from {len(group)} memories: {line}"
                        for line in self.apply(Extraction(memories=[pattern]), today=today)]
                q(
                    """
                    MATCH (p:Memory {user: $user, text: $text}) WHERE p.invalidated_at IS NULL
                    WITH p ORDER BY p.created_at DESC LIMIT 1
                    MATCH (m:Memory) WHERE m.id IN $ids
                    SET m.consolidated_at = $now MERGE (p)-[:SUMMARIZES]->(m)
                    WITH p, count(m) AS n SET p.evidence = coalesce(p.evidence, 0) + n
                    """, text=pattern.text, ids=[r["id"] for r in group])

        if prune:
            dormant = [r for r in q(
                """
                MATCH (m:Memory {user: $user}) WHERE m.kind IN ['status', 'feeling']
                  AND m.importance <= $max_imp AND NOT EXISTS { (m)<-[:SUPERSEDES]-() }
                  AND (m.consolidated_at IS NOT NULL OR m.created_at < $now - duration({days: $days}))
                RETURN m.id AS id, m.kind AS kind, m.importance AS importance,
                       duration.inSeconds(coalesce(m.last_confirmed, m.created_at), $now).seconds / 86400.0 AS age
                """, max_imp=PRUNE_MAX_IMPORTANCE, days=PATTERN_WINDOW_DAYS) if decay.is_dormant(r["importance"], r["kind"], r["age"])]
            if dormant:
                q("MATCH (m:Memory) WHERE m.id IN $ids DETACH DELETE m", ids=[r["id"] for r in dormant])
                log.append(f"pruned {len(dormant)} dormant status/feeling memories")
        return log

    # ------------------------------------------------------------------ inspection
    def memories(self, kind: str | None = None, include_retired: bool = False,
                 today: dt.date | None = None) -> list[dict]:
        """Current memories (not retired, not expired), or all of them with include_retired."""
        return self._q(
            """
            MATCH (m:Memory {user: $user})
            WHERE ($kind IS NULL OR m.kind = $kind)
              AND ($asof IS NULL OR m.created_at <= $asof)  // as of `today`, if one is given
              AND ($all OR (m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now)))
            OPTIONAL MATCH (m)-[:ABOUT]->(e)
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.importance AS importance,
                   m.status AS status, m.invalidated_at IS NOT NULL AS retired,
                   m.valid_until AS valid_until, m.valid_until <= $now AS expired,
                   coalesce(m.sensitive, false) AS sensitive, m.annual_date AS annual_date,
                   m.created_at AS created_at, collect(e.name) AS about
            ORDER BY created_at
            """,
            kind=kind, all=include_retired, now=clock.now_for(today or dt.date.today()),
            asof=clock.now_for(today) if today else None,
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
        """User-requested forgetting: hard delete (privacy beats provenance here).

        The turns that produced it are still in this episode's transcript, and the
        end-of-episode pass would extract it again. So a tombstone keeps only its embedding
        (no text), scoped to that episode, and apply() skips anything that matches it."""
        return bool(self._q(
            """
            MATCH (m:Memory {id: $id, user: $user})
            OPTIONAL MATCH (m)-[:FROM]->(ep:Episode)
            CREATE (t:Forgotten {user: $user, episode: coalesce(ep.id, ''), at: datetime()})
            WITH m, m.id AS id, t CALL db.create.setNodeVectorProperty(t, 'embedding', m.embedding)
            DETACH DELETE m RETURN id
            """,
            id=memory_id,
        ))

    def _tombstones(self, episode_id: str | None) -> list:
        if not episode_id:
            return []
        rows = self._q("MATCH (t:Forgotten {user: $user, episode: $ep}) RETURN t.embedding AS v", ep=episode_id)
        return [np.asarray(r["v"]) for r in rows]

    def forget_entity(self, name: str, known: dict[str, str] | None = None) -> dict | None:
        """"Forget everything about Jordan": delete the entity and every memory about or naming
        it, and redact the name from stored turns and episode summaries."""
        eid = self._lookup(name, known or {})
        if not eid or eid == self.user:
            return None
        ent = self._q("MATCH (e:Entity {id: $id}) RETURN e.name AS name, coalesce(e.aliases, []) AS aliases", id=eid)[0]
        terms = sorted({t for t in [ent["name"], *ent["aliases"]] if len(t) > 1}, key=len, reverse=True)
        # Whole words only, the same rule in Python and in Cypher: forgetting "Ann" must not
        # touch "anniversary". (?i) = ignore case; \Q...\E quotes the name for Java regex.
        pattern = re.compile(r"\b(" + "|".join(map(re.escape, terms)) + r")\b", re.I)
        cypher_re = "(?is).*\\b(" + "|".join(f"\\Q{t}\\E" for t in terms) + ")\\b.*"
        memories = self._q(
            """
            MATCH (m:Memory {user: $user})
            WHERE EXISTS { (m)-[:ABOUT]->(:Entity {id: $id}) } OR m.text =~ $re
            DETACH DELETE m RETURN count(*) AS n
            """,
            id=eid, re=cypher_re,
        )[0]["n"]
        turns = self._q(
            """
            MATCH (:Episode {user: $user})-[:HAS_TURN]->(t:Turn) WHERE t.text =~ $re
            RETURN elementId(t) AS id, t.text AS text
            """,
            re=cypher_re,
        )
        for t in turns:
            self._q("MATCH (t) WHERE elementId(t) = $id SET t.text = $text",
                    id=t["id"], text=pattern.sub("[forgotten]", t["text"]))
        episodes = self._q(
            """
            MATCH (ep:Episode {user: $user}) WHERE coalesce(ep.summary, '') =~ $re
            RETURN ep.id AS id, ep.summary AS summary
            """,
            re=cypher_re,
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
        # Other entities that mention them ("Kai: Jo's brother"), and examples on proposed types.
        for other in self._q(
            """
            MATCH (e:Entity {user: $user}) WHERE e.id <> $id
              AND (coalesce(e.summary, '') =~ $re OR any(a IN coalesce(e.aliases, []) WHERE a =~ $re))
            RETURN e.id AS id, e.name AS name, e.kind AS kind, coalesce(e.summary, '') AS summary,
                   coalesce(e.aliases, []) AS aliases
            """,
            id=eid, re=cypher_re,
        ):
            summary = pattern.sub("[forgotten]", other["summary"])
            self._q(
                """
                MATCH (e:Entity {id: $id}) SET e.summary = $summary, e.aliases = $aliases
                WITH e CALL db.create.setNodeVectorProperty(e, 'embedding', $vec)
                """,
                id=other["id"], summary=summary, aliases=[a for a in other["aliases"] if not pattern.search(a)],
                vec=embed([f"{other['name']} ({other['kind']}): {summary}"])[0],
            )
        self._q("MATCH (t:RelType) WHERE any(x IN coalesce(t.examples, []) WHERE x =~ $re) "
                "SET t.examples = [x IN t.examples WHERE NOT x =~ $re]", re=cypher_re)
        self._q("MATCH (e:Entity {id: $id}) DETACH DELETE e", id=eid)
        return {"name": ent["name"], "terms": terms, "memories": memories, "turns": len(turns), "episodes": len(episodes)}

    def wipe(self) -> None:
        """Delete everything stored for this user (used by the demo to start clean)."""
        self._q(
            """
            MATCH (n) WHERE (n:Memory OR n:Entity OR n:Episode OR n:Forgotten) AND n.user = $user
            OPTIONAL MATCH (n)-[:HAS_TURN]->(t:Turn)
            DETACH DELETE n, t
            """
        )
        self.__init__(self.driver, self.user_name)


SENSITIVE_NOTE = " (sensitive: only discuss if they raise it)"
STALE_NOTE = " (not confirmed in over a year: may be out of date, check before relying on it)"


def format_context(r: dict, today: dt.date | None = None) -> str:
    """Render a recall() result as the memory block placed in the system prompt."""
    today = today or dt.date.today()
    out: list[str] = []
    if r["core"]:
        out.append("Core profile:\n" + "\n".join(f"- {m['text']}" + (STALE_NOTE if m.get("stale") else "")
                                                for m in r["core"]))
    if r["people"]:
        out.append("People and things in their life:\n" + "\n".join(
            f"- {p['name']} ({p['relation']})" + (f" — {p['summary']}" if p.get("summary") else "")
            for p in r["people"]))
    if r["followups"]:
        out.append("Open plans to follow up on (today is " + today.isoformat() + "):\n" + "\n".join(
            f"- {m['text']} (due {m['due_on']})" + (SENSITIVE_NOTE if m.get("sensitive") else "")
            for m in r["followups"]))
    if r.get("upcoming"):
        out.append("Yearly dates coming up:\n" + "\n".join(
            f"- {m['on'].isoformat()}: {m['text']}" + (SENSITIVE_NOTE if m.get("sensitive") else "")
            for m in r["upcoming"]))
    seen = {m["text"] for m in r["core"] + r["followups"]}
    relevant = [m for m in r["semantic"] if m["text"] not in seen]
    for e in r["entities"]:
        relevant += [m for m in e["memories"] if m["text"] not in seen | {x["text"] for x in relevant}]
    if relevant:
        out.append("Relevant memories:\n" + "\n".join(
            f"- [{m['kind']}] {m['text']}" + (SENSITIVE_NOTE if m.get("sensitive") else "")
            + (STALE_NOTE if m.get("stale") else "") for m in relevant))
    if r.get("recent"):
        out.append("Recently, no longer true (you could ask how it went):\n" + "\n".join(
            f"- until {m['until'].to_native():%Y-%m-%d %H:%M}: {m['text']}"
            + (SENSITIVE_NOTE if m.get("sensitive") else "") for m in r["recent"]))
    if r.get("history"):
        out.append("No longer true (history, not current):\n" + "\n".join(
            f"- until {h['until']}: {h['text']}" + (f" → now: {h['replaced_by']}" if h["replaced_by"] else "")
            for h in r["history"]))
    if r["episodes"]:
        out.append("Recent conversations:\n" + "\n".join(
            f"- {str(e['at'])[:10]} (mood: {e['mood']}): {e['summary']}" for e in r["episodes"]))
    return "\n\n".join(out) or "(No memories yet — this is a new relationship.)"
