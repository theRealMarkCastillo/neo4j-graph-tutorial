"""Lessons 9-10 — MemoryStore: writes memories into the graph and recalls them for a turn.

Write path (Lesson 9):  resolve entities -> link relations -> dedupe -> supersede -> store
Read path  (Lesson 10): core profile + semantic (vector) + entity hop (graph) + follow-ups
                        + recent episodes, ranked by similarity, importance and recency
"""
from __future__ import annotations

import datetime as dt
import re
import threading
import uuid
from contextlib import contextmanager
from functools import partial

import numpy as np

from config import COMPANION_NAME, RERANK_MODEL
from observability import traced
from companion import clock, decay, ontology
from companion.embeddings import PROFILE, embed  # Lesson 19: local or remote, with its thresholds
from companion.models import EntityOut, Extraction, MemoryOut, RelationOut
from companion.schema import ensure_schema

_reranker = None
_ontology_loaded = False


def relevance(query: str, texts: list[str]) -> list[float]:
    """How much each text is ABOUT the query, from a cross-encoder (Lesson 18). Unlike an
    embedding, it reads the two together, so "dogs" vs "Mark's son is sick" scores near the
    floor (about -11) while "dogs" vs "Alex has a beagle" scores -5. Logits, not 0-1."""
    global _reranker
    if not texts:
        return []
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _reranker = TextCrossEncoder(RERANK_MODEL)
    return [float(s) for s in _reranker.rerank(query, texts)]


def _name_similarity(a: list[str], b: list[str]) -> float:
    """Best cosine similarity between any name/alias in `a` and any in `b`."""
    va, vb = np.array(embed(a)), np.array(embed(b))
    va /= np.linalg.norm(va, axis=1, keepdims=True)
    vb /= np.linalg.norm(vb, axis=1, keepdims=True)
    return float((va @ vb.T).max())


def _similar_names(name: str, known: list[str]) -> bool:
    """Another way to say a known name: "Bob" ~ "Robert", "Robert" in "Robert Smith"."""
    words = lambda n: set(re.findall(r"\w+", n.lower()))
    if any(words(name) <= words(k) or words(k) <= words(name) for k in known):
        return True
    return _name_similarity([name], known) >= NAME_MATCH_MIN


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


def _to_date(value: str | None) -> dt.date | None:
    try:
        return dt.date.fromisoformat(value) if value else None
    except ValueError:  # "2019" or "next spring" stay in the memory text instead
        return None


# Similarity thresholds belong to the embedding model (Lesson 19): companion/embeddings.py has
# one set per model. The numbers and measurements in these comments are bge-small's.
# Duplicate threshold on Neo4j's 0-1 rescaled cosine (0.975 ≈ raw cosine 0.95).
DUPLICATE_SCORE = PROFILE.duplicate
# A memory this close to one the user deleted in the same episode is not stored again.
FORGOTTEN_SCORE = PROFILE.forgotten
# Expired situational memories stay visible as "Recently ..." for this long.
RECENT_HOURS = 24
# Consolidation (Lesson 13)
LAPSE_AFTER_DAYS = 14        # an open plan this far past due stops being followed up
PATTERN_WINDOW_DAYS = 60     # look for repeats among memories this recent
PATTERN_MIN = 3              # memories ABOUT one activity on this many distinct days make a habit
PRUNE_MAX_IMPORTANCE = 3     # only trivia is ever deleted
# Entity match threshold when names differ ("Mom" vs "my mother").
ENTITY_MATCH_SCORE = PROFILE.entity_match
# ...and the NAMES must also be similar (raw cosine of name-only embeddings). Without this,
# "Seattle: the city where Riley now lives" matched "Boston: the city where Riley lives" —
# the summaries dominate the embedding. Mom/mother 0.87, NYC/New York 0.93; Seattle/Boston 0.64.
NAME_MATCH_MIN = PROFILE.name_match
# Sensitive memories (health, finances...) are recalled only when the message is clearly about
# them: (near) the best vector match for this message, not just above a fixed score. Small
# embedding models squeeze all scores into a narrow band (~0.63-0.80 here), so relative wins.
# Vector search always returns its nearest memories, related or not: with bge-small, "dogs"
# scored 0.74 against "Mark's son is sick" and "what do you remember about my kid?" 0.74
# against the right memory, so no vector cut-off works. A cross-encoder separates topical
# matches (unrelated messages <= -9.7, direct questions >= -7.6) but NOT inferential ones:
# "my meds don't help" vs the ADHD memory scored -10.9, like noise. So weak matches are
# never dropped: the chat model can make the leap. They're labelled in the web UI's trace only
# (in the prompt the label misled the model too: see format_context).
RELEVANCE_MIN = -8.5
SENSITIVE_MARGIN = PROFILE.sensitive_margin
SENSITIVE_FLOOR = PROFILE.sensitive_floor
# An entity this close to the message (index score) is looked up by the graph hop.
ENTITY_HOP_SCORE = PROFILE.entity_hop

# Keyword search (D11). Embeddings blur rare exact words (a medication name, a product code),
# so a full-text (BM25) search runs beside the vector search. Its hits join the candidates,
# and the two rankings are fused with reciprocal-rank fusion: each list votes 1 / (RRF_K +
# position). Only ranks are combined, never BM25 and cosine scores, which aren't comparable.
# 60 is the constant from the RRF paper (Cormack et al., 2009) and the usual default.
KEYWORD_LIMIT = 20
RRF_K = 60


def keyword_query(text: str, user: str) -> str | None:
    """A Lucene query for the message's words, limited to this user's memories. Lowercasing
    every word means AND / OR / NOT can't act as operators, and \\w+ drops the characters
    Lucene treats as syntax. None when the message has no words ("?!").

    The user filter is a phrase over analyzed tokens, so it's a pre-filter, not an exact
    match: "user:mark" also matches "user:mark-2" (the hyphen splits the token). The query
    re-checks m.user = $user exactly; the cost is that such a neighbour can use up some of
    the KEYWORD_LIMIT hits."""
    words = " ".join(w.lower() for w in re.findall(r"\w+", text))
    if not words:
        return None
    return f'text:({words}) AND user:"{user.replace(chr(92), "").replace(chr(34), "")}"'


def rrf(*rankings: list[str], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal-rank fusion: an id's score is the sum of 1 / (k + position) over the lists
    it appears in (positions start at 1). Being near the top of both lists beats being first
    in only one of them."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for pos, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1 / (k + pos)
    return fused

# A memory about someone a boundary protects ("don't bring up Dana") is never recalled.
# The user node is excluded: a topic boundary ("don't ask about my job") is ABOUT the user,
# and must not hide everything else about them (found by tests/test_10_recall.py).
NOT_PROTECTED = """NOT EXISTS {
    MATCH (m)-[:ABOUT]->(protected:Entity)<-[:ABOUT]-(b:Memory {kind: 'boundary'})
    WHERE b.invalidated_at IS NULL AND (b.valid_until IS NULL OR b.valid_until > $now) AND b <> m AND NOT protected:User }"""
# (Its variable names must not clash with the query it's pasted into: an outer `e` would
# silently turn "any entity" into "this entity" inside the EXISTS.)


def visible(v: str) -> str:
    """Lesson 16: may the current persona ($agent) see this memory or relation? Shared ones,
    its own private ones, and old ones from before personas existed (no agent)."""
    return f"({v}.agent IS NULL OR {v}.agent = $agent OR NOT coalesce({v}.private, false))"


def entity_visible(e: str) -> str:
    """An entity only another persona knows about, privately, is hidden too: its name and
    summary would give the private memory away ("Dr. Lee, the user's therapist")."""
    return (f"(EXISTS {{ MATCH ({e})<-[:ABOUT]-(vm:Memory) WHERE {visible('vm')} }}"
            f" OR EXISTS {{ MATCH ()-[vr:RELATED]-({e}) WHERE {visible('vr')} }}"
            f" OR NOT EXISTS {{ MATCH ({e})<-[:ABOUT|RELATED]-() }})")

def one_person_roles() -> set[str]:
    """Relations to exactly one person at a time: mother, father, and the "one at a time"
    person types in the ontology (partner, spouse)."""
    return {"mother", "father"} | {t for t, spec in ontology.ONTOLOGY.items()
                                   if spec.get("cardinality") == "one" and "person" in (spec.get("range") or [])}


def is_role_name(name: str) -> bool:
    """A placeholder for someone whose name isn't known yet: "Mom", "my boss", "Mark's son".
    The rest, after any "my " or "X's ", must be a relation type or synonym in the ontology."""
    rest = re.sub(r"^(my|the|his|her|their)\s+|^.+?['’]s\s+", "", name.strip(), flags=re.I)
    return ontology.normalize(rest)[1] in ("exact", "synonym")


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


def temporary_relation(ex: Extraction, rel: RelationOut, canon: str, now: dt.datetime) -> bool:
    """Shared expiry-based trip guard, evaluated on extraction references as in apply."""
    temporary = {n.lower() for m in ex.memories
                 if clock.parse_until(m.valid_until, now, m.kind) is not None for n in m.about}
    lasting = {n.lower() for m in ex.memories
               if clock.parse_until(m.valid_until, now, m.kind) is None for n in m.about}
    return (not rel.ended and ontology.rule(canon, "cardinality") == "one"
            and rel.target.lower() in temporary and rel.target.lower() not in lasting)


class MemoryStore:
    def __init__(self, driver, user_name: str, agent: str | None = None, *, read_only: bool = False):
        self.driver = driver
        self.user_name = user_name
        self.user = f"user:{user_name.lower()}"
        # Which persona this store reads and writes for (Lesson 16). Several personas can share
        # one user's memory; each sees shared memories plus its own private ones.
        self.agent = agent or COMPANION_NAME
        # The write transaction this thread is inside, if any (Lesson 14). Thread-local: with
        # extraction in a background thread, the chat thread's recall must not run its
        # queries inside the worker's open transaction.
        self._local = threading.local()
        self.entity_log: list[str] = []  # renames, merges, kind changes (see resolve_entity)
        self._read_only = read_only
        if read_only:
            return  # no schema, ontology, user, or embedding writes in a preview
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
            WITH u CALL db.create.setNodeVectorProperty(u, $vprop, $vec)
            """,
            name=user_name, vec=embed([f"{user_name}, the user"])[0],
        )

    @classmethod
    @contextmanager
    def read_view(cls, driver, user_name: str, agent: str | None = None):
        """A query-guarded view; routing alone does not prevent writes on every server."""
        view = cls(driver, user_name, agent, read_only=True)
        with driver.session(default_access_mode="READ") as session:
            with session.begin_transaction() as tx:
                view._local.tx = tx
                try:
                    yield view
                finally:
                    view._local.tx = None
                    if not tx.closed():
                        tx.rollback()

    def _q(self, query: str, now: dt.datetime | None = None, **params) -> list[dict]:
        """Run a query with $user and $now always set ($now: the conversation's clock).

        Inside `_write()` it joins that transaction; otherwise it's its own auto-committed one."""
        params = {"user": self.user, "agent": self.agent, "now": now or clock.now_for(dt.date.today()),
                  # the active embedding model's vector property and indexes (Lesson 19)
                  "vprop": PROFILE.prop, "memory_vec": PROFILE.index("memory_vec"),
                  "entity_vec": PROFILE.index("entity_vec"), **params}
        tx = getattr(self._local, "tx", None)
        if self._read_only:
            if tx is None:
                raise PermissionError("read-only queries require an open read_view")
            # EXPLAIN plans but never executes a statement, including procedure calls.
            if tx.run("EXPLAIN " + query, **params).consume().query_type != "r":
                raise PermissionError("read-only view rejected a write query")
        if tx is not None:
            return tx.run(query, **params).data()
        records, _, _ = self.driver.execute_query(query, **params)
        return [r.data() for r in records]

    def _write(self, work):
        """Run `work()` as ONE write transaction for this user (Lesson 14).

        - All or nothing: if anything fails half-way, every write rolls back.
        - One writer per user: the first statement writes to the user's node, which takes
          Neo4j's write lock on it until commit. A second writer for the same user (another
          thread or process) waits there, then sees the first one's committed memories, so
          its duplicate check works. Different users never wait for each other.
        - execute_write retries transient errors (deadlocks, leader changes) by running
          `work` again from the start, so `work` must not have side effects outside the graph.
        Nested calls join the outer transaction."""
        if self._read_only:
            raise PermissionError("read-only view cannot apply changes")
        if getattr(self._local, "tx", None) is not None:
            return work()

        def unit(tx):
            self._local.tx = tx
            try:
                tx.run("MATCH (u:User {id: $user}) SET u.write_lock = datetime()", user=self.user).consume()
                return work()
            finally:
                self._local.tx = None

        with self.driver.session() as session:
            return session.execute_write(unit)

    # ------------------------------------------------------------------ episodes
    def start_episode(self, on: dt.date | None = None) -> str:
        """Open a conversation episode. `on` backdates it (used by the scripted demo)."""
        episode_id = _new_id("ep")
        self._q(
            """
            OPTIONAL MATCH (prev:Episode {user: $user}) WHERE prev.ended_at IS NOT NULL
              AND (prev.agent IS NULL OR prev.agent = $agent)  // each persona has its own diary
            WITH prev ORDER BY prev.started_at DESC LIMIT 1
            CREATE (ep:Episode {id: $id, user: $user, agent: $agent,
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
            WITH ep CALL db.create.setNodeVectorProperty(ep, $vprop, $vec)
            """,
            ep=episode_id, summary=summary, mood=mood, vec=embed([summary])[0],
        )

    # ------------------------------------------------------------------ entities
    def _is_user(self, name: str) -> bool:
        return name.strip().lower() in {"user", "me", "i", self.user, self.user_name.lower()}

    def _find_entity(self, name: str, kind: str, aliases=(), summary: str = "",
                     ref: str | None = None, new: bool = False) -> tuple[str | None, np.ndarray]:
        """Read-only selection used by both writes and current-graph previews."""
        vec = embed([f"{name} ({kind}): {summary}"])[0]
        hit = self._q("MATCH (e:Entity {id: $id, user: $user}) RETURN e.id AS id", id=ref) if ref else []
        names = [n.lower() for n in [name, *aliases]]
        if not hit and not new:
            hit = self._q(
                """
                MATCH (e:Entity {user: $user}) WHERE e.kind <> 'self'
                  AND (toLower(e.name) IN $names OR any(a IN coalesce(e.aliases, []) WHERE toLower(a) IN $names))
                // a real name before a placeholder that merely lists it
                RETURN e.id AS id ORDER BY CASE WHEN toLower(e.name) = $name THEN 0 ELSE 1 END,
                       vector.similarity.cosine(e[$vprop], $vec) DESC LIMIT 1
                """,
                names=names, name=name.lower(), vec=vec,
            )
        if not hit and not new:
            candidates = self._q(
                """
                CALL db.index.vector.queryNodes($entity_vec, 50, $vec) YIELD node, score
                WHERE node.user = $user AND node.kind IN $kinds AND score >= $min
                RETURN node.id AS id, [node.name] + coalesce(node.aliases, []) AS names ORDER BY score DESC
                """,
                vec=vec, kinds=_compatible_kinds(kind), min=ENTITY_MATCH_SCORE,
            )
            hit = [c for c in candidates if _name_similarity([name, *aliases], c["names"]) >= NAME_MATCH_MIN][:1]
        return (hit[0]["id"] if hit else None), vec

    def _entity_changes(self, cur: dict, incoming: EntityOut) -> dict:
        name, kind, summary = incoming.name, incoming.kind, incoming.summary
        aliases, ref, corrected = incoming.aliases, incoming.ref, incoming.corrected
        known = [cur["name"], *cur["aliases"]]
        differs = name.lower() not in {k.lower() for k in known}
        rename = name.lower() != cur["name"].lower() and not is_role_name(name) and (
            is_role_name(cur["name"]) or (corrected and bool(ref)))
        keep_name = rename or not differs or is_role_name(name) or _similar_names(name, known)
        final = name if rename else cur["name"]
        all_aliases = list(dict.fromkeys([*cur["aliases"], *aliases, cur["name"], *([name] if keep_name else [])]))
        return {"name": final, "kind": kind if cur["kind"] == "thing" and kind != "thing" else cur["kind"],
                "summary": summary or cur["summary"],
                "aliases": [a for a in all_aliases if a.lower() != final.lower()],
                "rename": rename, "differs": differs, "keep_name": keep_name}

    def _role_merges(self, entity_id: str, aliases) -> list[dict]:
        return [d for d in self._q(
            "MATCH (e:Entity {user: $user}) WHERE e.id <> $id AND NOT e:User AND toLower(e.name) IN $names "
            "RETURN e.id AS id, e.name AS name", id=entity_id, names=[a.lower() for a in aliases]
        ) if is_role_name(d["name"])]

    def resolve_entity(self, name: str, kind: str = "thing", aliases=(), summary: str = "",
                       ref: str | None = None, new: bool = False, corrected: bool = False) -> str:
        """Find-or-create an entity.

        1. `ref` — the extractor pointed at a known entity id
        2. `new` — the extractor says it's a different entity with a known name (a second Sam)
        3. exact name/alias; if several share the name, the one whose summary is most similar
        4. embedding similarity (same or compatible kind) above ENTITY_MATCH_SCORE
        5. otherwise create it

        Then, for a known entity (each change is logged in self.entity_log):
        - the name: a real name replaces a role ("Mark's son" -> "Logan"), and `corrected`
          replaces a wrong one ("Maya" -> "Maia"); the old name stays as an alias
        - another name: kept as an alias only if it's a role ("Dad") or close to a known name
          ("Bob" ~ "Robert"). A different name ("Jordan" for Sam) is more likely a wrong `ref`
          than a nickname, and an alias would send every later "Jordan" to Sam.
        - the kind: "thing" upgrades to anything more specific, never the other way
        - a role-named duplicate ("Pytest's sister") listed in `aliases` is merged into it
        """
        if self._is_user(name):
            return self.user
        entity_id, vec = self._find_entity(name, kind, aliases, summary, ref, new)
        if not entity_id:  # a new entity: nothing to reconcile
            entity_id = _new_id("e")
            self._q(
                """
                CREATE (e:Entity {id: $id, user: $user, name: $name, kind: $kind, summary: $summary,
                                  aliases: [a IN apoc.coll.toSet($aliases) WHERE toLower(a) <> toLower($name)],
                                  created_at: datetime(), updated_at: datetime()})
                WITH e CALL db.create.setNodeVectorProperty(e, $vprop, $vec)
                """,
                id=entity_id, name=name, kind=kind, summary=summary, aliases=list(aliases), vec=vec,
            )
            return entity_id

        cur = self._q("MATCH (e:Entity {id: $id}) RETURN e.name AS name, e.kind AS kind, "
                      "coalesce(e.aliases, []) AS aliases, coalesce(e.summary, '') AS summary", id=entity_id)[0]
        changes = self._entity_changes(cur, EntityOut.model_construct(name=name, kind=kind, summary=summary,
                                                     aliases=list(aliases), ref=ref, corrected=corrected))
        rename, differs, keep_name = changes["rename"], changes["differs"], changes["keep_name"]
        if rename:
            how = "name corrected" if corrected and not is_role_name(cur["name"]) else f"{cur['name']} kept as an alias"
            self.entity_log.append(f"named {cur['name']} -> {name} ({how})")
        elif differs and not keep_name:
            self.entity_log.append(f"kept {cur['name']}: '{name}' matched it by id but isn't one of its names "
                                   "(not added as an alias)")
        new_kind = changes["kind"]
        if new_kind != cur["kind"]:
            self.entity_log.append(f"{cur['name'] if not rename else name} is a {new_kind} (was {cur['kind']})")
        final, final_summary = changes["name"], changes["summary"]
        self._q(
            """
            MATCH (e:Entity {id: $id})
            SET e.name = $final, e.kind = $kind, e.summary = $summary, e.updated_at = datetime(),
                e.aliases = $aliases
            WITH e CALL db.create.setNodeVectorProperty(e, $vprop, $vec)
            """,
            id=entity_id, final=final, kind=new_kind, summary=final_summary,
            aliases=changes["aliases"],
            vec=embed([f"{final} ({new_kind}): {final_summary}"])[0],
        )
        # A placeholder the extractor now says is this entity ("Maya", aliases ["Pytest's sister"]):
        # one person, so one node. Only role-named ones: two real "Sam"s may be two people.
        for dup in self._role_merges(entity_id, aliases):
            self.merge_entities(entity_id, dup["id"])
            self.entity_log.append(f"merged {dup['name']} into {final} (the same {kind})")
        return entity_id

    def merge_entities(self, keep: str, drop: str) -> None:
        """Move everything from `drop` onto `keep`, then delete `drop`: relations both ways
        (an existing edge of the same type absorbs the other, current wins over ended),
        memories ABOUT it, and its names as aliases."""
        self._q(
            """
            MATCH (keep:Entity {id: $keep, user: $user}), (drop:Entity {id: $drop, user: $user})
            CALL (keep, drop) {
              MATCH (drop)-[r:RELATED]->(x) WHERE x <> keep
              MERGE (keep)-[n:RELATED {type: r.type}]->(x)
              ON CREATE SET n = properties(r)
              ON MATCH SET n.until = CASE WHEN r.until IS NULL OR n.until IS NULL THEN null ELSE n.until END
            }
            CALL (keep, drop) {
              MATCH (x)-[r:RELATED]->(drop) WHERE x <> keep
              MERGE (x)-[n:RELATED {type: r.type}]->(keep)
              ON CREATE SET n = properties(r)
              ON MATCH SET n.until = CASE WHEN r.until IS NULL OR n.until IS NULL THEN null ELSE n.until END
            }
            CALL (keep, drop) {
              MATCH (m:Memory)-[:ABOUT]->(drop) MERGE (m)-[:ABOUT]->(keep)
            }
            SET keep.aliases = [a IN apoc.coll.toSet(coalesce(keep.aliases, []) + [drop.name] + coalesce(drop.aliases, []))
                                WHERE toLower(a) <> toLower(keep.name)],
                keep.summary = CASE WHEN coalesce(keep.summary, '') = '' THEN drop.summary ELSE keep.summary END
            DETACH DELETE drop
            """,
            keep=keep, drop=drop,
        )

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
    @staticmethod
    def _private(mem) -> bool:
        """Private to the persona that heard it: asked ("just between us"), or sensitive.
        Boundaries are never private: a boundary set with one persona protects with all."""
        return mem.kind != "boundary" and (mem.private or mem.sensitive)

    def _memory_decision(self, mem: MemoryOut, vec, about_ids: set[str], episode_id: str | None,
                         now: dt.datetime, stored: list, tombstones: list) -> dict:
        """Classify without writes; retain the write path's scores, filters and reasons."""
        gone = max((float(np.dot(vec, t)) for t, ids in tombstones if ids <= about_ids), default=float("-inf"))
        if gone >= FORGOTTEN_SCORE:
            return {"action": "skip", "reason": "skipped a memory the user asked to forget earlier in this conversation "
                    f"(cosine {gone:.3f} ≥ {FORGOTTEN_SCORE})"}
        repeat = max((float(np.dot(vec, v)) for v, ids in stored if ids <= about_ids), default=float("-inf"))
        if repeat >= DUPLICATE_SCORE:
            return {"action": "skip", "reason": f"skipped a repeat within this extraction: {mem.text} "
                    f"(cosine {repeat:.3f} ≥ {DUPLICATE_SCORE})"}
        nearest = self._q(
            """
            CALL db.index.vector.queryNodes($memory_vec, 50, $vec) YIELD node, score
            WHERE node.user = $user AND node.invalidated_at IS NULL
              AND (node.valid_until IS NULL OR node.valid_until > $now)  // expired: say it again = new
              AND """ + visible("node") + """
              // same dates too: "dentist on 2026-11-12" -> "...on 2026-11-17" reads almost the
              // same but is a real correction; an identical "replacement" is just a duplicate
              AND coalesce(node.due_on, date('1900-01-01')) = coalesce($due, date('1900-01-01'))
              AND coalesce(node.occurred_on, date('1900-01-01')) = coalesce($occurred, date('1900-01-01'))
              AND all(x IN [(node)-[:ABOUT]->(e) WHERE NOT e:User | e.id] WHERE x IN $about)
            RETURN node.id AS id, node.text AS text, score ORDER BY score DESC LIMIT 1
            """,
            vec=vec, about=sorted(about_ids), due=_to_date(mem.due_on), occurred=_to_date(mem.occurred_on), now=now,
        )
        if nearest and nearest[0]["score"] >= DUPLICATE_SCORE:
            return {"action": "reinforce", "reason": "matching current memory", **nearest[0]}
        why = (f"  [new: closest known {nearest[0]['score']:.3f} < {DUPLICATE_SCORE}]" if nearest
               else "  [new: nothing similar with the same dates and entities]")
        return {"action": "new", "reason": why}

    @traced("store.apply", "CHAIN")
    def apply(self, ex: Extraction, episode_id: str | None = None, today: dt.date | None = None) -> list[str]:
        """Store one extraction, all or nothing (one transaction). Returns a log of what changed."""
        return self._write(lambda: self._apply(ex, episode_id, today))

    def _apply(self, ex: Extraction, episode_id: str | None, today: dt.date | None) -> list[str]:
        today = today or dt.date.today()
        now = clock.now_for(today)
        q = partial(self._q, now=now)
        log: list[str] = []
        known: dict[str, str] = {}
        self.forgotten_terms: list[str] = []  # names forgotten by this call (the graph redacts its thread)
        self.written_ids: list[str] = []      # memories created or reinforced by this call
        self.entity_log = []
        for ent in ex.entities:
            eid = self.resolve_entity(ent.name, ent.kind, ent.aliases, ent.summary, ent.ref, ent.new,
                                      ent.corrected)
            for n in [ent.name, *ent.aliases, *([ent.ref] if ent.ref else [])]:
                if known.get(n.lower(), eid) != eid:  # two different Sams in one extraction
                    log.append(f"warning: '{n}' names two entities here; references to it go to the last one")
                known[n.lower()] = eid
            if ent.new and not ent.ref:
                log.append(f"new entity {ent.name} ({eid}), distinct from the known {ent.name}")
        log += self.entity_log

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
        # temporary memory mentions is a trip, not a move. Applying it would end the real home.
        kept = []
        for item in resolved:
            rel, canon = item[0], item[3]
            if temporary_relation(ex, rel, canon, now):
                log.append(f"ignored {rel.source} -{canon}-> {rel.target}: only a temporary memory "
                           "mentions it (a trip, not a move)")
                continue
            kept.append(item)
        resolved = kept
        # Lesson 16: a relation to someone only PRIVATE memories mention is private too,
        # or another persona's people list would give it away.
        private_names = {n.lower() for m in ex.memories if self._private(m) for n in m.about}
        shared_names = {n.lower() for m in ex.memories if not self._private(m) for n in m.about}
        linking = {(a, b, canon) for rel, a, b, canon, _ in resolved if not rel.ended}
        linked: list[tuple[str, str, str]] = []  # edges this extraction (re)asserts: evidence below
        for rel, a, b, canon, how in resolved:
            if rel.ended and (a, b, canon) in linking:
                continue  # "ended X" + "linked X" in one go: keep it linked
            info = q("MATCH (a:Entity {id: $a}), (b:Entity {id: $b}) RETURN a.name AS a, b.name AS b, b.kind AS kind",
                           a=a, b=b)[0]
            src, dst = ("user" if a == self.user else info["a"]), info["b"]
            said = f" (said '{rel.type}')" if how in ("synonym", "flipped") else ""
            if not rel.ended and a == self.user and canon in one_person_roles() and not is_role_name(dst):
                # "Mom" first, "my mother Linda" later, and the extractor didn't say they're the
                # same: someone has one mother (or partner), so a role-named one IS Linda.
                for y in q(
                    """
                    MATCH (:User {id: $user})-[r:RELATED]->(y:Entity) WHERE y.id <> $b AND r.until IS NULL
                      AND r.type IN $types AND NOT toLower(coalesce(r.said, '')) CONTAINS 'step'
                    RETURN y.id AS id, y.name AS name
                    """,
                    b=b, types=sorted(ontology.lineage(canon) | {canon}),
                ):
                    if is_role_name(y["name"]) and "step" not in rel.type.lower():
                        self.merge_entities(b, y["id"])
                        known = {k: (b if v == y["id"] else v) for k, v in known.items()}
                        log.append(f"merged {y['name']} into {dst} (one {canon})")
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
            for x in q(  # likes <-> dislikes: the new one ends the opposite edge to the same target
                """
                MATCH (a:Entity {id: $a})-[r:RELATED]->(b:Entity {id: $b})
                WHERE r.until IS NULL AND r.type IN $types
                SET r.until = $today RETURN r.type AS type
                """,
                a=a, b=b, types=ontology.rule(canon, "exclusive") or [], today=today,
            ):
                log.append(f"ended {src} -{x['type']}-> {dst} (now {canon})")
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
                    r.said = $said_as, r.agent = coalesce(r.agent, $agent),
                    r.private = $private AND coalesce(r.private, true)  // once shared, stays shared
                RETURN changed
                """,
                a=a, b=b, type=canon, since=rel.since, now=now, said_as=rel.type,
                private=rel.target.lower() in private_names and rel.target.lower() not in shared_names,
            )[0]["changed"]
            linked.append((a, b, canon))
            if changed:  # only log new or re-activated links
                log.append(f"linked {src} -{canon}-> {dst}{said}")

        evidence: list[tuple[str, set[str]]] = []  # (memory id, entity ids it is about)
        retired_pairs: list[tuple[str, str | None]] = []  # (retired memory, its replacement)
        for mid in ex.invalidate:
            if q(
                """
                MATCH (m:Memory {id: $id, user: $user}) WHERE m.invalidated_at IS NULL
                SET m.invalidated_at = $now RETURN m.text AS text
                """, id=mid, now=now,
            ):
                log.append(f"retired {mid}")
                retired_pairs.append((mid, None))
        # A plan the same extraction re-states (replaces it, or repeats it word for word with the
        # same due date) is still on: smaller models have "completed" a plan that was only moved.
        restated_ids = sorted({sid for m in ex.memories for sid in m.supersedes})
        restated = [{"due": d, "text": m.text.strip().lower()} for m in ex.memories
                    if m.kind == "plan" and (d := _to_date(m.due_on))]
        for mid in ex.completed_plans:
            [row] = q(
                """
                OPTIONAL MATCH (m:Memory {id: $id, user: $user, kind: 'plan', status: 'open'})
                WITH m, m IS NOT NULL AND (m.id IN $ids OR any(r IN $restated
                     WHERE r.due = m.due_on AND r.text = toLower(trim(m.text)))) AS restated
                FOREACH (_ IN CASE WHEN m IS NOT NULL AND NOT restated THEN [1] ELSE [] END |
                    SET m.status = 'done')
                RETURN m IS NOT NULL AS found, restated
                """, id=mid, ids=restated_ids, restated=restated,
            )
            if row["restated"]:
                log.append(f"kept plan {mid} open: the same extraction re-states it")
            elif row["found"]:
                log.append(f"plan done {mid}")

        vectors = embed([m.text for m in ex.memories]) if ex.memories else []
        tombstones = self._tombstones(episode_id)
        # Vectors created in THIS transaction: the vector index cannot see them yet.
        stored: list[tuple[np.ndarray, set[str]]] = []
        for mem, vec in zip(ex.memories, vectors):
            until = clock.parse_until(mem.valid_until, now, mem.kind)
            about_ids = {eid for n in mem.about if (eid := self._lookup(n, known)) and eid != self.user}
            decision = self._memory_decision(mem, vec, about_ids, episode_id, now, stored, tombstones)
            if decision["action"] == "skip":
                log.append(decision["reason"])
                continue
            dup = [decision] if decision["action"] == "reinforce" else []
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
                # ...and is about every entity it names now, not only those named before (a
                # re-read under new rules: "Logan loves Minecraft" gains Minecraft)
                q("MATCH (m:Memory {id: $id}) UNWIND $about AS eid MATCH (e:Entity {id: eid}) MERGE (m)-[:ABOUT]->(e)",
                  id=dup[0]["id"], about=[eid for n in mem.about if (eid := self._lookup(n, known))])
                log.append(f"reinforced {dup[0]['id']} \"{dup[0]['text']}\" "
                           f"(similarity {dup[0]['score']:.3f} ≥ {DUPLICATE_SCORE})")
                self.written_ids.append(dup[0]["id"])
                evidence.append((dup[0]["id"], {eid for n in (mem.about or ["user"]) if (eid := self._lookup(n, known))}))
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
                    retired_pairs.append((old["id"], dup[0]["id"]))
                continue

            mid = _new_id("m")
            stored.append((vec, about_ids))
            self.written_ids.append(mid)
            about = [eid for n in (mem.about or ["user"]) if (eid := self._lookup(n, known))]
            retired = q(
                """
                CREATE (m:Memory {id: $id, user: $user, text: $text, kind: $kind, importance: $imp,
                                  occurred_on: $occurred, due_on: $due, annual_date: $annual,
                                  sensitive: $sensitive, valid_until: $until,
                                  agent: $agent, private: $private,
                                  status: CASE WHEN $kind = 'plan' THEN 'open' ELSE null END,
                                  created_at: $now, access_count: 0})
                WITH m CALL db.create.setNodeVectorProperty(m, $vprop, $vec)
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
                private=self._private(mem),
                vec=vec, about=about, ep=episode_id or "", supersedes=mem.supersedes, now=now,
            )[0]["retired"]
            evidence.append((mid, set(about)))
            retired_pairs += [(old, mid) for old in retired]
            flags = (" (sensitive)" if mem.sensitive else "") + (f" (yearly {mem.annual_date})" if _annual(mem.annual_date) else "")
            if until:
                flags += f" (until {until:%Y-%m-%d %H:%M})"
            if self._private(mem):
                flags += f" (private to {self.agent})"
            why = decision["reason"]
            log.append(f"+ [{mem.kind}] {mem.text}{flags}" + (f"  (replaces {', '.join(retired)})" if retired else "") + why)

        # Provenance: each edge lists the memories that support it, those about BOTH its ends
        # ("Maya also lives in Boston" is no evidence for the user's own lives_in Boston). When
        # they are all retired with no heir, the edge ends with them.
        for a, b, canon in linked:
            ids = [mid for mid, ab in evidence if {a, b} <= ab]
            if ids:
                q("""
                  MATCH (:Entity {id: $a})-[r:RELATED {type: $type}]->(:Entity {id: $b})
                  SET r.evidence = apoc.coll.toSet(coalesce(r.evidence, []) + $ids)
                  """, a=a, b=b, type=canon, ids=ids)
        log += self._retire_links(retired_pairs, today, skip=linked)

        for name in ex.forget_entities:  # last, so memories made this turn about them go too
            gone = self.forget_entity(name, known)
            if gone:
                self.forgotten_terms += gone["terms"]
                log.append(f"forgot {gone['name']}: {gone['memories']} memories deleted, "
                           f"{gone['turns']} turns and {gone['episodes']} summaries redacted")
        return log

    def _retire_links(self, pairs: list[tuple[str, str | None]], today: dt.date,
                      skip=()) -> list[str]:
        """A memory left the graph (`old`, maybe replaced by `new`). Edges it supported pass the
        support on to `new` if `new` is about both their ends; an edge left with no support
        ends (kept as history). Edges from before provenance (no `evidence`) are never touched,
        nor edges this same extraction asserted (`skip`)."""
        log = []
        for old, new in pairs:
            for e in self._q(
                """
                MATCH (a:Entity)-[r:RELATED]->(b:Entity)
                WHERE r.until IS NULL AND $old IN coalesce(r.evidence, [])
                  AND NOT a.id + '|' + b.id + '|' + r.type IN $skip
                OPTIONAL MATCH (n:Memory {id: $new}) WHERE EXISTS { (n)-[:ABOUT]->(a) } AND EXISTS { (n)-[:ABOUT]->(b) }
                WITH a, r, b, n.id AS heir
                SET r.evidence = [i IN r.evidence WHERE i <> $old] + CASE WHEN heir IS NULL THEN [] ELSE [heir] END
                WITH a, r, b WHERE size(r.evidence) = 0
                SET r.until = $today
                RETURN CASE WHEN a:User THEN 'user' ELSE a.name END AS a, r.type AS type, b.name AS b
                """,
                old=old, new=new, today=today, skip=[f"{a}|{b}|{t}" for a, b, t in skip],
            ):
                log.append(f"ended {e['a']} -{e['type']}-> {e['b']} (its only memory was retired)")
        return log

    # ------------------------------------------------------------------ read path
    @traced("store.extraction_candidates", "RETRIEVER")
    def extraction_candidates(self, text: str, k: int = 10, episode_id: str | None = None,
                              today: dt.date | None = None) -> list[dict]:
        """Existing current memories the extractor may update or retire, with their ids:
        vector matches + memories about entities named in the text + open plans
        (+ everything already taken from this episode, for the end-of-episode pass)."""
        return self._q(
            """
            CALL db.index.vector.queryNodes($memory_vec, 50, $vec) YIELD node AS m, score
            WHERE m.user = $user AND m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now) AND coalesce(m.status, '') <> 'done'
              AND """ + visible("m") + """
            WITH m ORDER BY score DESC LIMIT $k  // filter first, then keep the top k
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.status AS status
            UNION
            MATCH (m:Memory)-[:ABOUT]->(e:Entity {user: $user}) WHERE e.kind <> 'self' AND m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now)
              AND coalesce(m.status, '') <> 'done' AND """ + visible("m") + """
              AND any(n IN [e.name] + coalesce(e.aliases, []) WHERE size(n) > 2 AND toLower($text) CONTAINS toLower(n))
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.status AS status
            UNION
            MATCH (m:Memory {user: $user, kind: 'plan', status: 'open'}) WHERE m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now)
              AND """ + visible("m") + """
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.status AS status
            UNION
            MATCH (m:Memory)-[:FROM]->(:Episode {id: $ep}) WHERE m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now)
              AND """ + visible("m") + """
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.status AS status
            """,
            k=k, vec=embed([text], query=True)[0], text=text, ep=episode_id or "",
            now=clock.now_for(today or dt.date.today()),
        )

    def known_entities(self) -> list[str]:
        """Entities shown to the extractor, with ids so it can tell two Sams apart."""
        rows = self._q(
            """
            MATCH (e:Entity {user: $user}) WHERE e.kind <> 'self' AND """ + entity_visible("e") + """
            OPTIONAL MATCH (:User {id: $user})-[r:RELATED]->(e) WHERE r.until IS NULL AND """ + visible("r") + """
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
              AND """ + visible("r") + """
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

    @traced("store.recall", "RETRIEVER")
    def recall(self, query: str, k: int = 8, today: dt.date | None = None, touch: bool = True) -> dict:
        """What the companion knows that's relevant to `query`. `touch=False` only looks
        (the web UI's preview): it doesn't count as an access."""
        today = today or dt.date.today()
        now = clock.now_for(today)
        q = partial(self._q, now=now)  # every query below sees the conversation's clock
        vec = embed([query], query=True)[0]  # a message searched against memories (Lesson 19)
        # Core profile: every boundary + the most important stable facts and preferences.
        core = q(
            """
            CALL () {
              MATCH (m:Memory {user: $user, kind: 'boundary'}) WHERE m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now)
              RETURN m  // boundaries are never private: every persona respects them
              UNION
              MATCH (m:Memory {user: $user}) WHERE m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now)
                AND m.kind IN ['fact', 'preference'] AND m.importance >= 6
                AND NOT coalesce(m.sensitive, false) AND """ + NOT_PROTECTED + """ AND """ + visible("m") + """
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
        # Semantic: vector similarity re-ranked by importance and recency (feelings fade fast),
        # plus keyword (full-text) hits fused in by rank (D11). A keyword hit the vector search
        # missed gets its cosine computed directly, so every candidate has a comparable score.
        semantic = q(
            """
            CALL () {
              CALL db.index.vector.queryNodes($memory_vec, 50, $vec) YIELD node AS m, score
              RETURN m, score, null AS bm25
              UNION ALL
              WITH 1 AS one WHERE $lucene IS NOT NULL  // no words in the message: no keyword search
              CALL db.index.fulltext.queryNodes('memory_text', $lucene, {limit: $klimit}) YIELD node AS m, score AS bm25
              WITH m, bm25 WHERE m[$vprop] IS NOT NULL  // not yet embedded for this model (Lesson 19)
              RETURN m, vector.similarity.cosine(m[$vprop], $vec) AS score, bm25
            }
            WITH m, max(score) AS score, max(bm25) AS bm25  // found by both: one row
            WHERE m.user = $user AND m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now) AND coalesce(m.status, '') <> 'done'
              AND (m.kind = 'boundary' OR """ + NOT_PROTECTED + """) AND """ + visible("m") + """
            WITH collect({m: m, score: score, bm25: bm25}) AS hits
            WITH hits, reduce(top = 0.0, h IN hits | CASE WHEN h.score > top THEN h.score ELSE top END) AS top
            UNWIND hits AS h
            WITH h.m AS m, h.score AS score, h.bm25 AS bm25, top
            WHERE NOT coalesce(m.sensitive, false)  // sensitive: only as (near) the best match
               OR (score >= top - $margin AND score >= $floor)
            // Age since it was stored or last *confirmed* by the user, on the conversation's clock.
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.occurred_on AS occurred_on, m.due_on AS due_on,
                   m.importance AS importance, coalesce(m.sensitive, false) AS sensitive,
                   round(score, 3) AS score, round(bm25, 3) AS bm25,
                   duration.inSeconds(coalesce(m.last_confirmed, m.created_at), $now).seconds / 86400.0 AS age
            """,
            vec=vec, margin=SENSITIVE_MARGIN, floor=SENSITIVE_FLOOR,
            lucene=keyword_query(query, self.user), klimit=KEYWORD_LIMIT,
        )
        # Decay (Lesson 13) in Python, where the curve is unit-tested: drop dormant memories,
        # then order by match + strength.
        dormant = [{"id": m["id"], "kind": m["kind"], "text": m["text"], "score": m["score"],
                    "strength": round(decay.strength(m["importance"], m["kind"], m["age"]), 3)}
                   for m in semantic if decay.is_dormant(m["importance"], m["kind"], m["age"])]
        semantic = [m for m in semantic if not decay.is_dormant(m["importance"], m["kind"], m["age"])]
        # Relevance (Lesson 18): mark what the vector search returned only because it had to
        # return something. Weak matches go after strong ones and are labelled in the prompt.
        # A current status ("at the gym" until 14:00) is never weak: it's context for any
        # message, like "Recently" below ("what am I doing?" scored -8.9 against it).
        # Lesson 19: a model whose scores DO separate (Qwen3-8B) decides with its own score,
        # and the cross-encoder isn't needed (it called "how is my kid doing?" vs "Mark's son
        # Logan attends Elk Grove Elementary" weak).
        if PROFILE.weak_below is not None:
            scores, cut = [m["score"] for m in semantic], PROFILE.weak_below
        else:
            scores, cut = relevance(query, [m["text"] for m in semantic]), RELEVANCE_MIN
        for m, r in zip(semantic, scores):
            m["relevance"] = round(r, 3 if PROFILE.weak_below is not None else 2)
            m["weak"] = r < cut and m["kind"] != "status"
        semantic = [{**m, "rank": round(decay.rank(m["score"], m["importance"], m["kind"], m["age"]), 3),
                     "stale": decay.is_stale(m["kind"], m["age"])} for m in semantic]
        # Fuse two rankings (D11): match x importance x decay, and BM25 over the words. The
        # keyword vote only reorders; weak matches still go last, so a hit on a common word
        # ("my", "today") can't lift a memory that isn't about this message.
        by_rank = [m["id"] for m in sorted(semantic, key=lambda m: -m["rank"])]
        by_words = [m["id"] for m in sorted((m for m in semantic if m["bm25"] is not None), key=lambda m: -m["bm25"])]
        fused = rrf(by_rank, by_words)
        for m in semantic:
            m["keyword_rank"] = by_words.index(m["id"]) + 1 if m["id"] in by_words else None
            m["fused"] = round(fused[m["id"]], 4)
        semantic = sorted(semantic, key=lambda m: (m["weak"], -m["fused"]))[:k]
        # History ("where did I live before?"): embeddings can't express "before", so walk the
        # graph instead — older versions of the best-matching current memories.
        strong = [m for m in semantic if not m["weak"]]  # no history for a match that isn't about this
        top = max((m["score"] for m in strong), default=0)
        best = [m["id"] for m in sorted(strong, key=lambda m: -m["score"])[:3] if m["score"] >= top - 0.02]
        history = q(
            """
            MATCH (cur:Memory)-[:SUPERSEDES*1..3]->(m:Memory) WHERE cur.id IN $ids AND """ + NOT_PROTECTED + """
              AND """ + visible("m") + """
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
              CALL db.index.vector.queryNodes($entity_vec, 50, $vec) YIELD node AS e, score
              WITH e, score WHERE e.user = $user AND e.kind <> 'self' AND score >= $hop
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
              AND """ + entity_visible("e") + """
            OPTIONAL MATCH (:User {id: $user})-[r:RELATED]->(e) WHERE r.until IS NULL AND """ + visible("r") + """
            OPTIONAL MATCH (m:Memory)-[:ABOUT]->(e)
              WHERE m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now) AND coalesce(m.status, '') <> 'done' AND NOT coalesce(m.sensitive, false)
                AND """ + NOT_PROTECTED + """ AND """ + visible("m") + """
            WITH e, r, m ORDER BY m.importance DESC
            WITH e, head(collect(r.type)) AS relation,
                 collect(DISTINCT m {.id, .kind, .text, .importance, .occurred_on, .due_on,
                                     age: duration.inSeconds(coalesce(m.last_confirmed, m.created_at), $now).seconds / 86400.0})[..20] AS memories
            // Its links to other entities, both directions and up to two steps (Logan -likes->
            // Minecraft is found from either end; "my son's friend Max" -> Max's dog), each end
            // labelled with its relation to the user if it has one. Never through the user, a
            // protected entity or a link this persona can't see.
            WITH e, relation, memories, COLLECT {
              MATCH p = (e)-[:RELATED*1..2]-(o:Entity)
              WHERE all(x IN nodes(p)[1..] WHERE NOT x:User AND """ + entity_visible("x") + """
                        AND NOT EXISTS { MATCH (b2:Memory {kind: 'boundary'})-[:ABOUT]->(x)
                                         WHERE b2.invalidated_at IS NULL AND (b2.valid_until IS NULL OR b2.valid_until > $now) })
                AND all(r2 IN relationships(p) WHERE r2.until IS NULL AND """ + visible("r2") + """)
              UNWIND relationships(p) AS r2
              WITH DISTINCT r2, startNode(r2) AS s, endNode(r2) AS t
              RETURN {source: s.name, type: r2.type, target: t.name,
                      source_rel: COLLECT { MATCH (:User {id: $user})-[u:RELATED]->(s) WHERE u.until IS NULL AND """ + visible("u") + """ RETURN u.type }[0],
                      target_rel: COLLECT { MATCH (:User {id: $user})-[u:RELATED]->(t) WHERE u.until IS NULL AND """ + visible("u") + """ RETURN u.type }[0]}
              LIMIT 15
            } AS links
            RETURN e.id AS id, e.name AS name, e.kind AS kind, e.summary AS summary, relation, memories, links
            LIMIT 12
            """,
            q=query, vec=vec, hop=ENTITY_HOP_SCORE, types=group_types, inferred=inferred_ids,
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
            MATCH (:User {id: $user})-[r:RELATED]->(e:Entity) WHERE r.until IS NULL AND """ + visible("r") + """
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
              AND """ + NOT_PROTECTED + """ AND """ + visible("m") + """
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
              AND """ + NOT_PROTECTED + """ AND """ + visible("m") + """
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
              AND """ + NOT_PROTECTED + """ AND """ + visible("m") + """
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.valid_until AS until,
                   coalesce(m.sensitive, false) AS sensitive ORDER BY m.valid_until DESC LIMIT 5
            """,
            hours=RECENT_HOURS,
        )
        episodes = q(
            """
            MATCH (ep:Episode {user: $user}) WHERE ep.summary IS NOT NULL
              AND (ep.agent IS NULL OR ep.agent = $agent)  // summaries can quote private things
            RETURN ep.started_at AS at, ep.summary AS summary, ep.mood AS mood
            ORDER BY ep.started_at DESC LIMIT 3
            """
        )
        touched = [m["id"] for m in semantic if not m["weak"]] + [m["id"] for e in entities for m in e["memories"]]
        if touch:
            q(
            """
                MATCH (m:Memory) WHERE m.id IN $ids
                SET m.last_accessed = datetime(), m.access_count = coalesce(m.access_count, 0) + 1
                """,
                ids=touched,
            )
        return {"dormant": dormant, "relevance_min": RELEVANCE_MIN if PROFILE.weak_below is None else PROFILE.weak_below,
                "relevance_by": "cross-encoder" if PROFILE.weak_below is None else "vector score", "core": core, "semantic": semantic, "entities": entities, "people": people,
                "followups": followups, "upcoming": upcoming, "history": history, "episodes": episodes,
                "recent": recent}

    # ------------------------------------------------------------------ consolidation
    def consolidate(self, today: dt.date | None = None, writer=None, prune: bool = True) -> list[str]:
        """Background upkeep (Lesson 13). Run after a conversation, or nightly:

        1. lapse   open plans long past due stop being followed up
        2. roll up memories ABOUT one activity on 3+ days ("at the gym" x3) into one lasting
                   pattern (see _habits), written by `writer(user, texts, today) -> MemoryOut
                   | None` (an LLM call, injected so tests can use a fake); the members are
                   marked consolidated and the user -practices-> the activity
        3. prune   delete dormant, low-importance memories that have expired (data minimization),
                   but only once they're rolled up or older than the pattern window: pruning
                   earlier would delete the repeats before they could ever form a pattern
        4. orphans entities with no memory and no link left are deleted (see _drop_orphans)

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
            for h in self._habits(now):
                if h["pattern"]:  # a known habit: new days are more evidence, not a new sentence
                    self._write(lambda: self._summarize([h["pattern"]], h["members"], now))
                    log.append(f"pattern {h['pattern']} gained {len(h['members'])} memories ({h['name']})")
                    continue
                pattern = writer(self.user_name, [r["text"] for r in h["members"]], today)
                if pattern is None:
                    continue
                log += self._write(lambda: self._store_pattern(pattern, h, today, now))
        if prune:
            log += self._prune(now)
        log += self._drop_orphans()
        return log

    def _drop_orphans(self) -> list[str]:
        """4. orphans  an entity nothing points at any more — no memory (current or retired)
        is about it and it has no link, past or present — is left over from a forget, a prune
        or an extraction that named it without saying how it connects ("Disney+"). It would
        float alone in the life graph and be matched by recall for nothing, so it goes."""
        gone = self._q(
            """
            MATCH (e:Entity {user: $user}) WHERE NOT e:User
              AND NOT EXISTS { (e)<-[:ABOUT]-(:Memory) } AND NOT EXISTS { (e)-[:RELATED]-() }
            WITH e, e.name AS name DETACH DELETE e RETURN name
            """
        )
        return [f"dropped {len(gone)} unconnected entities: {', '.join(sorted(g['name'] for g in gone))}"] if gone else []

    def _habits(self, now: dt.datetime) -> list[dict]:
        """2. habits  an activity with memories ABOUT it on PATTERN_MIN+ distinct days in the
        window. Read from the graph, not from how the sentences are worded: the vector
        clustering this replaced broke when a new extraction model wrote longer, dated
        sentences (structure-over-prose spec). A day is when it happened (occurred_on, or a
        plan's due day once it has come), else the day it was said; several mentions on one
        day count once. A plan still in the future isn't a day yet, and valid_until is when a
        status ends, not another day ("gym Mon, Wed, Fri next week" is one message, not a
        habit), and plans alone are no habit until something happened. Occurrences only (event, status, plan, feeling): facts and preferences are
        what a habit becomes. `pattern` is the activity's current habit memory, if any."""
        return self._q(
            """
            MATCH (a:Entity {user: $user, kind: 'activity'})<-[:ABOUT]-(m:Memory {user: $user})
            WHERE m.invalidated_at IS NULL
              // already evidence for THIS activity's habit; a memory about two activities
              // still counts for the other one, even in a later run
              AND NOT EXISTS { (p:Memory)-[:SUMMARIZES]->(m) WHERE (p)-[:ABOUT]->(a) }
              AND m.kind IN ['event', 'status', 'plan', 'feeling']
              AND coalesce(m.status, '') <> 'lapsed'  // a plan that never happened is no evidence
              AND NOT coalesce(m.sensitive, false)
              AND NOT coalesce(m.private, false)  // a shared pattern must not be built from private memories
              AND m.created_at > $now - duration({days: $days})
              AND NOT coalesce(m.due_on > date($now), false)  // not yet: no evidence
            WITH a, m, coalesce(m.occurred_on, m.due_on, date(m.created_at)) AS day
            ORDER BY m.created_at
            WITH a, collect(m {.id, .text}) AS members, count(DISTINCT day) AS days,
                 count(CASE WHEN m.kind <> 'plan' THEN 1 END) AS happened
            WHERE days >= $min AND happened > 0  // plans alone: nobody said it happened
            RETURN a.id AS activity, a.name AS name, members,
                   COLLECT { MATCH (p:Memory {user: $user})-[:ABOUT]->(a)
                             WHERE p.evidence IS NOT NULL AND p.invalidated_at IS NULL
                             RETURN p.id ORDER BY p.created_at }[0] AS pattern
            """, now=now, days=PATTERN_WINDOW_DAYS, min=PATTERN_MIN)

    def _store_pattern(self, pattern, habit: dict, today: dt.date, now: dt.datetime) -> list[str]:
        """The pattern, its SUMMARIZES links and user -practices-> activity commit together
        (the LLM call happened before the transaction, so the user's write lock isn't held
        while waiting on it). The pattern is ABOUT the activity, so the habit is in the graph."""
        group = habit["members"]
        pattern = pattern.model_copy(update={"about": list(dict.fromkeys([*(pattern.about or ["user"]), habit["activity"]]))})
        found = Extraction(memories=[pattern],
                           relations=[RelationOut(source="user", target=habit["activity"], type="practices")])
        log = [f"pattern from {len(group)} memories: {line}" for line in self.apply(found, today=today)]
        # By id, not by text: if apply() reinforced an existing pattern ("goes to the gym
        # regularly" vs "works out at the gym often"), the texts differ.
        self._summarize(self.written_ids, group, now)
        return log

    def _summarize(self, pattern_ids: list[str], members: list[dict], now: dt.datetime) -> None:
        """Link a pattern to its members. evidence goes up by the members newly linked: a
        member linked twice (a memory about two activities whose patterns turned out to be
        one) counts once, and pruning a member later (its edge goes too) doesn't lower it."""
        self._q(
            """
            MATCH (p:Memory {user: $user}) WHERE p.id IN $pattern
            CALL (p) {
              MATCH (m:Memory) WHERE m.id IN $ids
              WITH m, EXISTS { (p)-[:SUMMARIZES]->(m) } AS had
              SET m.consolidated_at = $now MERGE (p)-[:SUMMARIZES]->(m)
              RETURN count(CASE WHEN NOT had THEN 1 END) AS added
            }
            SET p.evidence = coalesce(p.evidence, 0) + added
            """, now=now, pattern=pattern_ids, ids=[r["id"] for r in members])

    def _prune(self, now: dt.datetime) -> list[str]:
        """Delete dormant trivia: one statement, so it's atomic without _write(). Only what
        was temporary and has ended (valid_until passed): "temporary" is read from the field,
        not from `kind` (structure-over-prose spec). Kind still sets how fast it decays."""
        dormant = [r for r in self._q(
            """
            MATCH (m:Memory {user: $user}) WHERE m.valid_until IS NOT NULL AND m.valid_until <= $now
              AND m.importance <= $max_imp AND NOT EXISTS { (m)<-[:SUPERSEDES]-() }
              AND (m.consolidated_at IS NOT NULL OR m.created_at < $now - duration({days: $days}))
            RETURN m.id AS id, m.kind AS kind, m.importance AS importance,
                   duration.inSeconds(coalesce(m.last_confirmed, m.created_at), $now).seconds / 86400.0 AS age
            """, now=now, max_imp=PRUNE_MAX_IMPORTANCE, days=PATTERN_WINDOW_DAYS)
            if decay.is_dormant(r["importance"], r["kind"], r["age"])]
        if not dormant:
            return []
        self._q("MATCH (m:Memory) WHERE m.id IN $ids DETACH DELETE m", ids=[r["id"] for r in dormant])
        return [f"pruned {len(dormant)} dormant expired memories"]

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
              AND """ + visible("m") + """
            OPTIONAL MATCH (m)-[:ABOUT]->(e)
            RETURN m.id AS id, m.kind AS kind, m.text AS text, m.importance AS importance,
                   m.status AS status, m.invalidated_at IS NOT NULL AS retired,
                   m.valid_until AS valid_until, m.valid_until <= $now AS expired,
                   coalesce(m.sensitive, false) AS sensitive, m.annual_date AS annual_date,
                   m.created_at AS created_at, collect(e.name) AS about,
                   m.agent AS agent, coalesce(m.private, false) AS private
            ORDER BY created_at
            """,
            kind=kind, all=include_retired, now=clock.now_for(today or dt.date.today()),
            asof=clock.now_for(today) if today else None,
        )

    def timeline(self) -> list[dict]:
        return self._q(
            """
            MATCH (m:Memory {user: $user})
            WHERE coalesce(m.occurred_on, m.due_on) IS NOT NULL  // anything with a day, whatever its kind
              AND """ + visible("m") + """
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

    def forget(self, memory_id: str, today: dt.date | None = None) -> bool:
        """User-requested forgetting: hard delete (privacy beats provenance here).

        The turns that produced it are still in this episode's transcript, and the
        end-of-episode pass would extract it again. A tombstone keeps its embedding and
        non-user entity ids (no text), scoped to that episode. apply() skips matching memories."""
        deleted = bool(self._q(
            """
            MATCH (m:Memory {id: $id, user: $user})
            OPTIONAL MATCH (m)-[:FROM]->(ep:Episode)
            CREATE (t:Forgotten {user: $user, episode: coalesce(ep.id, ''), at: datetime()})
            SET t.about = [(m)-[:ABOUT]->(e) WHERE NOT e:User | e.id]
            WITH m, m.id AS id, t
            CALL (m, t) { WITH * WHERE m[$vprop] IS NOT NULL  // none yet if made before a model switch
                          CALL db.create.setNodeVectorProperty(t, $vprop, m[$vprop]) }
            DETACH DELETE m RETURN id
            """,
            id=memory_id,
        ))
        if deleted:  # edges it alone supported go with it
            self._retire_links([(memory_id, None)], today or dt.date.today())
        return deleted

    def _tombstones(self, episode_id: str | None) -> list:
        if not episode_id:
            return []
        # a tombstone from before an embedding-model switch has no vector for this model (Lesson 19)
        rows = self._q("MATCH (t:Forgotten {user: $user, episode: $ep}) WHERE t[$vprop] IS NOT NULL "
                       "RETURN t[$vprop] AS v, coalesce(t.about, []) AS about", ep=episode_id)
        return [(np.asarray(r["v"]), set(r["about"])) for r in rows]

    def forget_entity(self, name: str, known: dict[str, str] | None = None) -> dict | None:
        """"Forget everything about Jordan": delete the entity and every memory about or naming
        it, and redact the name from stored turns and episode summaries. One transaction:
        a half-finished forget would leave the name in some places."""
        return self._write(lambda: self._forget_entity(name, known))

    def _forget_entity(self, name: str, known: dict[str, str] | None) -> dict | None:
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
                WITH ep CALL db.create.setNodeVectorProperty(ep, $vprop, $vec)
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
                WITH e CALL db.create.setNodeVectorProperty(e, $vprop, $vec)
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
        self.__init__(self.driver, self.user_name, self.agent)


SENSITIVE_NOTE = " (sensitive: only discuss if they raise it)"
STALE_NOTE = " (not confirmed in over a year: may be out of date, check before relying on it)"


# The extractor calls Minecraft a "thing" one turn and an "activity" the next: for matching,
# those two are one kind, or the graph grows a second Minecraft.
_KIND_GROUPS = [{"thing", "activity"}]


def _compatible_kinds(kind: str) -> list[str]:
    return sorted(next((g for g in _KIND_GROUPS if kind in g), {kind}))


def _link_line(l: dict) -> str:
    """An edge between two entities, readable. Role edges read "target is source's role"
    (Max is Logan (your son)'s friend); verb-like ones (likes, lives_in, works_at) read
    straight through (Logan (your son) likes Minecraft)."""
    side = lambda name, rel: f"{name} (your {rel.replace('_', ' ')})" if rel else name
    src, dst = side(l["source"], l.get("source_rel")), side(l["target"], l.get("target_rel"))
    t = l["type"]
    if "_" in t or "interests" in ontology.ancestors(t):
        return f"{src} {t.replace('_', ' ')} {dst}"
    return f"{dst} is {src}'s {t}"


def day_note(m: dict) -> str:
    """A memory's day, from its field: known days aren't written into the text (extraction
    rule 2), so the prompt adds them back."""
    if m.get("occurred_on"):
        return f" (on {m['occurred_on']})"
    if m.get("due_on"):
        return f" (due {m['due_on']})"
    return ""


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
    # Weak matches stay in this list, after the strong ones, with no label: the reranker
    # misses inferential links ("plan my training" vs the half marathon: -11.2), so a "maybe
    # unrelated" label told the model to ignore its best memory (the personas scenario).
    seen = {m["text"] for m in r["core"] + r["followups"]}
    relevant = [m for m in r["semantic"] if m["text"] not in seen]
    for e in r["entities"]:
        relevant += [m for m in e["memories"] if m["text"] not in seen | {x["text"] for x in relevant}]
    if relevant:
        out.append("Relevant memories:\n" + "\n".join(
            f"- [{m['kind']}] {m['text']}{day_note(m)}" + (SENSITIVE_NOTE if m.get("sensitive") else "")
            + (STALE_NOTE if m.get("stale") else "") for m in relevant))
    links = list(dict.fromkeys(_link_line(l) for e in r["entities"] for l in e.get("links", [])))
    if links:
        out.append("How they connect:\n" + "\n".join(f"- {l}" for l in links))
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
