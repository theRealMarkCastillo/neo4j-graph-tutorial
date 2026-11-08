"""A preview must read real matching rules without executing even temporary writes."""
import datetime as dt

import pytest

from companion import clock
from companion.models import EntityOut, Extraction, MemoryOut, RelationOut
from companion.store import MemoryStore
from companion.reextract import episodes

DAY = dt.date(2026, 11, 6)


def test_read_view_skips_initialization(monkeypatch):
    from companion import store as module

    def forbidden(*args, **kwargs):
        raise AssertionError("read-only construction attempted initialization")

    monkeypatch.setattr(module, "ensure_schema", forbidden)
    monkeypatch.setattr(module.ontology, "ensure_ontology", forbidden)
    monkeypatch.setattr(module, "embed", forbidden)
    view = MemoryStore(None, "preview nonexistent", read_only=True)
    with pytest.raises(PermissionError, match="read-only"):
        view.apply(Extraction())


@pytest.mark.parametrize("query", [
    "CREATE (:Entity {user:$user, name:'accidental'})",
    "MATCH (u:User {id:$user}) SET u.name = 'accidental'",
    "CALL db.create.setNodeVectorProperty(null, 'embedding', [0.1])",
])
def test_read_view_rejects_writes(store, query):
    with MemoryStore.read_view(store.driver, store.user_name) as view:
        with pytest.raises(PermissionError, match="read-only"):
            view._q(query)
        assert view._q("MATCH (e:Entity {user:$user, name:'accidental'}) RETURN e") == []
        with pytest.raises(PermissionError, match="read-only"):
            view.apply(Extraction())


def test_read_view_unknown_user_is_not_created(driver):
    name = "preview nonexistent"
    with MemoryStore.read_view(driver, name) as view:
        assert episodes(view) == []
        assert view._q("MATCH (u:User {id:$user}) RETURN u") == []
    assert driver.execute_query("MATCH (u:User {id:$user}) RETURN u", user="user:" + name).records == []


@pytest.mark.parametrize("before,incoming,expected", [
    ({"name": "my son", "kind": "person", "summary": "", "aliases": []},
     {"name": "Logan", "kind": "person", "ref": "e_1"}, ("Logan", "person", ["my son"])),
    ({"name": "Maya", "kind": "person", "summary": "", "aliases": []},
     {"name": "Maia", "kind": "person", "ref": "e_1", "corrected": True}, ("Maia", "person", ["Maya"])),
    ({"name": "running", "kind": "thing", "summary": "", "aliases": []},
     {"name": "running", "kind": "activity"}, ("running", "activity", [])),
])
def test_entity_preview_reports_reconciliation(before, incoming, expected):
    view = MemoryStore(None, "pytest", read_only=True)
    changes = view._entity_changes(before, EntityOut(**incoming))
    assert (changes["name"], changes["kind"], changes["aliases"]) == expected


def test_entity_match_preview_honors_explicit_new(monkeypatch):
    from companion import store as module
    import numpy as np
    monkeypatch.setattr(module, "embed", lambda texts: [np.array([1., 0.])])
    view = MemoryStore(None, "pytest", read_only=True)
    # An explicit new entity needs no database match when no ref is given.
    eid, vec = view._find_entity("Sam", "person", (), "", None, True)
    assert eid is None
    assert list(vec) == [1., 0.]


@pytest.mark.parametrize("case,want", [("new", "new"), ("duplicate", "reinforce"),
                                      ("repeat", "skip"), ("tombstone", "skip")])
def test_memory_decision_classifies_candidates(monkeypatch, case, want):
    import numpy as np
    from companion import store as module
    view = MemoryStore(None, "pytest", read_only=True)
    monkeypatch.setattr(module, "DUPLICATE_SCORE", 0.9)
    monkeypatch.setattr(module, "FORGOTTEN_SCORE", 0.9)
    monkeypatch.setattr(view, "_q", lambda *a, **kw: [{"id": "m_1", "text": "old", "score": 1.}] if case == "duplicate" else [])
    vec = np.array([1., 0.])
    mem = MemoryOut(text="A fact", kind="fact", importance=6)
    pairs = [(vec, set())]
    result = view._memory_decision(mem, vec, set(), None, clock.now_for(DAY),
                                   pairs if case == "repeat" else [], pairs if case == "tombstone" else [])
    assert result["action"] == want
    if want == "reinforce":
        assert result["id"] == "m_1" and result["score"] == 1.


@pytest.mark.parametrize("kind", ["person", "activity", "thing"])
def test_entity_match_preview_agrees_with_apply(store, kind):
    eid = store.resolve_entity("Sam", kind, aliases=["buddy"])
    with MemoryStore.read_view(store.driver, store.user_name) as view:
        assert view._find_entity("buddy", kind, (), "", None, False)[0] == eid
        assert view._find_entity("Sam", kind, (), "", eid, False)[0] == eid
        assert view._find_entity("Sam", kind, (), "", None, True)[0] is None
    assert store.resolve_entity("buddy", kind) == eid


@pytest.mark.parametrize("different", [None, "date", "entity", "expired", "private"])
def test_memory_decision_agrees_with_apply(store, monkeypatch, different):
    from companion import store as module
    from companion.embeddings import embed
    monkeypatch.setattr(module, "DUPLICATE_SCORE", 0.0)
    old = MemoryOut(text="Pytest lives in Boston", kind="fact", importance=5, about=["Boston"],
                    due_on="2026-11-06", valid_until="2026-11-07T23:59" if different == "expired" else None,
                    private=different == "private")
    store.apply(Extraction(entities=[EntityOut(name="Boston", kind="place")], memories=[old]), today=DAY)
    new = old.model_copy(update={"private": False, "due_on": "2026-11-08" if different == "date" else old.due_on,
                                "about": [] if different == "entity" else ["Boston"]})
    now = clock.now_for(DAY + dt.timedelta(days=3) if different == "expired" else DAY)
    agent = "Other" if different == "private" else store.agent
    with MemoryStore.read_view(store.driver, store.user_name, agent) as view:
        about = {view._lookup(n, {}) for n in new.about}
        result = view._memory_decision(new, embed([new.text])[0], about, None, now, [], [])
    assert result["action"] == ("reinforce" if different is None else "new")
    writer = MemoryStore(store.driver, store.user_name, agent=agent)
    log = writer.apply(Extraction(memories=[new]), today=now.date())
    assert any(line.startswith("reinforced") for line in log) is (result["action"] == "reinforce")


def test_preview_includes_text_and_dates():
    from companion.preview import format_preview
    item = {"category": "memory", "action": "new", "text": "Pytest plans to visit Denver",
            "reason": "no current match", "details": {"kind": "plan", "about": ["Denver"],
            "due_on": "2026-11-19", "valid_until": "2026-11-20T23:59", "annual_date": "11-19"}}
    text = "\n".join(format_preview([item]))
    for value in ("Pytest plans to visit Denver", "plan", "Denver", "2026-11-19", "2026-11-20T23:59", "11-19", "new"):
        assert value in text


@pytest.mark.parametrize("kind,until,want", [("event", "2026-11-10T23:59", True), ("status", None, True),
                                             ("fact", None, False)])
def test_preview_trip_rule_is_expiry_based(kind, until, want):
    from companion.store import temporary_relation
    ex = Extraction(memories=[MemoryOut(text="In Denver", kind=kind, valid_until=until,
                                       about=["Denver"], importance=3)])
    rel = RelationOut(source="user", target="Denver", type="lives_in")
    assert temporary_relation(ex, rel, "lives_in", clock.now_for(DAY)) is want


def test_preview_relations_use_ontology_and_trip_guard(store):
    from companion.preview import preview_extraction
    store.apply(Extraction(entities=[EntityOut(name="Chicago", kind="place")],
                           relations=[RelationOut(source="user", target="Chicago", type="lives_in")]), today=DAY)
    ex = Extraction(entities=[EntityOut(name="Denver", kind="place")],
                    relations=[RelationOut(source="user", target="Denver", type="lives_in"),
                               RelationOut(source="user", target="Denver", type="places"),
                               RelationOut(source="user", target="Denver", type="sister")],
                    memories=[MemoryOut(text="In Denver this week", kind="event", importance=3, about=["Denver"],
                                        valid_until="2026-11-10T23:59")])
    with MemoryStore.read_view(store.driver, store.user_name) as view:
        items = preview_extraction(view, ex, "", DAY)
    relations = [i for i in items if i["category"] == "relation"]
    assert [i["action"] for i in relations] == ["skip", "reject", "reject"]
    assert "trip" in relations[0]["reason"]
    assert "range" in relations[2]["reason"]


def test_preview_unresolved_and_proposed_entities(store):
    from companion.preview import preview_extraction
    ex = Extraction(entities=[EntityOut(name="swimming", kind="activity", new=True)], memories=[
        MemoryOut(text="Pytest went swimming", kind="event", importance=3, about=["swimming"]),
        MemoryOut(text="Pytest met someone", kind="event", importance=3, about=["unidentified"]),
    ])
    with MemoryStore.read_view(store.driver, store.user_name) as view:
        items = preview_extraction(view, ex, "", DAY)
    assert items[0]["action"] == "new" and items[0].get("id") is None
    memories = [i for i in items if i["category"] == "memory"]
    assert [i["action"] for i in memories] == ["new", "unresolved"]
    assert "proposal:" not in str(items)


def test_preview_reports_memory_side_effects(store):
    from companion.preview import preview_extraction
    store.apply(Extraction(memories=[MemoryOut(text="Pytest plays cello", kind="fact", importance=3),
                                    MemoryOut(text="Pytest's old hobby", kind="fact", importance=3),
                                    MemoryOut(text="Dentist", kind="plan", importance=3, due_on="2026-11-10")]), today=DAY)
    rows = store.memories(today=DAY)
    ids = {m["text"]: m["id"] for m in rows}
    ex = Extraction(memories=[MemoryOut(text="Pytest plays cello", kind="fact", importance=9,
                                       supersedes=[ids["Pytest's old hobby"]])],
                    completed_plans=[ids["Dentist"]])
    with MemoryStore.read_view(store.driver, store.user_name) as view:
        items = preview_extraction(view, ex, "", DAY)
    memory = next(i for i in items if i["category"] == "memory")
    assert memory["action"] == "reinforce" and memory["id"] == ids["Pytest plays cello"]
    assert memory["details"]["changes"]["importance"] == 9
    assert any(i["category"] == "retirement" and i["id"] == ids["Pytest's old hobby"] for i in items)
    assert any(i["category"] == "completion" and i["id"] == ids["Dentist"] for i in items)


def test_dry_run_empty_user_skips_model(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from companion import reextract as module

    class ReadOnlyStore:
        def __init__(self, *args, **kw):
            raise AssertionError("dry run attempted normal store construction")

        @classmethod
        @contextmanager
        def read_view(cls, driver, user, agent=None):
            yield SimpleNamespace(_q=lambda *args, **kw: [])

    monkeypatch.setattr(module, "MemoryStore", ReadOnlyStore)
    monkeypatch.setattr(module, "episode_pass", lambda *args, **kw: pytest.fail("empty users must not invoke model"))
    assert "no episodes" in " ".join(module.reextract(None, "absent", dry_run=True)).lower()


def database_state(driver):
    """Hash snapshots to keep private database content out of assertion diffs."""
    import hashlib
    import json
    queries = [
        "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS p ORDER BY id",
        "MATCH (a)-[r]->(b) RETURN elementId(r) AS id, elementId(a) AS a, elementId(b) AS b, type(r) AS type, properties(r) AS p ORDER BY id",
        "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, options RETURN * ORDER BY name",
        "SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties RETURN * ORDER BY name",
    ]
    return tuple(hashlib.sha256(json.dumps([r.data() for r in driver.execute_query(q).records],
                 sort_keys=True, default=str).encode()).hexdigest() for q in queries)


def test_dry_run_preserves_full_database_state(store, monkeypatch):
    from companion import graph
    from companion.reextract import reextract
    ep = store.start_episode(on=DAY)
    store.log_turn(ep, store.user_name, "I took up swimming")
    monkeypatch.setattr(graph, "extract_episode", lambda *a, **kw: Extraction(
        entities=[EntityOut(name="swimming", kind="activity", new=True)],
        memories=[MemoryOut(text="Pytest enjoys swimming", kind="preference", importance=6, about=["swimming"])],
    ))
    before = database_state(store.driver)
    output = "\n".join(reextract(store.driver, store.user_name, dry_run=True))
    assert "Pytest enjoys swimming" in output and "current-graph preview" in output
    assert "entity:new" in output and "memory:new" in output
    assert database_state(store.driver) == before


def test_dry_run_uses_episode_date_and_persona(store, monkeypatch):
    from companion import graph
    from companion.reextract import reextract
    other = MemoryStore(store.driver, store.user_name, agent="Coach")
    other.apply(Extraction(memories=[MemoryOut(text="Private cello secret", kind="fact", importance=6, private=True)]), today=DAY)
    ep = store.start_episode(on=DAY)
    store.log_turn(ep, store.user_name, "I enjoy swimming")
    seen = []

    def extract(user, transcript, candidates, entities, today, refresh):
        seen.append((today, refresh, candidates))
        return Extraction()

    monkeypatch.setattr(graph, "extract_episode", extract)
    output = "\n".join(reextract(store.driver, store.user_name, dry_run=True))
    assert seen and seen[0][0:2] == (DAY, True)
    assert all(c["text"] != "Private cello secret" for c in seen[0][2])
    assert store.agent in output


def test_dry_run_missing_index_is_actionable(monkeypatch):
    from contextlib import contextmanager
    from companion import reextract as module
    from neo4j.exceptions import ClientError

    class ReadOnlyStore:
        @classmethod
        @contextmanager
        def read_view(cls, *args, **kw):
            raise ClientError("No such vector schema index")
            yield

    monkeypatch.setattr(module, "MemoryStore", ReadOnlyStore)
    with pytest.raises(RuntimeError, match="setup"):
        module.reextract(None, "pytest", dry_run=True)


def test_apply_mode_still_applies(store, monkeypatch):
    from companion import graph
    from companion.reextract import reextract
    ep = store.start_episode(on=DAY)
    store.log_turn(ep, store.user_name, "I started swimming")
    monkeypatch.setattr(graph, "extract_episode", lambda *a, **kw: Extraction(
        memories=[MemoryOut(text="Pytest started swimming", kind="fact", importance=6)]))
    reextract(store.driver, store.user_name)
    assert any(m["text"] == "Pytest started swimming" for m in store.memories(today=DAY))


def test_shared_entity_helper_preserves_direct_store_kind_support(store):
    # resolve_entity historically accepts graph kinds beyond the extractor's enum.
    eid = store.resolve_entity("Preview project", "project")
    assert store.resolve_entity("Preview project", "project") == eid


def test_preview_normalizes_flipped_existing_relations(store):
    from companion.preview import preview_extraction
    store.apply(Extraction(entities=[EntityOut(name="Maya", kind="person")],
                           relations=[RelationOut(source="user", target="Maya", type="sister")]), today=DAY)
    ex = Extraction(relations=[RelationOut(source="Maya", target="user", type="sister_of")])
    with MemoryStore.read_view(store.driver, store.user_name) as view:
        [item] = preview_extraction(view, ex, "", DAY)
    assert item["action"] == "existing"
    assert item["details"]["flipped"] is True
    assert item["text"] == "user -sister-> Maya"


def test_preview_reports_cardinality_endings_and_unknown_types(store):
    from companion.preview import preview_extraction
    store.apply(Extraction(entities=[EntityOut(name="Chicago", kind="place")],
                           relations=[RelationOut(source="user", target="Chicago", type="lives_in")]), today=DAY)
    ex = Extraction(entities=[EntityOut(name="Denver", kind="place", new=True)], relations=[
        RelationOut(source="user", target="Denver", type="moved_to"),
        RelationOut(source="user", target="Denver", type="visits_weekly"),
    ])
    before = database_state(store.driver)
    with MemoryStore.read_view(store.driver, store.user_name) as view:
        items = preview_extraction(view, ex, "", DAY)
    assert any(i["action"] == "end" and "Chicago" in i["text"] for i in items)
    assert any(i["action"] == "new" and "unknown type" in i["reason"] for i in items)
    assert database_state(store.driver) == before


def test_preview_respects_tombstones_and_same_extraction_repeats(store):
    from companion.preview import preview_extraction
    ep = store.start_episode(on=DAY)
    memory = MemoryOut(text="Pytest enjoys calligraphy", kind="preference", importance=5)
    ex = Extraction(memories=[memory])
    with MemoryStore.read_view(store.driver, store.user_name) as view:
        items = preview_extraction(view, Extraction(memories=[memory, memory]), ep, DAY)
    assert [i["action"] for i in items] == ["new", "skip"]
    store.apply(ex, ep, DAY)
    [row] = store.memories(today=DAY)
    store.forget(row["id"], today=DAY)
    with MemoryStore.read_view(store.driver, store.user_name) as view:
        [item] = preview_extraction(view, ex, ep, DAY)
    assert item["action"] == "skip" and "forget" in item["reason"]


def test_preview_reports_restated_plans_and_forgetting(store):
    from companion.preview import preview_extraction
    plan = MemoryOut(text="Dentist", kind="plan", importance=5, due_on="2026-11-09")
    store.apply(Extraction(entities=[EntityOut(name="Jordan", kind="person")], memories=[plan]), today=DAY)
    [row] = store.memories(today=DAY)
    ex = Extraction(memories=[plan], completed_plans=[row["id"]], forget_entities=["Jordan"])
    with MemoryStore.read_view(store.driver, store.user_name) as view:
        items = preview_extraction(view, ex, "", DAY)
    assert any(i["category"] == "completion" and i["action"] == "skip" and "keep open" in i["reason"] for i in items)
    assert any(i["category"] == "forgetting" and i["action"] == "end" for i in items)


@pytest.mark.parametrize("query_type", [None, "rw", "w", "s"])
def test_read_view_rejects_unknown_or_write_query_classification(query_type):
    from types import SimpleNamespace
    view = MemoryStore(None, "pytest", read_only=True)

    class Tx:
        def run(self, query, **kwargs):
            if not query.startswith("EXPLAIN "):
                pytest.fail("unapproved query executed")
            return SimpleNamespace(consume=lambda: SimpleNamespace(query_type=query_type))

    view._local.tx = Tx()
    with pytest.raises(PermissionError, match="read-only"):
        view._q("RETURN 1 AS value")


def test_read_view_is_closed_after_context_exit(store):
    with MemoryStore.read_view(store.driver, store.user_name) as view:
        assert view._q("RETURN 1 AS value") == [{"value": 1}]
    with pytest.raises(PermissionError, match="open read_view"):
        view._q("RETURN 1 AS value")


def test_preview_exclusivity_only_ends_opposite_link_to_same_target(store):
    from companion.preview import preview_extraction
    store.apply(Extraction(entities=[EntityOut(name="Coffee", kind="thing", new=True),
                                     EntityOut(name="Tea", kind="thing", new=True)],
                           relations=[RelationOut(source="user", target="Coffee", type="dislikes"),
                                      RelationOut(source="user", target="Tea", type="dislikes")]), today=DAY)
    ex = Extraction(relations=[RelationOut(source="user", target="Coffee", type="likes")])
    with MemoryStore.read_view(store.driver, store.user_name) as view:
        items = preview_extraction(view, ex, "", DAY)
    endings = [i["text"] for i in items if i["category"] == "relation" and i["action"] == "end"]
    assert endings == ["user -dislikes-> Coffee"]
    store.apply(ex, today=DAY)
    assert store._q("MATCH (:User {id:$user})-[r:RELATED {type:'dislikes'}]->(e) WHERE r.until IS NULL RETURN e.name AS name") == [{"name": "Tea"}]


@pytest.mark.parametrize("before,incoming,after,action", [
    (False, True, False, "existing"), (True, False, False, "update"),
    (True, True, True, "existing"), (False, False, False, "existing"),
])
def test_preview_relation_privacy_matches_effective_apply_rule(store, before, incoming, after, action):
    from companion.preview import preview_extraction
    def extraction(private):
        return Extraction(entities=[EntityOut(name="Coffee", kind="thing")],
                          relations=[RelationOut(source="user", target="Coffee", type="likes")],
                          memories=[MemoryOut(text="Pytest likes coffee", kind="preference", importance=5,
                                              about=["Coffee"], private=private)])
    store.apply(extraction(before), today=DAY)
    ex = extraction(incoming)
    with MemoryStore.read_view(store.driver, store.user_name) as view:
        item = next(i for i in preview_extraction(view, ex, "", DAY) if i["category"] == "relation")
    assert item["details"]["private"] is after
    assert item["details"]["incoming_private"] is incoming
    assert item["details"]["private_before"] is before
    assert item["details"]["private_after"] is after
    assert item["action"] == action
    if before != after:
        assert "visibility" in item["reason"] and "shared" in item["reason"]
    store.apply(ex, today=DAY)
    [link] = store._q("MATCH (:User {id:$user})-[r:RELATED {type:'likes'}]->() RETURN r.private AS private")
    assert link["private"] is after


def test_preview_marks_relations_to_merging_entities_unresolved(store):
    from companion.preview import preview_extraction
    store.apply(Extraction(entities=[EntityOut(name="Maya", kind="person", new=True),
                                     EntityOut(name="my sister", kind="person", new=True)]), today=DAY)
    rows = store._q("MATCH (e:Entity {user:$user}) WHERE e.name IN ['Maya', 'my sister'] RETURN e.name AS name, e.id AS id")
    ids = {r["name"]: r["id"] for r in rows}
    ex = Extraction(entities=[EntityOut(name="Maya", kind="person", ref=ids["Maya"], aliases=["my sister"])],
                    relations=[RelationOut(source="user", target=ids["my sister"], type="friend"),
                               RelationOut(source="user", target=ids["Maya"], type="sister")])
    with MemoryStore.read_view(store.driver, store.user_name) as view:
        relations = [i for i in preview_extraction(view, ex, "", DAY) if i["category"] == "relation"]
    assert len(relations) == 2
    assert all(i["action"] == "unresolved" and "merge" in i["reason"] for i in relations)


def test_retirement_preview_discloses_unsimulated_downstream_effects():
    from companion.preview import format_preview
    item = {"category": "retirement", "action": "end", "text": "old fact", "reason": "requested"}
    assert "downstream relation" in " ".join(format_preview([item])).lower()


@pytest.mark.parametrize("field", ["invalidate", "completed_plans"])
def test_explicit_action_ids_do_not_disclose_other_personas_private_memory(store, field):
    from companion.preview import preview_extraction
    import json
    other = MemoryStore(store.driver, store.user_name, agent="Other")
    other.apply(Extraction(memories=[MemoryOut(text="Private hidden cello secret", kind="plan", importance=6,
                                              private=True, due_on="2026-11-09")]), today=DAY)
    [memory] = other.memories(today=DAY)
    ex = Extraction(**{field: [memory["id"]]})
    with MemoryStore.read_view(store.driver, store.user_name) as view:
        items = preview_extraction(view, ex, "", DAY)
    assert "Private hidden cello secret" not in json.dumps(items, default=str)
    assert any(i["action"] == "unresolved" and "visible" in i["reason"] for i in items)


def test_preview_items_are_json_serializable_without_custom_encoder(store):
    from companion.preview import preview_extraction
    import json
    store.apply(Extraction(entities=[EntityOut(name="Maya", kind="person")],
                           relations=[RelationOut(source="user", target="Maya", type="sister")]), today=DAY)
    with MemoryStore.read_view(store.driver, store.user_name) as view:
        items = preview_extraction(view, Extraction(relations=[RelationOut(source="user", target="Maya", type="sister")]), "", DAY)
    assert json.loads(json.dumps(items)) == items
