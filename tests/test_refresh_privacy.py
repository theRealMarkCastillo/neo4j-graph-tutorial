"""Historical re-extraction is not a fresh permission to share private material."""
import datetime as dt

import pytest

from companion import graph
from companion.models import EntityOut, Extraction, MemoryOut, RelationOut
from companion.store import MemoryStore

DAY = dt.date(2026, 11, 6)


def memory(text, about, private=False, kind="fact", **kwargs):
    return MemoryOut(text=text, kind=kind, importance=6, about=about, private=private, **kwargs)


def source_episode(store, private=True):
    ep = store.start_episode(on=DAY)
    store.log_turn(ep, store.user_name, "Ana is my friend and works as a nurse.")
    store.apply(Extraction(entities=[EntityOut(name="Ana", kind="person", summary="a friend")],
                           relations=[RelationOut(source="user", target="Ana", type="friend")],
                           memories=[memory("Pytest's friend Ana works as a nurse", ["user", "Ana"], private)]), ep, DAY)
    return ep


def refreshed(monkeypatch, extraction):
    monkeypatch.setattr(graph, "extract_episode", lambda *args, **kwargs: extraction)


def test_refresh_preserves_private_relation_and_new_private_derivations(store, monkeypatch):
    ep = source_episode(store)
    refreshed(monkeypatch, Extraction(
        entities=[EntityOut(name="Ana", kind="person"), EntityOut(name="pottery", kind="activity", new=True)],
        relations=[RelationOut(source="user", target="Ana", type="friend"),
                   RelationOut(source="Ana", target="pottery", type="likes")],
        memories=[memory("Ana enjoys pottery", ["Ana", "pottery"])],
    ))
    graph.episode_pass(store, ep, DAY, refresh=True)
    links = store._q("MATCH (:Entity {user:$user})-[r:RELATED]->() RETURN r.type AS type, r.private AS private")
    assert links and all(r["private"] is True for r in links)
    assert all(m["private"] for m in store.memories(today=DAY))
    other = MemoryStore(store.driver, store.user_name, agent="Other")
    assert other.memories(today=DAY) == []
    assert other.known_entities() == []


def test_normal_chat_still_allows_explicitly_shared_information(store):
    source_episode(store)
    store.apply(Extraction(relations=[RelationOut(source="user", target="Ana", type="friend")],
                           memories=[memory("Pytest knows Ana", ["Ana"])]), today=DAY)
    [link] = store._q("MATCH (:User {id:$user})-[r:RELATED {type:'friend'}]->() RETURN r.private AS private")
    assert link["private"] is False


def test_public_only_refresh_still_adds_shared_information(store, monkeypatch):
    ep = source_episode(store, private=False)
    refreshed(monkeypatch, Extraction(entities=[EntityOut(name="pottery", kind="activity", new=True)],
                                      relations=[RelationOut(source="Ana", target="pottery", type="likes")],
                                      memories=[memory("Ana enjoys pottery", ["Ana", "pottery"])]))
    graph.episode_pass(store, ep, DAY, refresh=True)
    other = MemoryStore(store.driver, store.user_name, agent="Other")
    assert any("pottery" in m["text"] for m in other.memories(today=DAY))


def test_private_refresh_keeps_shared_entity_metadata_and_hides_orphans(store, monkeypatch):
    ep = source_episode(store)
    eid = store.resolve_entity("Acme", "organization", summary="public description")
    store.apply(Extraction(memories=[memory("Pytest works at Acme", ["Acme"])]), today=DAY)
    refreshed(monkeypatch, Extraction(entities=[
        EntityOut(name="Acme", kind="organization", ref=eid, summary="confidential customer details"),
        EntityOut(name="Secret clinic", kind="organization", summary="private health detail", new=True),
    ]))
    graph.episode_pass(store, ep, DAY, refresh=True)
    [row] = store._q("MATCH (e:Entity {id:$id}) RETURN e.summary AS summary", id=eid)
    assert row["summary"] == "public description"
    other = MemoryStore(store.driver, store.user_name, agent="Other")
    assert not any("Secret clinic" in e or "confidential" in e for e in other.known_entities())


def test_private_refresh_does_not_add_private_about_to_shared_memory(store, monkeypatch):
    ep = source_episode(store)
    public = memory("Pytest knows Ana", ["user", "Ana"])
    store.apply(Extraction(memories=[public]), today=DAY)
    refreshed(monkeypatch, Extraction(entities=[EntityOut(name="Secret project", kind="thing", new=True)],
                                      memories=[public.model_copy(update={"about": ["user", "Ana", "Secret project"]})]))
    graph.episode_pass(store, ep, DAY, refresh=True)
    assert store._q("MATCH (m:Memory {user:$user, text:'Pytest knows Ana'})-[:ABOUT]->(e:Entity {name:'Secret project'}) RETURN e") == []
    other = MemoryStore(store.driver, store.user_name, agent="Other")
    assert not any("Secret project" in e for e in other.known_entities())


def test_private_refresh_defers_global_boundaries_and_ontology_examples(store, monkeypatch):
    ep = source_episode(store)
    refreshed(monkeypatch, Extraction(relations=[RelationOut(source="user", target="Ana", type="pytest_private_refresh_contact")],
                                      memories=[memory("Do not mention Ana's confidential job", ["Ana"], kind="boundary")]))
    try:
        log = graph.episode_pass(store, ep, DAY, refresh=True)
        assert not any(m["kind"] == "boundary" for m in store.memories(today=DAY))
        assert store._q("MATCH (t:RelType {name:'pytest_private_refresh_contact'}) RETURN t") == []
        assert any("deferred" in line for line in log)
    finally:
        store._q("MATCH (t:RelType {name:'pytest_private_refresh_contact', proposed:true}) "
                 "WHERE t.examples=['user -pytest_private_refresh_contact-> Ana'] DETACH DELETE t")


def test_private_refresh_defers_destructive_reconciliation(store, monkeypatch):
    ep = source_episode(store)
    store.apply(Extraction(entities=[EntityOut(name="Boston", kind="place")],
                           relations=[RelationOut(source="user", target="Boston", type="lives_in")],
                           memories=[memory("Public appointment", ["user"], kind="plan", due_on="2026-11-09")]), today=DAY)
    plan = next(m for m in store.memories(today=DAY) if m["kind"] == "plan")
    refreshed(monkeypatch, Extraction(entities=[EntityOut(name="Denver", kind="place", new=True)],
        relations=[RelationOut(source="user", target="Denver", type="lives_in"),
                   RelationOut(source="user", target="Boston", type="lives_in", ended=True)],
        invalidate=[plan["id"]], completed_plans=[plan["id"]],
        memories=[memory("Private schedule changed", ["user"], supersedes=[plan["id"]])]))
    graph.episode_pass(store, ep, DAY, refresh=True)
    assert store._q("MATCH (:User {id:$user})-[r:RELATED {type:'lives_in'}]->(e) WHERE r.until IS NULL RETURN e.name AS name") == [{"name": "Boston"}]
    [row] = store._q("MATCH (m:Memory {id:$id}) RETURN m.status AS status, m.invalidated_at AS retired", id=plan["id"])
    assert row == {"status": "open", "retired": None}


def test_unknown_source_privacy_defaults_to_private(store, monkeypatch):
    ep = store.start_episode(on=DAY)
    store.log_turn(ep, store.user_name, "I enjoy photography")
    refreshed(monkeypatch, Extraction(memories=[memory("Pytest enjoys photography", ["user"])]))
    graph.episode_pass(store, ep, DAY, refresh=True)
    [row] = store.memories(today=DAY)
    assert row["private"] is True


def test_refresh_preview_and_apply_share_the_privacy_floor(store, monkeypatch):
    from companion.preview import preview_extraction
    from companion.reextract import reextract
    from test_reextract_preview import database_state
    ep = source_episode(store)
    ex = Extraction(entities=[EntityOut(name="pottery", kind="activity", new=True)],
                    relations=[RelationOut(source="user", target="Ana", type="friend"),
                               RelationOut(source="Ana", target="pottery", type="likes")],
                    memories=[memory("Ana enjoys pottery", ["Ana", "pottery"])])
    refreshed(monkeypatch, ex)
    before = database_state(store.driver)
    with MemoryStore.read_view(store.driver, store.user_name) as view:
        items = preview_extraction(view, ex, ep, DAY, refresh=True)
    new = next(i for i in items if i["category"] == "memory")
    assert new["action"] == "new" and new["details"]["private"] is True
    old_link = next(i for i in items if i["category"] == "relation" and "friend" in i["text"])
    assert old_link["details"]["private_after"] is True
    assert not any(i["action"] == "update" and "shared" in i["reason"] for i in items)
    log = "\n".join(reextract(store.driver, store.user_name, dry_run=True))
    assert "privacy floor" in log and "private to" in log
    assert "would change visibility from private to shared" not in log
    assert database_state(store.driver) == before
    graph.episode_pass(store, ep, DAY, refresh=True)
    assert all(m["private"] for m in store.memories(today=DAY))


def test_private_refresh_defers_forgetting(store, monkeypatch):
    ep = source_episode(store)
    refreshed(monkeypatch, Extraction(forget_entities=["Ana"]))
    log = graph.episode_pass(store, ep, DAY, refresh=True)
    assert store._q("MATCH (e:Entity {user:$user, name:'Ana'}) RETURN e.id AS id")
    assert any("deferred forgetting" in line for line in log)


def test_private_evidence_from_another_episode_protects_new_links(store, monkeypatch):
    source_episode(store)
    ep = store.start_episode(on=DAY)
    store.log_turn(ep, store.user_name, "We discussed weekend activities")
    store.apply(Extraction(memories=[memory("Pytest discussed weekend activities", ["user"])]), ep, DAY)
    refreshed(monkeypatch, Extraction(entities=[EntityOut(name="pottery", kind="activity", new=True)],
                                      relations=[RelationOut(source="Ana", target="pottery", type="likes")],
                                      memories=[memory("Ana enjoys pottery", ["Ana", "pottery"])]))
    graph.episode_pass(store, ep, DAY, refresh=True)
    [row] = store._q("MATCH (:Entity {user:$user, name:'Ana'})-[r:RELATED {type:'likes'}]->() RETURN r.private AS private")
    assert row["private"] is True


def test_unrelated_private_material_does_not_privatize_public_refresh(store, monkeypatch):
    source_episode(store)
    ep = store.start_episode(on=DAY)
    store.log_turn(ep, store.user_name, "Bob is my coworker")
    store.apply(Extraction(entities=[EntityOut(name="Bob", kind="person", new=True)],
                           memories=[memory("Bob is Pytest's coworker", ["Bob"])]), ep, DAY)
    refreshed(monkeypatch, Extraction(entities=[EntityOut(name="chess", kind="activity", new=True)],
                                      relations=[RelationOut(source="Bob", target="chess", type="likes")],
                                      memories=[memory("Bob enjoys chess", ["Bob", "chess"])]))
    graph.episode_pass(store, ep, DAY, refresh=True)
    other = MemoryStore(store.driver, store.user_name, agent="Other")
    assert any("Bob enjoys chess" == m["text"] for m in other.memories(today=DAY))
    assert not any("Ana" in m["text"] for m in other.memories(today=DAY))


def test_legacy_globally_visible_memory_is_not_enriched_with_private_about(store, monkeypatch):
    ep = source_episode(store)
    store._q("MATCH (m:Memory {user:$user}) REMOVE m.agent")
    [row] = store.memories(today=DAY)
    refreshed(monkeypatch, Extraction(entities=[EntityOut(name="Private archive", kind="thing", new=True)],
                                      memories=[memory(row["text"], ["user", "Ana", "Private archive"])]))
    graph.episode_pass(store, ep, DAY, refresh=True)
    assert store._q("MATCH (m:Memory {user:$user})-[:ABOUT]->(:Entity {name:'Private archive'}) RETURN m") == []
    other = MemoryStore(store.driver, store.user_name, agent="Other")
    assert not any("Private archive" in e for e in other.known_entities())


def test_refresh_uses_source_persona_for_private_additions(store, monkeypatch):
    coach = MemoryStore(store.driver, store.user_name, agent="Coach")
    ep = source_episode(coach)
    refreshed(monkeypatch, Extraction(entities=[EntityOut(name="Secret hobby", kind="activity", new=True)],
                                      memories=[memory("Ana has a secret hobby", ["Ana", "Secret hobby"])]))
    graph.episode_pass(coach, ep, DAY, refresh=True)
    assert any(m["text"] == "Ana has a secret hobby" for m in coach.memories(today=DAY))
    assert store.memories(today=DAY) == []
    assert not any("Secret hobby" in e for e in store.known_entities())


def test_refresh_does_not_bind_other_personas_hidden_entity_metadata(store, monkeypatch):
    source_episode(store)
    [original] = store._q("MATCH (e:Entity {user:$user, name:'Ana'}) SET e.summary='private clinical details' RETURN e.id AS id")
    other = MemoryStore(store.driver, store.user_name, agent="Other")
    ep = other.start_episode(on=DAY)
    other.log_turn(ep, other.user_name, "I discussed work")
    other.apply(Extraction(memories=[memory("Pytest discussed work", ["user"])]), ep, DAY)
    refreshed(monkeypatch, Extraction(entities=[EntityOut(name="Ana", kind="person")],
                                      memories=[memory("Ana enjoys hiking", ["Ana"])]))
    graph.episode_pass(other, ep, DAY, refresh=True)
    assert not any("private clinical details" in e or original["id"] in e for e in other.known_entities())


def test_refresh_preview_shows_stored_privacy_for_reinforced_memory(store):
    from companion.preview import preview_extraction
    ep = source_episode(store)
    ex = Extraction(memories=[memory("Pytest's friend Ana works as a nurse", ["user", "Ana"])])
    with MemoryStore.read_view(store.driver, store.user_name) as view:
        item = next(i for i in preview_extraction(view, ex, ep, DAY, refresh=True) if i["category"] == "memory")
    assert item["action"] == "reinforce"
    assert item["details"]["private"] is True
    assert item["details"]["private_before"] is True and item["details"]["private_after"] is True
    assert item["details"]["incoming_private"] is False


def test_refresh_scope_restores_normal_matching_after_failure(store, monkeypatch):
    other = MemoryStore(store.driver, store.user_name, agent="Other")
    source_episode(other)
    [entity] = store._q("MATCH (e:Entity {user:$user, name:'Ana'}) RETURN e.id AS id")
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic refresh failure")
    monkeypatch.setattr(store, "_apply", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        store.apply(Extraction(), refresh=True)
    # The existing normal-store matching contract remains unchanged after the failure.
    assert store.resolve_entity("Ana", "person") == entity["id"]
