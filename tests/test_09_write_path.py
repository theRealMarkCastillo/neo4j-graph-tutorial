"""Lesson 9 — the memory write path, WITHOUT the LLM.

The extractor (LLM) and the store (code) meet at one typed object: `Extraction`. Hand-write
the Extraction a model *would* return and you can test every storage rule exactly and for
free. The LLM half is graded separately in test_11_extraction_evals.py.
"""
import datetime as dt

import pytest
from pydantic import ValidationError

from companion.models import EntityOut, Extraction, MemoryOut, RelationOut

T = dt.date(2026, 11, 5)


def mem(text, kind="fact", **kw):
    return MemoryOut(text=text, kind=kind, importance=kw.pop("importance", 6), **kw)


def current(store):
    return {m["text"] for m in store.memories()}


def test_schema_rejects_bad_llm_output():
    # Validation is the first eval: a malformed answer fails loudly instead of corrupting data.
    with pytest.raises(ValidationError):
        MemoryOut(text="x", kind="gossip", importance=5)
    with pytest.raises(ValidationError):
        MemoryOut(text="x", kind="fact", importance=11)


def test_entities_relations_and_memories_are_linked(store):
    log = store.apply(Extraction(
        entities=[EntityOut(name="Maya", kind="person", aliases=["my sister"])],
        relations=[RelationOut(source="user", target="Maya", type="sister")],
        memories=[mem("Pytest's sister Maya lives in Denver", about=["Maya"])],
    ), today=T)
    assert "linked user -sister-> Maya" in log
    about = store._q("""MATCH (m:Memory {user:$user})-[:ABOUT]->(e:Entity {name:'Maya'})
                        RETURN m.text AS text""")
    assert [a["text"] for a in about] == ["Pytest's sister Maya lives in Denver"]


def test_aliases_resolve_to_the_same_entity(store):
    store.apply(Extraction(entities=[EntityOut(name="Maya", kind="person", aliases=["my sister"])]), today=T)
    store.apply(Extraction(entities=[EntityOut(name="my sister", kind="person")]), today=T)
    assert len(store._q("MATCH (e:Entity {user:$user, kind:'person'}) WHERE NOT e:User RETURN e")) == 1


def test_duplicate_memory_is_reinforced_not_copied(store):
    ex = Extraction(memories=[mem("Pytest loves rock climbing", "preference")])
    store.apply(ex, today=T)
    log = store.apply(ex, today=T)
    assert log[0].startswith("reinforced")
    assert len(store.memories()) == 1


def test_same_text_different_date_is_not_a_duplicate(store):
    store.apply(Extraction(memories=[mem("Dentist appointment", "plan", due_on="2026-11-12")]), today=T)
    store.apply(Extraction(memories=[mem("Dentist appointment", "plan", due_on="2026-11-17")]), today=T)
    assert len(store.memories()) == 2


def test_supersedes_retires_the_old_fact_and_keeps_history(store):
    store.apply(Extraction(memories=[mem("Pytest lives in Austin")]), today=T)
    [old] = store.memories()
    store.apply(Extraction(memories=[mem("Pytest lives in Chicago", supersedes=[old["id"]])]), today=T)
    assert current(store) == {"Pytest lives in Chicago"}
    history = store._q("""MATCH (:Memory {user:$user})-[:SUPERSEDES]->(o:Memory)
                          RETURN o.text AS text, o.invalidated_at IS NOT NULL AS retired""")
    assert history == [{"text": "Pytest lives in Austin", "retired": True}]


def test_ids_from_another_user_are_ignored(store, driver):
    # The extractor only ever sees this user's ids, but the store must not trust that.
    driver.execute_query("CREATE (:Memory {id:'m_other', user:'user:someone-else', text:'secret'})")
    try:
        store.apply(Extraction(invalidate=["m_other"],
                               memories=[mem("Pytest fact", supersedes=["m_other"])]), today=T)
        [r] = driver.execute_query("MATCH (m:Memory {id:'m_other'}) RETURN m.invalidated_at AS at").records
        assert r["at"] is None
    finally:
        driver.execute_query("MATCH (m:Memory {id:'m_other'}) DETACH DELETE m")


def test_plans_get_an_open_status(store):
    store.apply(Extraction(memories=[mem("Job interview at Acme", "plan", due_on="2026-11-09")]), today=T)
    [m] = store._q("MATCH (m:Memory {user:$user, kind:'plan'}) RETURN m.status AS s, toString(m.due_on) AS due")
    assert m == {"s": "open", "due": "2026-11-09"}


def test_forget_entity_removes_memories_and_redacts_turns(store):
    ep = store.start_episode(on=T)
    store.log_turn(ep, "user", "My ex Jordan texted me")
    store.apply(Extraction(entities=[EntityOut(name="Jordan", kind="person")],
                           relations=[RelationOut(source="user", target="Jordan", type="ex")],
                           memories=[mem("Jordan texted Pytest", "event", about=["Jordan"])]), ep, T)
    log = store.apply(Extraction(forget_entities=["Jordan"]), ep, T)
    assert any(line.startswith("forgot Jordan") for line in log)
    leftovers = store._q("""
        MATCH (n) WHERE n.user = $user AND (n:Memory OR n:Entity)
          AND toLower(coalesce(n.text, n.name)) CONTAINS 'jordan' RETURN n""")
    assert leftovers == []
    turns = store._q("MATCH (:Episode {id:$ep})-[:HAS_TURN]->(t) RETURN t.text AS text", ep=ep)
    assert all("Jordan" not in t["text"] for t in turns)


# --- regressions from the code review (each failed before its fix) -----------------------

def test_reinforcing_a_duplicate_still_applies_supersedes(store):
    store.apply(Extraction(memories=[mem("Pytest's brother is named Leo"),
                                     mem("Pytest's brother is named Theo")]), today=T)
    wrong = next(m for m in store.memories() if "Theo" in m["text"])
    # The correction matches an existing memory exactly: reinforce it AND retire the wrong one.
    store.apply(Extraction(memories=[mem("Pytest's brother is named Leo", supersedes=[wrong["id"]])]), today=T)
    assert current(store) == {"Pytest's brother is named Leo"}


def test_forget_entity_matches_whole_words_only(store):
    store.apply(Extraction(entities=[EntityOut(name="Ann", kind="person")],
                           memories=[mem("Ann is Pytest's coworker", about=["Ann"]),
                                     mem("Pytest is planning an anniversary trip")]), today=T)
    store.apply(Extraction(forget_entities=["Ann"]), today=T)
    assert current(store) == {"Pytest is planning an anniversary trip"}


def test_forget_entity_redacts_short_names_and_other_summaries(store):
    ep = store.start_episode(on=T)
    store.log_turn(ep, "user", "Jo is my ex")
    store.apply(Extraction(entities=[EntityOut(name="Jo", kind="person"),
                                     EntityOut(name="Kai", kind="person", summary="Jo's brother")]), ep, T)
    store.apply(Extraction(forget_entities=["Jo"]), ep, T)
    [turn] = store._q("MATCH (:Episode {id:$ep})-[:HAS_TURN]->(t) RETURN t.text AS text", ep=ep)
    assert "Jo" not in turn["text"].replace("[forgotten]", "")
    [kai] = store._q("MATCH (e:Entity {user:$user, name:'Kai'}) RETURN e.summary AS s")
    assert "Jo" not in kai["s"].replace("[forgotten]", "")


def test_forgotten_memory_is_not_re_extracted_in_the_same_episode(store):
    ep = store.start_episode(on=T)
    store.apply(Extraction(memories=[mem("Pytest owes Sam 200 dollars")]), ep, T)
    [m] = store.memories()
    assert store.forget(m["id"])
    # The end-of-episode pass re-reads the transcript and finds the same fact again:
    log = store.apply(Extraction(memories=[mem("Pytest owes Sam $200")]), ep, T)
    assert store.memories() == [], log


def test_someone_elses_move_does_not_move_the_user(store):
    store.apply(Extraction(entities=[EntityOut(name="Boston", kind="place"), EntityOut(name="Maya", kind="person")],
                           relations=[RelationOut(source="user", target="Boston", type="lives_in")],
                           memories=[mem("Maya also lives in Boston", about=["Maya", "Boston"])]), today=T)
    [old] = [m for m in store.memories() if "Maya" in m["text"]]
    store.apply(Extraction(entities=[EntityOut(name="Denver", kind="place")],
                           memories=[mem("Maya moved to Denver", about=["Maya", "Denver"], supersedes=[old["id"]])]), today=T)
    homes = store._q("MATCH (:User {id:$user})-[r:RELATED {type:'lives_in'}]->(e) WHERE r.until IS NULL RETURN e.name AS n")
    assert homes == [{"n": "Boston"}]


def test_episode_whose_summary_failed_is_still_closed(store):
    ep = store.start_episode(on=T)
    store.log_turn(ep, "user", "hello there")
    store.end_episode(ep, None, None)
    [r] = store._q("MATCH (e:Episode {id:$ep}) RETURN e.ended_at IS NOT NULL AS closed", ep=ep)
    assert r["closed"]
    empty = store.start_episode(on=T)
    store.end_episode(empty, None, None)
    assert store._q("MATCH (e:Episode {id:$ep}) RETURN e", ep=empty) == []
