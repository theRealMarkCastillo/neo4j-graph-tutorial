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



def test_a_plan_the_same_extraction_restates_stays_open(store):
    # Seen in MLflow traces (gemini-3.5-flash-lite): the episode pass re-stated the dentist plan
    # AND listed it as completed. Saying it is still on wins over the contradiction.
    store.apply(Extraction(memories=[mem("Dentist appointment on 2026-11-17", "plan", due_on="2026-11-17")]), today=T)
    [plan] = store.memories(kind="plan")
    for restated in (mem("Dentist appointment on 2026-11-17", "plan", due_on="2026-11-17"),
                     mem("Dentist appointment on 2026-11-17", "plan", due_on="2026-11-17", supersedes=[plan["id"]])):
        log = store.apply(Extraction(memories=[restated], completed_plans=[plan["id"]]), today=T)
        assert f"plan done {plan['id']}" not in log
        assert [m["status"] for m in store.memories(kind="plan")] == ["open"], log


def test_a_plan_can_still_be_completed(store):
    store.apply(Extraction(memories=[mem("Dentist appointment on 2026-11-17", "plan", due_on="2026-11-17")]), today=T)
    [plan] = store.memories(kind="plan")
    log = store.apply(Extraction(memories=[mem("Pytest went to the dentist on 2026-11-17", "event",
                                               occurred_on="2026-11-17")],
                                 completed_plans=[plan["id"]]), today=T)
    assert f"plan done {plan['id']}" in log


def test_another_plan_on_the_same_day_does_not_block_completion(store):
    store.apply(Extraction(memories=[mem("Job interview at Acme on 2026-11-17", "plan", due_on="2026-11-17")]), today=T)
    [plan] = store.memories(kind="plan")
    log = store.apply(Extraction(memories=[mem("Dentist appointment on 2026-11-17", "plan", due_on="2026-11-17")],
                                 completed_plans=[plan["id"]]), today=T)
    assert f"plan done {plan['id']}" in log

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


def entities(store):
    return {r["name"]: r for r in store._q(
        "MATCH (e:Entity {user: $user}) WHERE NOT e:User RETURN e.id AS id, e.name AS name, e.aliases AS aliases")}


def test_a_placeholder_is_renamed_when_the_name_is_learned(store):
    """"My son is sick" -> "Pytest's son"; later "his name is Logan" must rename that node,
    not leave the name only inside a memory (a real conversation found this)."""
    store.apply(Extraction(entities=[EntityOut(name="Pytest's son", kind="person")],
                           relations=[RelationOut(source="user", target="Pytest's son", type="son")]), today=T)
    son = entities(store)["Pytest's son"]["id"]
    log = store.apply(Extraction(entities=[EntityOut(name="Logan", kind="person", ref=son,
                                                     aliases=["Pytest's son"])]), today=T)
    assert log == ["named Pytest's son -> Logan (Pytest's son kept as an alias)"]
    after = entities(store)
    assert list(after) == ["Logan"] and after["Logan"]["id"] == son  # same node: its links stay
    assert after["Logan"]["aliases"] == ["Pytest's son"]
    assert store._lookup("Pytest's son", {}) == son  # the old name still finds him


def test_a_placeholder_is_renamed_even_without_a_ref(store):
    """The extractor may skip `ref` and just list the role name as an alias."""
    store.apply(Extraction(entities=[EntityOut(name="Mom", kind="person")]), today=T)
    store.apply(Extraction(entities=[EntityOut(name="Linda", kind="person", aliases=["Mom"])]), today=T)
    assert {n: e["aliases"] for n, e in entities(store).items()} == {"Linda": ["Mom"]}


def test_a_real_name_is_never_replaced(store):
    """A role ("Dad") or a nickname ("Mike") for a named person becomes an alias only.
    (Not "Bob" for "Robert": Qwen scores that pair just under its name cut, a known miss
    listed in test_19's KNOWN_MISSES. Mike ~ Michael clears it on both models.)"""
    store.apply(Extraction(entities=[EntityOut(name="Michael", kind="person")]), today=T)
    mike = entities(store)["Michael"]["id"]
    store.apply(Extraction(entities=[EntityOut(name="Dad", kind="person", ref=mike)]), today=T)
    store.apply(Extraction(entities=[EntityOut(name="Mike", kind="person", ref=mike)]), today=T)
    assert {n: sorted(e["aliases"]) for n, e in entities(store).items()} == {"Michael": ["Dad", "Mike"]}


@pytest.mark.parametrize("name,role", [("Mom", True), ("my boss", True), ("Mark's son", True),
                                       ("Riley’s sister", True), ("Logan", False),
                                       ("St. Mary's Hospital", False), ("my sister Maya", False)])
def test_what_counts_as_a_role_name(name, role):
    from companion.store import is_role_name
    assert is_role_name(name) is role


# A code review after the Logan bug found four more ways the graph lost track of who's who.
# Each test below failed before its fix.

def test_a_vague_kind_is_upgraded(store):
    """Acme saved as a 'thing' could never be worked at: works_at needs an organization."""
    store.apply(Extraction(entities=[EntityOut(name="Acme", kind="thing")]), today=T)
    log = store.apply(Extraction(entities=[EntityOut(name="Acme", kind="organization")],
                                 relations=[RelationOut(source="user", target="Acme", type="works_at")]), today=T)
    assert log == ["Acme is a organization (was thing)", "linked user -works_at-> Acme"]
    store.apply(Extraction(entities=[EntityOut(name="Acme", kind="thing")]), today=T)  # never back down
    assert store._q("MATCH (e:Entity {user: $user, name: 'Acme'}) RETURN e.kind AS k")[0]["k"] == "organization"


def test_a_corrected_name_replaces_the_wrong_one(store):
    store.apply(Extraction(entities=[EntityOut(name="Maya", kind="person")]), today=T)
    maya = entities(store)["Maya"]["id"]
    log = store.apply(Extraction(entities=[EntityOut(name="Maia", kind="person", ref=maya, corrected=True)]), today=T)
    assert log == ["named Maya -> Maia (name corrected)"]
    assert {n: e["aliases"] for n, e in entities(store).items()} == {"Maia": ["Maya"]}


def test_a_placeholder_and_a_named_duplicate_are_merged(store):
    """Both nodes existed; then the extractor said they're one person."""
    store.apply(Extraction(entities=[EntityOut(name="Pytest's sister", kind="person")],
                           relations=[RelationOut(source="user", target="Pytest's sister", type="sister")],
                           memories=[mem("Pytest's sister is a nurse", about=["Pytest's sister"])]), today=T)
    store.apply(Extraction(entities=[EntityOut(name="Maya", kind="person")],
                           relations=[RelationOut(source="user", target="Maya", type="sister")]), today=T)
    log = store.apply(Extraction(entities=[EntityOut(name="Maya", kind="person", aliases=["Pytest's sister"])]), today=T)
    assert log == ["merged Pytest's sister into Maya (the same person)"]
    assert {n: e["aliases"] for n, e in entities(store).items()} == {"Maya": ["Pytest's sister"]}
    rels = store._q("MATCH (:User {id: $user})-[r:RELATED]->(e) RETURN r.type AS t, e.name AS n")
    assert rels == [{"t": "sister", "n": "Maya"}]  # one edge, not two
    about = store._q("MATCH (m:Memory)-[:ABOUT]->(e:Entity {user: $user, name: 'Maya'}) RETURN m.text AS t")
    assert [a["t"] for a in about] == ["Pytest's sister is a nurse"]  # the memory moved with her


def test_one_mother_means_the_placeholder_is_her(store):
    """'Mom' first, then 'my mother Linda' with no alias: one mother, so one node."""
    store.apply(Extraction(entities=[EntityOut(name="Mom", kind="person")],
                           relations=[RelationOut(source="user", target="Mom", type="mom")],
                           memories=[mem("Pytest's mom loves gardening", kind="preference", about=["Mom"])]), today=T)
    log = store.apply(Extraction(entities=[EntityOut(name="Linda", kind="person")],
                                 relations=[RelationOut(source="user", target="Linda", type="mother")]), today=T)
    assert "merged Mom into Linda (one mother)" in log
    assert {n: e["aliases"] for n, e in entities(store).items()} == {"Linda": ["Mom"]}
    about = store._q("MATCH (m:Memory)-[:ABOUT]->(e:Entity {user: $user, name: 'Linda'}) RETURN m.text AS t")
    assert [a["t"] for a in about] == ["Pytest's mom loves gardening"]


def test_siblings_and_stepparents_are_never_merged_by_role(store):
    """Two sisters, or a mom and a stepmom, are different people."""
    store.apply(Extraction(entities=[EntityOut(name="Pytest's sister", kind="person"), EntityOut(name="Mom", kind="person")],
                           relations=[RelationOut(source="user", target="Pytest's sister", type="sister"),
                                      RelationOut(source="user", target="Mom", type="mother")]), today=T)
    store.apply(Extraction(entities=[EntityOut(name="Maya", kind="person"), EntityOut(name="Carol", kind="person")],
                           relations=[RelationOut(source="user", target="Maya", type="sister"),
                                      RelationOut(source="user", target="Carol", type="stepmom")]), today=T)
    assert set(entities(store)) == {"Pytest's sister", "Mom", "Maya", "Carol"}


def test_an_unrelated_name_from_a_wrong_ref_is_not_kept(store):
    """A wrong `ref` (Jordan -> Sam) must not make 'Jordan' a permanent alias of Sam."""
    store.apply(Extraction(entities=[EntityOut(name="Sam", kind="person", summary="coworker")]), today=T)
    sam = entities(store)["Sam"]["id"]
    log = store.apply(Extraction(entities=[EntityOut(name="Jordan", kind="person", ref=sam)]), today=T)
    assert log == ["kept Sam: 'Jordan' matched it by id but isn't one of its names (not added as an alias)"]
    assert entities(store)["Sam"]["aliases"] == [] and store._lookup("Jordan", {}) is None


# ---- the life graph beyond the user (Logan -likes-> Minecraft)

def links(store):
    return {(r["a"], r["type"], r["b"]) for r in store._q(
        "MATCH (a:Entity {user: $user})-[r:RELATED]->(b:Entity) WHERE r.until IS NULL "
        "RETURN CASE WHEN a:User THEN 'user' ELSE a.name END AS a, r.type AS type, b.name AS b")}


def logan_likes_minecraft(store, ep=None):
    return store.apply(Extraction(
        entities=[EntityOut(name="Logan", kind="person"), EntityOut(name="Minecraft", kind="thing")],
        relations=[RelationOut(source="user", target="Logan", type="son"),
                   RelationOut(source="Logan", target="Minecraft", type="likes")],
        memories=[mem("Pytest has a son named Logan", about=["user", "Logan"]),
                  mem("Pytest's son Logan loves Minecraft", "preference", about=["Logan", "Minecraft"])],
    ), ep, T)


def test_a_thing_and_an_activity_with_close_names_are_one_entity(store):
    a = store.resolve_entity("Minecraft", "thing", summary="a video game")
    b = store.resolve_entity("Minecraft game", "activity", summary="a video game")
    assert a == b


def test_dislikes_ends_likes_of_the_same_thing(store):
    logan_likes_minecraft(store)
    log = store.apply(Extraction(
        relations=[RelationOut(source="Logan", target="Minecraft", type="dislikes")],
        memories=[mem("Pytest's son Logan is bored of Minecraft", "preference", about=["Logan", "Minecraft"])],
    ), None, T)
    assert ("Logan", "dislikes", "Minecraft") in links(store)
    assert ("Logan", "likes", "Minecraft") not in links(store), log


def test_a_link_ends_when_its_only_memory_is_retired(store):
    logan_likes_minecraft(store)
    [like] = [m for m in store.memories() if "Minecraft" in m["text"]]
    log = store.apply(Extraction(invalidate=[like["id"]]), None, T)
    assert ("Logan", "likes", "Minecraft") not in links(store), log
    assert ("user", "son", "Logan") in links(store)  # still backed by its own memory


def test_a_link_survives_when_its_memory_is_replaced_by_one_about_the_same_thing(store):
    logan_likes_minecraft(store)
    [like] = [m for m in store.memories() if "Minecraft" in m["text"]]
    store.apply(Extraction(memories=[mem("Pytest's son Logan plays Minecraft every weekend", "preference",
                                         about=["Logan", "Minecraft"], supersedes=[like["id"]])]), None, T)
    assert ("Logan", "likes", "Minecraft") in links(store)


def test_forgetting_a_memory_ends_the_link_it_alone_supported(store):
    logan_likes_minecraft(store)
    [like] = [m for m in store.memories() if "Minecraft" in m["text"]]
    assert store.forget(like["id"], today=T)
    assert ("Logan", "likes", "Minecraft") not in links(store)


def test_consolidate_drops_entities_nothing_points_at(store):
    logan_likes_minecraft(store)
    store.resolve_entity("Disney+", "organization")  # named, never linked or remembered
    log = store.consolidate(today=T)
    names = {r["n"] for r in store._q("MATCH (e:Entity {user: $user}) WHERE NOT e:User RETURN e.name AS n")}
    assert names == {"Logan", "Minecraft"}, log


def test_a_repeated_memory_gains_the_entities_it_names(store):
    """A re-read under new rules says the same memory again, about more entities."""
    store.apply(Extraction(entities=[EntityOut(name="Logan", kind="person")],
                           memories=[mem("Pytest's son Logan loves Minecraft", "preference", about=["Logan"])]), None, T)
    store.apply(Extraction(entities=[EntityOut(name="Logan", kind="person"), EntityOut(name="Minecraft", kind="thing")],
                           memories=[mem("Pytest's son Logan loves Minecraft", "preference",
                                         about=["Logan", "Minecraft"])]), None, T)
    [m] = store.memories()
    assert sorted(m["about"]) == ["Logan", "Minecraft"]
