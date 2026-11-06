"""Lesson 13 — memory over time: validity, decay, consolidation.

Time makes tests flaky unless the clock is an input. Every function here takes `today`
(clock.now_for turns it into "now"), so a test can store a memory on day 1 and recall on
day 30 in the same second, with no sleeping and no mocking of datetime.
"""
import datetime as dt

import pytest

from companion import clock, decay
from companion.models import Extraction, MemoryOut
from companion.embeddings import embed_for_clustering
from companion.store import PATTERN_SIMILARITY, _clusters, format_context

D1 = dt.date(2026, 11, 18)      # noon on a Wednesday, per clock.now_for
NOON = clock.now_for(D1)


def mem(text, kind="event", importance=3, **kw):
    return MemoryOut(text=text, kind=kind, importance=importance, **kw)


# --- clock (unit) --------------------------------------------------------------------------

def test_simulated_days_are_noon_local():
    assert (NOON.hour, NOON.minute) == (12, 0) and NOON.tzinfo is not None


@pytest.mark.parametrize("value,kind,expected", [
    ("2026-11-18T14:00", "status", (2026, 11, 18, 14, 0)),
    ("2026-11-18", "feeling", (2026, 11, 18, 23, 59)),      # a date: the end of that day
    (None, "status", (2026, 11, 18, 15, 0)),                 # a status always ends (3 h default)
    ("2026-11-18T09:00", "status", (2026, 11, 18, 15, 0)),   # already past: default instead
    ("whenever", "status", (2026, 11, 18, 15, 0)),           # unreadable: default
    (None, "fact", None),                                    # lasting things never expire
])
def test_parse_until(value, kind, expected):
    got = clock.parse_until(value, NOON, kind)
    assert (got and (got.year, got.month, got.day, got.hour, got.minute)) == expected


# --- decay (unit) --------------------------------------------------------------------------

def test_strength_halves_every_half_life():
    assert decay.strength(4, "feeling", 0) == pytest.approx(0.4)
    assert decay.strength(4, "feeling", 7) == pytest.approx(0.2)
    assert decay.strength(4, "feeling", 14) == pytest.approx(0.1)


def test_facts_do_not_decay_they_go_stale():
    assert decay.strength(7, "fact", 3000) == pytest.approx(0.7)
    assert not decay.is_dormant(2, "fact", 3000)
    assert decay.is_stale("fact", 400) and not decay.is_stale("fact", 30)


def test_dormancy():
    assert not decay.is_dormant(3, "feeling", 7)
    assert decay.is_dormant(3, "feeling", 30)             # a month-old minor mood
    assert not decay.is_dormant(9, "event", 3000)         # milestones never go dormant
    assert decay.days_until_dormant(2, "feeling") == pytest.approx(14, abs=0.1)


def test_a_strong_old_memory_can_still_outrank_a_weak_new_one():
    # ...but only if it matches the message better: strength is a tiebreaker, not a filter.
    old_close = decay.rank(0.90, 6, "event", 90)
    new_vague = decay.rank(0.70, 3, "event", 0)
    assert old_close > new_vague


# --- validity through the store (Neo4j, no LLM) -------------------------------------------

def recalled(store, query, day):
    r = store.recall(query, today=day)
    return ([m["text"] for m in r["semantic"]], [m["text"] for m in r["recent"]],
            format_context(r, day))


def test_a_status_is_current_then_recent_then_gone(store):
    log = store.apply(Extraction(memories=[mem("Pytest is at the gym", "status", 2,
                                               valid_until="2026-11-18T14:00")]), today=D1)
    assert "(until 2026-11-18 14:00)" in log[0]

    current, recent, _ = recalled(store, "what am I doing?", D1)                    # 12:00
    assert "Pytest is at the gym" in current and recent == []

    current, recent, block = recalled(store, "what am I doing?", D1 + dt.timedelta(days=1))
    assert "Pytest is at the gym" not in current
    assert recent == ["Pytest is at the gym"]                                      # ask how it went
    assert "Recently, no longer true" in block

    current, recent, _ = recalled(store, "what am I doing?", D1 + dt.timedelta(days=3))
    assert "Pytest is at the gym" not in current + recent


def test_memories_is_an_as_of_query(store):
    # Valid time vs transaction time: a status stored on day 3 was never "current" on day 1,
    # even though its valid_until is later than day 1.
    store.apply(Extraction(memories=[mem("Pytest is in Denver", "status", 3,
                                         valid_until="2026-11-25")]), today=D1 + dt.timedelta(days=2))
    assert store.memories(today=D1) == []
    assert len(store.memories(today=D1 + dt.timedelta(days=3))) == 1


def test_expired_memories_are_not_listed_as_current(store):
    store.apply(Extraction(memories=[mem("Pytest is at the airport", "status", 2,
                                         valid_until="2026-11-18T15:00")]), today=D1)
    assert len(store.memories(today=D1)) == 1
    assert store.memories(today=D1 + dt.timedelta(days=1)) == []
    assert len(store.memories(today=D1 + dt.timedelta(days=1), include_retired=True)) == 1
    assert len(store.memories()) == 1  # no `today`: no as-of cut (real today is before D1)


def test_a_trip_does_not_end_the_real_home(store):
    from companion.models import EntityOut, RelationOut
    store.apply(Extraction(entities=[EntityOut(name="Chicago", kind="place")],
                           relations=[RelationOut(source="user", target="Chicago", type="lives_in")]), today=D1)
    log = store.apply(Extraction(
        entities=[EntityOut(name="Denver", kind="place")],
        relations=[RelationOut(source="user", target="Denver", type="lives_in")],  # the LLM's mistake
        memories=[mem("Pytest is in Denver for work", "status", 3, about=["user", "Denver"])]), today=D1)
    assert any(line.startswith("ignored") for line in log)
    homes = store._q("MATCH (:User {id:$user})-[r:RELATED {type:'lives_in'}]->(e) WHERE r.until IS NULL RETURN e.name AS n")
    assert homes == [{"n": "Chicago"}]


def test_saying_a_status_again_on_another_day_is_new(store):
    s = mem("Pytest is at the gym", "status", 2)
    store.apply(Extraction(memories=[s]), today=D1)
    log = store.apply(Extraction(memories=[s]), today=D1 + dt.timedelta(days=2))
    assert log[0].startswith("+ [status]"), log    # the old one expired: not a duplicate


def test_a_one_day_boundary_expires(store):
    store.apply(Extraction(memories=[
        mem("Pytest's job interview went badly", "event", 5),
        mem("Pytest doesn't want to talk about work today", "boundary", 9,
            about=["user"], valid_until="2026-11-18")]), today=D1)
    core_today = [m["text"] for m in store.recall("hi", today=D1)["core"]]
    core_later = [m["text"] for m in store.recall("hi", today=D1 + dt.timedelta(days=1))["core"]]
    assert any("work today" in t for t in core_today)
    assert not any("work today" in t for t in core_later)


# --- decay through the store ----------------------------------------------------------------

def test_old_minor_feelings_go_dormant(store):
    store.apply(Extraction(memories=[mem("Pytest felt grumpy about the rain", "feeling", 3)]), today=D1)
    assert recalled(store, "how was I feeling about the rain?", D1 + dt.timedelta(days=3))[0]
    assert not recalled(store, "how was I feeling about the rain?", D1 + dt.timedelta(days=40))[0]


def test_confirming_resets_the_clock(store):
    f = mem("Pytest felt grumpy about the rain", "feeling", 3)
    store.apply(Extraction(memories=[f]), today=D1)
    store.apply(Extraction(memories=[f]), today=D1 + dt.timedelta(days=25))   # said again: reinforced
    assert recalled(store, "how was I feeling about the rain?", D1 + dt.timedelta(days=40))[0]


def test_old_facts_are_marked_stale_not_dropped(store):
    store.apply(Extraction(memories=[mem("Pytest works at Acme", "fact", 7)]), today=D1)
    block = recalled(store, "how is work?", D1 + dt.timedelta(days=400))[2]
    assert "Pytest works at Acme (not confirmed in over a year" in block


# --- consolidation ---------------------------------------------------------------------------

def fake_writer(calls):
    def write(user, texts, today):
        calls.append(texts)
        return MemoryOut(text="Pytest works out at the gym regularly", kind="preference", importance=6)
    return write


def test_clustering_finds_the_group_an_early_pair_would_hide():
    """Regression from the scenario run. In time order, "heading" and "just got back" pair up
    first (0.86); "after work" (0.80 to "got back") and "before work" (0.81) can't join, so
    first-fit made two pairs. heading / after work / before work are all >= 0.85 to each
    other: that's the group of three to find."""
    texts = ["Riley is heading to the gym",
             "Riley just got back from the gym.",
             "Riley's legs feel exhausted after a gym session.",
             "Riley is going to the gym after work and then having dinner",
             "Riley is at the gym before work."]
    rows = [{"text": t, "vec": v} for t, v in zip(texts, embed_for_clustering(texts), strict=True)]
    groups = _clusters(rows, PATTERN_SIMILARITY, 3)
    # One group with those three. A stronger model (Lesson 19) may also see "just got back"
    # and "legs exhausted" as the same habit, which is right too: the regression was TWO pairs.
    assert len(groups) == 1 and {texts[0], texts[3], texts[4]} <= {r["text"] for r in groups[0]}


def test_clustering_does_not_chain():
    """a~b and b~c, but a is not like c: complete linkage keeps a group of three from forming."""
    a, b, c = [1.0, 0.0], [0.8, 0.6], [0.28, 0.96]   # a.b = 0.8, b.c = 0.8, a.c = 0.28
    rows = [{"text": n, "vec": v} for n, v in zip("abc", (a, b, c))]
    assert _clusters(rows, 0.75, 3) == []
    assert len(_clusters(rows, 0.75, 2)) == 1


def test_repeated_statuses_roll_up_into_a_pattern(store):
    for i, text in enumerate(["Pytest is at the gym", "Pytest went to the gym after work",
                              "Pytest is heading to the gym for leg day", "Pytest had pizza for dinner"]):
        store.apply(Extraction(memories=[mem(text, "status", 2)]), today=D1 + dt.timedelta(days=i))
    calls = []
    log = store.consolidate(D1 + dt.timedelta(days=4), writer=fake_writer(calls), prune=False)
    assert len(calls) == 1 and len(calls[0]) == 3 and not any("pizza" in t for t in calls[0])
    assert any("pattern from 3 memories" in line for line in log)
    links = store._q("MATCH (p:Memory {user:$user})-[:SUMMARIZES]->(m) RETURN p.text AS p, count(m) AS n")
    assert links == [{"p": "Pytest works out at the gym regularly", "n": 3}]
    [p] = store._q("MATCH (p:Memory {user:$user}) WHERE p.evidence IS NOT NULL RETURN p.evidence AS n")
    assert p["n"] == 3  # kept on the pattern, so pruning the details later loses nothing
    # running it again finds nothing new: members are marked consolidated
    assert store.consolidate(D1 + dt.timedelta(days=4), writer=fake_writer(calls), prune=False) == []


def test_writer_can_decline(store):
    for i in range(3):
        store.apply(Extraction(memories=[mem("Pytest is at the gym", "status", 2)]), today=D1 + dt.timedelta(days=2 * i))
    assert store.consolidate(D1 + dt.timedelta(days=6), writer=lambda u, t, d: None, prune=False) == []


def test_prune_deletes_only_dormant_trivia(store):
    store.apply(Extraction(memories=[
        mem("Pytest felt grumpy about the rain", "feeling", 2),
        mem("Pytest felt overjoyed at their wedding", "feeling", 9),
        mem("Pytest went to a concert", "event", 2)]), today=D1)
    assert store.consolidate(D1 + dt.timedelta(days=30)) == []  # dormant, but may still form a pattern
    log = store.consolidate(D1 + dt.timedelta(days=61))         # past the pattern window: prune
    assert log == ["pruned 1 dormant status/feeling memories"]
    left = {m["text"] for m in store.memories(include_retired=True, today=D1 + dt.timedelta(days=61))}
    assert left == {"Pytest felt overjoyed at their wedding", "Pytest went to a concert"}


def test_consolidated_details_are_pruned_once_dormant(store):
    for i, text in enumerate(["Pytest is at the gym", "Pytest went to the gym after work",
                              "Pytest is heading to the gym for leg day"]):
        store.apply(Extraction(memories=[mem(text, "status", 2)]), today=D1 + dt.timedelta(days=i))
    log = store.consolidate(D1 + dt.timedelta(days=10), writer=fake_writer([]))
    assert "pruned 3 dormant status/feeling memories" in log
    [p] = store._q("MATCH (p:Memory {user:$user}) WHERE p.evidence IS NOT NULL RETURN p.evidence AS n")
    assert p["n"] == 3


def test_old_open_plans_lapse(store):
    store.apply(Extraction(memories=[mem("Pytest plans to call the bank", "plan", 4, due_on="2026-11-01")]), today=D1)
    assert store.consolidate(D1, prune=False) == ["plan lapsed: Pytest plans to call the bank"]


# --- LLM evals (pytest -m llm) ---------------------------------------------------------------

def extract(msg, reply="Have fun!"):
    from companion.extract import extract_memories
    return extract_memories("Riley", msg, reply, [], [], D1, [])


def ends(ex, kinds=("status", "feeling")):
    return [clock.parse_until(m.valid_until, NOON, m.kind) for m in ex.memories if m.kind in kinds]


@pytest.mark.llm
def test_heading_to_the_gym_is_a_short_status(llm_ready):
    ex = extract("Off to the gym, talk later!")
    dump = ex.model_dump_json(exclude_defaults=True)
    statuses = [m for m in ex.memories if m.kind == "status"]
    assert statuses and all(m.importance <= 4 for m in statuses), dump
    until = clock.parse_until(statuses[0].valid_until, NOON, "status")
    assert NOON < until <= NOON + dt.timedelta(hours=6), dump
    assert not any(m.kind == "event" and m.occurred_on for m in ex.memories), "not a past event: " + dump


@pytest.mark.llm
def test_tired_today_ends_today(llm_ready):
    ex = extract("Ugh, I'm so tired today.", "Sorry to hear that.")
    dump = ex.model_dump_json(exclude_defaults=True)
    found = [u for u in ends(ex) if u]
    assert found and all(u.date() == D1 for u in found), dump


@pytest.mark.llm
def test_trip_for_the_week_ends_within_the_week(llm_ready):
    # The first run added user -lives_in-> Denver, which would end the real home. A prompt
    # rule didn't fix it; an example did, and the store now has a guard as well.
    from conftest import trials

    def check():
        ex = extract("I'm in Denver for the week for a work conference.", "Enjoy Denver!")
        dump = ex.model_dump_json(exclude_defaults=True)
        found = [u for u in ends(ex) if u]
        assert found and all(NOON + dt.timedelta(days=1) < u <= NOON + dt.timedelta(days=8) for u in found), dump
        assert not any(r.type in ("places", "lives_in", "moved_to") for r in ex.relations), "a trip isn't a move: " + dump
    trials(check)


@pytest.mark.llm
def test_lasting_facts_never_expire(llm_ready):
    ex = extract("I live in Chicago now, I moved here in the spring.", "Nice!")
    facts = [m for m in ex.memories if m.kind == "fact"]
    assert facts and all(m.valid_until is None for m in facts), ex.model_dump_json(exclude_defaults=True)


@pytest.mark.llm
def test_pattern_writer_finds_a_habit(llm_ready):
    from companion.extract import find_pattern
    p = find_pattern("Riley", ["Riley is at the gym", "Riley went to the gym after work",
                               "Riley is heading to the gym for leg day"], D1)
    assert p and "gym" in p.text.lower() and p.kind in ("preference", "fact"), p
    assert not any(ch.isdigit() for ch in p.text), f"a pattern has no dates: {p.text}"


@pytest.mark.llm
def test_pattern_writer_declines_coincidences(llm_ready):
    from companion.extract import find_pattern
    p = find_pattern("Riley", ["Riley is at the airport", "Riley is at the dentist",
                               "Riley is at the post office"], D1)
    assert p is None, p
