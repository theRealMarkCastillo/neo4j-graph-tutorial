"""Lesson 13 — memory over time: validity, decay, consolidation.

Time makes tests flaky unless the clock is an input. Every function here takes `today`
(clock.now_for turns it into "now"), so a test can store a memory on day 1 and recall on
day 30 in the same second, with no sleeping and no mocking of datetime.
"""
import datetime as dt

import pytest

from companion import clock, decay
from companion.models import EntityOut, Extraction, MemoryOut
from companion.store import format_context

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


def did(store, text, kind="status", day=0, about=("gym",), **kw):
    """One memory on day D1 + `day`, ABOUT the given activities (created if new)."""
    store.apply(Extraction(entities=[EntityOut(name=a, kind="activity") for a in about],
                           memories=[mem(text, kind, 2, about=list(about), **kw)]),
                today=D1 + dt.timedelta(days=day))


def habit(store):
    return store._q(
        """
        MATCH (p:Memory {user: $user}) WHERE p.evidence IS NOT NULL
        OPTIONAL MATCH (p)-[:ABOUT]->(a:Entity {kind: 'activity'})
        RETURN p.evidence AS evidence, collect(a.name) AS about
        """)


def test_three_days_at_the_gym_make_a_habit(store):
    """A habit is a graph count (structure-over-prose spec, D3): memories ABOUT one activity
    on 3+ distinct days. A plan counts: saying "going to the gym after work" is evidence."""
    did(store, "Pytest is at the gym", "status", 0)
    did(store, "Pytest plans to go to the gym after work", "plan", 1, due_on=(D1 + dt.timedelta(days=1)).isoformat())
    did(store, "Pytest went to the gym", "event", 3, occurred_on=(D1 + dt.timedelta(days=3)).isoformat())
    calls = []
    log = store.consolidate(D1 + dt.timedelta(days=4), writer=fake_writer(calls), prune=False)
    assert len(calls) == 1 and len(calls[0]) == 3, (calls, log)
    assert habit(store) == [{"evidence": 3, "about": ["gym"]}]
    assert store._q("""MATCH (:User {id: $user})-[r:RELATED {type: 'practices'}]->(a:Entity {name: 'gym'})
                       WHERE r.until IS NULL RETURN a""")
    # running it again finds nothing new: members are marked consolidated
    assert store.consolidate(D1 + dt.timedelta(days=4), writer=fake_writer(calls), prune=False) == []


def test_one_day_of_gym_talk_is_not_a_habit(store):
    for text in ["Pytest is heading to the gym", "Pytest is at the gym", "Pytest got back from the gym"]:
        did(store, text, "status", 0)
    calls = []
    store.consolidate(D1 + dt.timedelta(days=1), writer=fake_writer(calls), prune=False)
    assert calls == []


def test_a_memory_counts_for_each_activity(store):
    for day in range(3):
        did(store, f"Pytest did the gym and then yoga, session {day}", "event", day, about=("gym", "yoga"))
    calls = []
    store.consolidate(D1 + dt.timedelta(days=3), writer=fake_writer(calls), prune=False)
    assert len(calls) == 2, calls


def test_wording_does_not_matter(store):
    """Regression (gemini-3.8-flash run): these three sentences were too unlike each other for
    bge-small's similarity cut, so no habit formed. Linked to the gym, they count."""
    did(store, "Pytest is heading to the gym.", "status", 0)
    did(store, "Pytest plans to go to the gym after work and then have dinner", "plan", 3,
        due_on=(D1 + dt.timedelta(days=3)).isoformat())
    did(store, "Pytest went to the gym before work to keep up a workout streak", "event", 5,
        occurred_on=(D1 + dt.timedelta(days=5)).isoformat())
    calls = []
    store.consolidate(D1 + dt.timedelta(days=6), writer=fake_writer(calls), prune=False)
    assert len(calls) == 1 and len(calls[0]) == 3, calls


def test_unlinked_memories_form_no_habit(store):
    for day in range(3):
        store.apply(Extraction(memories=[mem("Pytest is at the gym", "status", 2)]), today=D1 + dt.timedelta(days=day))
    calls = []
    store.consolidate(D1 + dt.timedelta(days=3), writer=fake_writer(calls), prune=False)
    assert calls == []


def test_a_lapsed_plan_is_no_evidence(store):
    did(store, "Pytest is at the gym", "status", 0)
    did(store, "Pytest plans to go to the gym", "plan", 1, due_on=(D1 + dt.timedelta(days=1)).isoformat())
    did(store, "Pytest is heading to the gym", "status", 2)
    store._q("MATCH (m:Memory {user: $user, kind: 'plan'}) SET m.status = 'lapsed'")
    calls = []
    store.consolidate(D1 + dt.timedelta(days=3), writer=fake_writer(calls), prune=False)
    assert calls == []


def test_private_occurrences_dont_form_a_pattern(store):
    did(store, "Pytest is at the gym", "status", 0)
    did(store, "Pytest is heading to the gym", "status", 1, private=True)
    did(store, "Pytest went to the gym", "event", 2)
    calls = []
    store.consolidate(D1 + dt.timedelta(days=3), writer=fake_writer(calls), prune=False)
    assert calls == []


def test_future_plans_are_not_days_yet(store):
    """Review finding: "gym Mon, Wed, Fri next week" made three plans with three due dates,
    and that one message counted as a 3-day habit before any visit happened."""
    for day in (2, 4, 6):
        did(store, f"Pytest plans to go to the gym, day {day}", "plan", 0,
            due_on=(D1 + dt.timedelta(days=day)).isoformat())
    calls = []
    store.consolidate(D1 + dt.timedelta(days=1), writer=fake_writer(calls), prune=False)
    assert calls == []


def test_a_long_status_counts_on_the_day_it_was_said(store):
    """valid_until is when a status ENDS, not another day it happened."""
    did(store, "Pytest went to the gym", "event", 0, occurred_on=D1.isoformat())
    did(store, "Pytest is training at the gym all week", "status", 0,
        valid_until=f"{(D1 + dt.timedelta(days=5)).isoformat()}T20:00")
    did(store, "Pytest is at the gym", "status", 1)
    calls = []
    store.consolidate(D1 + dt.timedelta(days=2), writer=fake_writer(calls), prune=False)
    assert calls == []  # two days: D1 and D1 + 1


def test_a_known_habit_gathers_new_evidence_without_a_rewrite(store):
    """Review finding: a second run asked the writer again, and its new wording rarely
    cleared the text dedupe, so near-copies of the habit piled up. The graph already knows
    the activity has a pattern: new days attach to it."""
    for day in range(3):
        did(store, "Pytest is at the gym", "status", day)
    store.consolidate(D1 + dt.timedelta(days=3), writer=fake_writer([]), prune=False)
    for day in range(10, 13):
        did(store, "Pytest is at the gym", "status", day)
    calls = []
    other = lambda u, t, d: calls.append(t) or MemoryOut(text="Pytest trains at the gym several times a week",
                                                          kind="preference", importance=6)
    store.consolidate(D1 + dt.timedelta(days=13), writer=other, prune=False)
    assert calls == []
    assert habit(store) == [{"evidence": 6, "about": ["gym"]}]


def test_pruning_members_does_not_shrink_the_evidence(store):
    """Final review: evidence was recounted from SUMMARIZES edges, and pruning an expired
    member deletes its edge, so a habit built from statuses never grew past 3-5 days."""
    for day in range(3):
        did(store, "Pytest is at the gym", "status", day)
    store.consolidate(D1 + dt.timedelta(days=3), writer=fake_writer([]))
    for day in range(10, 13):
        did(store, "Pytest is at the gym", "status", day)
    store.consolidate(D1 + dt.timedelta(days=13), writer=fake_writer([]))
    assert habit(store) == [{"evidence": 6, "about": ["gym"]}]


def test_plans_alone_are_not_a_habit(store):
    """Final review: "gym Mon, Wed, Fri next week" is three plans from one message. Once the
    days pass they would count as three visits nobody confirmed. Plans still count, next to
    at least one thing that happened."""
    for day in (1, 3, 5):
        did(store, "Pytest plans to go to the gym", "plan", 0, due_on=(D1 + dt.timedelta(days=day)).isoformat())
    calls = []
    store.consolidate(D1 + dt.timedelta(days=7), writer=fake_writer(calls), prune=False)
    assert calls == []


def test_a_shared_memory_still_counts_for_a_later_habit(store):
    """Final review: once the gym habit summarized "ran to the gym", the global
    consolidated_at hid it from running, which then never reached 3 days."""
    did(store, "Pytest went to the gym", "event", 0)
    for day in (1, 2):
        did(store, f"Pytest ran to the gym, day {day}", "event", day, about=("gym", "running"))
    store.consolidate(D1 + dt.timedelta(days=3), writer=fake_writer([]), prune=False)
    did(store, "Pytest went running", "event", 4, about=("running",))
    calls = []
    store.consolidate(D1 + dt.timedelta(days=5), writer=fake_writer(calls), prune=False)
    assert len(calls) == 1 and len(calls[0]) == 3, calls


def test_shared_members_are_counted_once(store):
    """Review finding: a memory ABOUT gym and yoga summarized by one pattern (the second
    write deduped into the first) added its members twice: evidence 6 for 3 memories."""
    for day in range(3):
        did(store, f"Pytest did the gym and then yoga, session {day}", "event", day, about=("gym", "yoga"))
    store.consolidate(D1 + dt.timedelta(days=3), writer=fake_writer([]), prune=False)
    [h] = habit(store)
    assert h["evidence"] == 3 and sorted(h["about"]) == ["gym", "yoga"]


def test_writer_can_decline(store):
    for i in range(3):
        did(store, "Pytest is at the gym", "status", 2 * i)
    assert store.consolidate(D1 + dt.timedelta(days=6), writer=lambda u, t, d: None, prune=False) == []


def test_prune_deletes_only_dormant_trivia(store):
    end_of_day = f"{D1.isoformat()}T23:59"  # feelings end with the day (extraction rule 3)
    store.apply(Extraction(memories=[
        mem("Pytest felt grumpy about the rain", "feeling", 2, valid_until=end_of_day),
        mem("Pytest felt overjoyed at their wedding", "feeling", 9, valid_until=end_of_day),
        mem("Pytest went to a concert", "event", 2)]), today=D1)
    assert store.consolidate(D1 + dt.timedelta(days=30)) == []  # dormant, but may still form a pattern
    log = store.consolidate(D1 + dt.timedelta(days=61))         # past the pattern window: prune
    assert log == ["pruned 1 dormant expired memories"]
    left = {m["text"] for m in store.memories(include_retired=True, today=D1 + dt.timedelta(days=61))}
    assert left == {"Pytest felt overjoyed at their wedding", "Pytest went to a concert"}


def test_consolidated_details_are_pruned_once_dormant(store):
    for i, text in enumerate(["Pytest is at the gym", "Pytest went to the gym after work",
                              "Pytest is heading to the gym for leg day"]):
        did(store, text, "status", i, valid_until=f"{(D1 + dt.timedelta(days=i)).isoformat()}T14:00")
    log = store.consolidate(D1 + dt.timedelta(days=10), writer=fake_writer([]))
    assert "pruned 3 dormant expired memories" in log
    [p] = store._q("MATCH (p:Memory {user:$user}) WHERE p.evidence IS NOT NULL RETURN p.evidence AS n")
    assert p["n"] == 3


def test_prune_needs_an_expiry(store):
    """"Temporary" is a property (structure-over-prose spec, D4): only a memory whose
    valid_until has passed is pruned. A feeling stored without one is kept."""
    store.apply(Extraction(memories=[mem("Pytest felt grumpy", "feeling", 2)]), today=D1)  # no valid_until
    assert store.consolidate(D1 + dt.timedelta(days=61)) == []


def test_an_expired_event_is_pruned_like_a_status(store):
    store.apply(Extraction(memories=[mem("Pytest was stuck in traffic", "event", 2,
                                         valid_until=f"{D1.isoformat()}T18:00")]), today=D1)
    assert store.consolidate(D1 + dt.timedelta(days=400)) == ["pruned 1 dormant expired memories"]


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
