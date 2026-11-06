"""Lesson 10 — recall and chat. Deterministic parts first (the small-talk gate, recall
queries, the prompt block); how the LLM *uses* the memories is graded in test_11."""
import datetime as dt

import pytest

from companion.graph import worth_remembering
from companion.models import EntityOut, Extraction, MemoryOut, RelationOut
from companion.store import format_context

T = dt.date(2026, 11, 5)


@pytest.mark.parametrize("text,expected", [
    ("ok", False), ("haha yeah", False), ("Thanks!", False), ("good night", False),
    ("ok so my sister is moving to Denver", True),
    ("My dog Rex is sick", True),
    ("no", False),
])
def test_small_talk_gate(text, expected):
    # Table-driven tests make a regex's behaviour readable, and every bug becomes a new row.
    assert worth_remembering(text) is expected


def test_yes_to_a_question_is_worth_remembering():
    # Prompt review: "yep" answering "Is the interview Friday?" confirms a fact.
    assert worth_remembering("yep", "Is your interview on Friday?")
    assert not worth_remembering("yep", "That sounds fun.")
    # ...but laughter after a question is still small talk (scenario cross_turn_facts caught this)
    assert not worth_remembering("haha yeah", "Are you excited?")


def m(text, kind="fact", **kw):
    return MemoryOut(text=text, kind=kind, importance=kw.pop("importance", 7), **kw)


@pytest.fixture
def life(store):
    store.apply(Extraction(
        entities=[EntityOut(name="Maya", kind="person"), EntityOut(name="Rex", kind="pet")],
        relations=[RelationOut(source="user", target="Maya", type="sister"),
                   RelationOut(source="user", target="Rex", type="pet")],
        memories=[m("Pytest's sister Maya is a nurse in Denver", about=["Maya"]),
                  m("Pytest's dog Rex is afraid of thunderstorms", about=["Rex"]),
                  m("Pytest doesn't want to talk about their job", "boundary", importance=9),
                  m("Pytest is being treated for anxiety", sensitive=True, importance=8),
                  m("Pytest has a job interview at Acme", "plan", due_on="2026-11-06")],
    ), today=T)
    return store


def test_semantic_recall_finds_the_right_memory(life):
    r = life.recall("how is your sister doing?", today=T)
    texts = [x["text"] for x in r["semantic"]] + [x["text"] for e in r["entities"] for x in e["memories"]]
    assert "Pytest's sister Maya is a nurse in Denver" in texts


def test_an_unrelated_message_gets_only_loosely_related_memories(life):
    """Vector search always returns its nearest memories: "do you like pizza?" still scores
    about 0.7 against the dog and the sister (found in the web UI). The reranker labels them
    for the trace. They're not dropped: inferential matches score just as low."""
    r = life.recall("do you like pizza?", today=T)
    assert r["semantic"] and all(x["weak"] for x in r["semantic"])
    if r["relevance_by"] == "cross-encoder":  # bge-small: the reranker's scores (logits)
        assert all(x["relevance"] < -8.5 for x in r["semantic"])
    else:  # a model with its own cut (Lesson 19): every score is under it
        assert all(x["relevance"] < r["relevance_min"] for x in r["semantic"])


def test_a_related_message_keeps_its_match_on_top(life):
    r = life.recall("my dog hates storms", today=T)
    top = r["semantic"][0]
    assert top["text"] == "Pytest's dog Rex is afraid of thunderstorms" and not top["weak"]


def test_an_inferential_match_is_kept_even_when_the_reranker_misses_it(store):
    """"My meds don't help" -> the ADHD memory: the reranker scores it like noise (-10.9),
    so dropping weak matches would hide exactly this (the sensitive scenario caught it).
    Qwen3-Embedding-8B labels it by its own score (0.74) and gets it right (Lesson 19)."""
    store.apply(Extraction(memories=[m("Pytest was diagnosed with ADHD in 2025 and takes medication for it",
                                       sensitive=True)]), today=T)
    r = store.recall("I'm struggling to focus at work, my meds don't seem to help", today=T)
    assert "ADHD" in format_context(r, T)
    if r["relevance_by"] == "cross-encoder":
        assert r["semantic"][0]["weak"]  # the reranker did miss it: only the vector search found it
    else:
        assert "ADHD" in r["semantic"][0]["text"] and not r["semantic"][0]["weak"]


def test_boundaries_are_always_in_core(life):
    r = life.recall("what should I cook tonight?", today=T)
    assert any("doesn't want to talk about their job" in x["text"] for x in r["core"])


def test_sensitive_memories_stay_out_of_core(life):
    r = life.recall("what should I cook tonight?", today=T)
    assert not any("anxiety" in x["text"] for x in r["core"])


def test_followups_include_upcoming_plans(life):
    # Regression: the "job" boundary is ABOUT the user, and used to hide every memory
    # about the user, this plan included.
    r = life.recall("hi", today=T)
    assert any("interview" in x["text"] for x in r["followups"])


def test_person_boundary_hides_that_person_only(life):
    life.apply(Extraction(memories=[m("Don't bring up Maya", "boundary", about=["Maya"], importance=9)]), today=T)
    r = life.recall("how is your sister doing?", today=T)
    texts = [x["text"] for x in r["semantic"]] + [x["text"] for e in r["entities"] for x in e["memories"]]
    assert not any("nurse" in t for t in texts)
    assert any("interview" in x["text"] for x in r["followups"])


def test_prompt_block_has_labelled_sections(life):
    block = format_context(life.recall("how is Rex?", today=T), today=T)
    assert "Core profile:" in block and "People and things in their life:" in block
    assert "Rex (pet)" in block


def test_empty_memory_says_so():
    empty = {k: [] for k in ["core", "semantic", "entities", "people", "followups", "upcoming",
                             "history", "episodes", "recent"]}
    assert "No memories yet" in format_context(empty)


def test_protected_person_does_not_leak_through_history_or_entity_hop(store):
    store.apply(Extraction(
        entities=[EntityOut(name="Dana", kind="person"), EntityOut(name="Maya", kind="person")],
        memories=[m("Pytest's partner is Dana", about=["user", "Dana"]),
                  m("Maya introduced Pytest to Dana", "event", about=["Maya", "Dana"])]), today=T)
    old = next(x for x in store.memories() if "partner" in x["text"])
    store.apply(Extraction(memories=[
        m("Pytest is single now", supersedes=[old["id"]]),
        m("Don't bring up Dana", "boundary", about=["Dana"], importance=9)]), today=T)
    block = format_context(store.recall("am I single? how is Maya?", today=T), today=T)
    assert "Dana" not in block.replace("Don't bring up Dana", "")


def test_entity_hop_follows_links_between_entities_from_either_end(store):
    """user -son-> Logan -likes-> Minecraft: asking about Minecraft must reach "your son
    Logan", and asking about Logan must reach Minecraft (a two-hop life graph)."""
    store.apply(Extraction(
        entities=[EntityOut(name="Logan", kind="person"), EntityOut(name="Minecraft", kind="thing")],
        relations=[RelationOut(source="user", target="Logan", type="son"),
                   RelationOut(source="Logan", target="Minecraft", type="likes")],
        memories=[m("Pytest's son Logan is 7 years old", about=["Logan"])],
    ), today=T)
    for question in ("does anyone here play Minecraft?", "what should I get Logan?"):
        block = format_context(store.recall(question, today=T, touch=False), T)
        assert "Logan (your son) likes Minecraft" in block, block


def test_a_protected_entity_is_not_reached_through_a_link(store):
    store.apply(Extraction(
        entities=[EntityOut(name="Logan", kind="person"), EntityOut(name="Dana", kind="person")],
        relations=[RelationOut(source="user", target="Logan", type="son"),
                   RelationOut(source="Logan", target="Dana", type="mother")],
        memories=[m("Pytest doesn't want to talk about Dana", "boundary", about=["Dana"], importance=9)],
    ), today=T)
    r = store.recall("how is Logan?", today=T, touch=False)
    links = [l for e in r["entities"] for l in e["links"]]
    assert "Logan" in [e["name"] for e in r["entities"]]
    assert all("Dana" not in (l["source"], l["target"]) for l in links), links


def test_entity_hop_reaches_two_steps_out(store):
    store.apply(Extraction(
        entities=[EntityOut(name="Logan", kind="person"), EntityOut(name="Max", kind="person"),
                  EntityOut(name="Biscuit", kind="pet")],
        relations=[RelationOut(source="user", target="Logan", type="son"),
                   RelationOut(source="Logan", target="Max", type="friend"),
                   RelationOut(source="Max", target="Biscuit", type="pet")],
        memories=[m("Pytest's son Logan is 7 years old", about=["Logan"])],
    ), today=T)
    block = format_context(store.recall("how is Logan doing?", today=T, touch=False), T)
    assert "Max is Logan (your son)'s friend" in block and "Biscuit is Max's pet" in block, block
