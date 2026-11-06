"""Lesson 18 — seeing memory work: the per-turn trace and the inspection endpoints.

    .venv/bin/python -m pytest tests/test_18_inspect.py -v

Same setup as Lesson 17: TestClient and a fake LLM, so these check what the page is shown,
not what a model would say. Each test pins one thing a learner reads off the page: if the
trace lies, the lesson teaches the wrong thing.
"""
import datetime as dt

from test_17_web import D1, client, wait_for_memories  # noqa: F401  (client is a fixture)

from companion.embeddings import PROFILE
from companion.models import EntityOut, Extraction, MemoryOut, RelationOut


def start(client, persona="Sage"):
    return client.post("/api/session", json={"user": "pytest", "persona": persona}).json()


def finished_turn(client, n):
    wait_for_memories(client)
    return client.get("/api/trace", params={"n": n}).json()


def test_a_turn_shows_recall_the_prompt_and_the_reasons_for_each_write(client):
    start(client)
    assert client.get("/api/trace").json() == {"count": 0, "turn": None}
    client.post("/api/chat", json={"text": "I adopted a cat named Miso!"})
    t = finished_turn(client, 1)["turn"]

    assert t["message"] == "I adopted a cat named Miso!"
    assert {"core", "semantic", "entities", "dormant"} <= t["recall"].keys()
    assert "<memories>" in t["system_prompt"] and t["window"] == 1  # just this message
    assert t["recall_ms"] >= 0 and t["respond_ms"] >= 0

    x = t["extraction"]
    assert x["status"] == "done" and x["llm_ms"] >= 0 and x["write_ms"] >= 0
    assert len(x["extracted"]["memories"]) == 2
    new = [line for line in x["log"] if line.startswith("+ [event]")]
    assert new and "[new: nothing similar with the same dates]" in new[0]  # the reason, not just the result


def test_the_second_mention_is_reinforced_and_says_how_similar_it_was(client):
    start(client)
    client.post("/api/chat", json={"text": "I adopted a cat named Miso!"})
    finished_turn(client, 1)
    client.post("/api/chat", json={"text": "Did I tell you about Miso?"})
    r = finished_turn(client, 2)

    log = r["turn"]["extraction"]["log"]
    assert any(line.startswith("reinforced") and f"similarity 1.000 ≥ {PROFILE.duplicate}" in line for line in log), log
    # the extractor was shown what's already known, with ids it can reuse
    assert any(c["text"] == "Pytest adopted a cat named Miso" for c in r["turn"]["extraction"]["candidates"])
    # recall found Miso through the graph hop, so the page lights Miso up
    assert "Miso" in [e["name"] for e in r["turn"]["recall"]["entities"]]
    miso = [n["id"] for n in client.get("/api/graph").json()["nodes"] if n["name"] == "Miso"]
    assert miso and miso[0] in r["highlight"]


def test_small_talk_shows_that_extraction_was_skipped(client):
    start(client)
    client.post("/api/chat", json={"text": "thanks!"})
    x = client.get("/api/trace").json()["turn"]["extraction"]
    assert x["status"] == "skipped" and "small talk" in x["reason"]


def test_the_recall_preview_only_looks(client, store):
    start(client)
    client.post("/api/chat", json={"text": "I adopted a cat named Miso!"})
    wait_for_memories(client)
    count = lambda: store._q("MATCH (m:Memory {user: $user}) RETURN sum(m.access_count) AS n")[0]["n"]
    before = count()
    r = client.get("/api/recall", params={"q": "tell me about my cat"}).json()
    assert any("Miso" in m["text"] for m in r["recall"]["semantic"])
    assert "Miso" in r["memory_block"]
    assert count() == before  # a preview is not an access: it mustn't change what it measures
    assert client.get("/api/recall", params={"q": " "}).status_code == 422


def test_looking_ahead_shows_expiry_and_fading_without_writing(client):
    start(client)
    client.post("/api/chat", json={"text": "I adopted a cat named Miso!"})
    wait_for_memories(client)
    now = {m["text"]: m for m in client.get("/api/memories").json()["memories"]}
    later = client.get("/api/memories", params={"ahead": 90}).json()
    then = {m["text"]: m for m in later["memories"]}

    assert later["as_of"] == (D1 + dt.timedelta(days=90)).isoformat()
    assert not now["Pytest is tired today"]["expired"] and then["Pytest is tired today"]["expired"]
    adopted = then["Pytest adopted a cat named Miso"]
    assert adopted["strength"] == 0.3  # an event has a 90-day half-life: 0.6 -> 0.3
    assert client.get("/api/memories").json()["memories"][0]["age_days"] == 0  # nothing changed
    assert client.get("/api/memories", params={"ahead": -1}).status_code == 422


def test_the_persona_lens_shows_what_another_persona_would_see(client, store):
    store.agent = "Sage"
    store.apply(Extraction(
        entities=[EntityOut(name="Dr. Lee", kind="person", summary="Pytest's therapist")],
        relations=[RelationOut(source="user", target="Dr. Lee", type="therapist")],
        memories=[MemoryOut(text="Pytest sees Dr. Lee for anxiety", kind="fact", importance=7,
                            sensitive=True, about=["user", "Dr. Lee"])]), today=D1)
    start(client, "Sage")
    names = lambda persona: {n["name"] for n in client.get("/api/graph", params={"persona": persona}).json()["nodes"]}
    assert "Dr. Lee" in names("Sage")
    assert "Dr. Lee" not in names("Coach")
    edge = client.get("/api/graph").json()["edges"][0]
    assert edge["private"] and edge["agent"] == "Sage"  # the page draws it dashed


def test_the_diary_lists_the_open_episode_and_the_worker_reports_its_queue(client):
    start(client)
    client.post("/api/chat", json={"text": "I adopted a cat named Miso!"})
    assert isinstance(client.get("/api/log").json()["queued"], int)
    wait_for_memories(client)
    r = client.get("/api/episodes").json()
    open_now = [e for e in r["episodes"] if e["open"]]
    assert len(open_now) == 1 and open_now[0]["turns"] == 2 and open_now[0]["memories"] == 2  # message + reply
    assert r["patterns"] == []


def test_reasons_in_the_store_log(store):
    """The reasons come from MemoryStore itself, so the terminal chat shows them too."""
    fact = MemoryOut(text="Pytest plays the cello", kind="fact", importance=5)
    first = store.apply(Extraction(memories=[fact]), today=D1)
    assert first == ["+ [fact] Pytest plays the cello  [new: nothing similar with the same dates]"]
    again = store.apply(Extraction(memories=[fact, fact]), today=D1)
    assert again[0].startswith("reinforced") and '"Pytest plays the cello"' in again[0]
    other = store.apply(Extraction(memories=[MemoryOut(text="Pytest likes hiking", kind="preference",
                                                       importance=5)]), today=D1)
    assert "[new: closest known 0." in other[0] and f"< {PROFILE.duplicate}]" in other[0]
