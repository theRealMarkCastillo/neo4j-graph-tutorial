"""Lesson 17 — the web UI's API, end to end, with a fake LLM.

    .venv/bin/python -m pytest tests/test_17_web.py -v

FastAPI's TestClient calls the app in-process (no server, no port). The chat model, the
extractor and the diary summarizer are faked, so this tests the wiring (sessions, the
background worker, JSON shapes, privacy) for free and the same way every time. The LLM parts
already have their own evals in earlier lessons.
"""
import datetime as dt
import time

import pytest
from fastapi.testclient import TestClient

from companion.models import EntityOut, EpisodeSummary, Extraction, MemoryOut, RelationOut

D1 = dt.date(2026, 11, 18)


@pytest.fixture
def client(store, driver, monkeypatch):
    """The web app on a simulated date, with every LLM call faked. `store` wipes user "pytest"."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    import companion.graph as g
    from companion.web import create_app

    monkeypatch.setattr(g, "chat_model", lambda: FakeListChatModel(responses=["Nice!"]))
    monkeypatch.setattr(g, "extract_memories", lambda *_, **__: Extraction(
        entities=[EntityOut(name="Miso", kind="pet", summary="Pytest's tabby cat")],
        relations=[RelationOut(source="user", target="Miso", type="pet")],
        memories=[MemoryOut(text="Pytest adopted a cat named Miso", kind="event", importance=6,
                            about=["user", "Miso"]),
                  MemoryOut(text="Pytest is tired today", kind="feeling", importance=3,
                            valid_until="2026-11-18")]))
    monkeypatch.setattr(g, "summarize_episode", lambda *_: EpisodeSummary(summary="Pytest adopted Miso.", mood="happy"))
    with TestClient(create_app(driver, today=D1)) as c:
        yield c


def wait_for_memories(client, timeout=60.0) -> list[str]:  # remote embeddings can take seconds
    lines, end = [], time.time() + timeout
    while time.time() < end:
        r = client.get("/api/log").json()
        lines += r["lines"]
        if not r["busy"] and lines:
            return lines
        time.sleep(0.05)
    raise AssertionError(f"background worker didn't finish: {lines}")


def test_the_page_is_served(client):
    html = client.get("/").text
    assert "<title>Companion</title>" in html
    assert "d3@7.9.0" in html  # pinned version: a CDN "latest" can change under you


def test_chat_needs_a_session_first(client):
    assert client.post("/api/chat", json={"text": "hi"}).status_code == 409


def test_a_whole_conversation(client):
    s = client.post("/api/session", json={"user": "pytest", "persona": "Sage"}).json()
    assert (s["persona"], s["today"]) == ("Sage", "2026-11-18")
    assert "I'm Sage" in s["greeting"]  # empty memory: the fixed first greeting, no LLM

    assert client.post("/api/chat", json={"text": "I adopted a cat named Miso!"}).json() == {"reply": "Nice!", "turn": 1}
    lines = wait_for_memories(client)
    assert "linked user -pet-> Miso" in lines

    memories = {m["text"]: m for m in client.get("/api/memories").json()["memories"]}
    adopted, tired = memories["Pytest adopted a cat named Miso"], memories["Pytest is tired today"]
    assert (adopted["half_life"], adopted["strength"]) == (90, 0.6)  # an event, brand new: importance/10
    assert tired["half_life"] == 7 and tired["valid_until"].startswith("2026-11-18T23:59")

    graph = client.get("/api/graph").json()
    assert {"pytest", "Miso"} <= {n["name"] for n in graph["nodes"]}
    assert graph["edges"] and graph["edges"][0]["type"] == "pet"

    assert client.get("/api/related", params={"method": "nope"}).status_code == 422
    assert isinstance(client.get("/api/related").json()["related"], list)

    end = client.post("/api/end").json()["log"]
    assert any(line.startswith("episode saved: Pytest adopted Miso.") for line in end)
    assert client.get("/api/log").status_code == 409  # no open session any more


def test_empty_messages_are_rejected(client):
    client.post("/api/session", json={"user": "pytest", "persona": "Sage"})
    assert client.post("/api/chat", json={"text": "   "}).status_code == 422


def test_the_api_shows_only_what_the_persona_may_see(client, store):
    """The UI goes through the same visibility rules as the chat (Lesson 16)."""
    store.agent = "Sage"
    store.apply(Extraction(memories=[MemoryOut(text="Pytest has panic attacks before races", kind="fact",
                                               importance=7, sensitive=True)]), today=D1)
    client.post("/api/session", json={"user": "pytest", "persona": "Coach"})
    texts = [m["text"] for m in client.get("/api/memories").json()["memories"]]
    assert not any("panic" in t for t in texts)
