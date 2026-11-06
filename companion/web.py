"""Lesson 17 — a small web UI for the companion.

    .venv/bin/python -m companion.web                    # http://127.0.0.1:8000
    .venv/bin/python -m companion.web --today 2026-11-20 # simulate a date (the page picks user and persona)

One page: the chat, the life graph, how strong each memory still is (Lesson 13 decay), and
what the graph embeddings think belongs together (Lesson 15). Memory extraction runs on the
background worker (Lesson 14), and the page polls /api/log for the ✎ lines.

It's a local, single-user tutorial app: one conversation at a time, bound to 127.0.0.1, no
login. Putting it on a network would need authentication first.
"""
from __future__ import annotations

import argparse
import datetime as dt
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage
from neo4j.time import Date, DateTime
from pydantic import BaseModel

from config import COMPANION_NAME, COMPANION_USER
from companion import clock, decay, graph as companion_graph, insights
from companion.store import MemoryStore, entity_visible, visible
from companion.worker import Background

STATIC = Path(__file__).parent / "static"


class Chat:
    """The one open conversation: its store, LangGraph app, worker and episode."""

    def __init__(self, driver, user: str, persona: str, today: dt.date):
        self.store = MemoryStore(driver, user, agent=persona)
        self.background = Background()
        self.app = companion_graph.build_companion(self.store, self.background)
        self.today = today
        self.episode_id = self.store.start_episode(on=None if today == dt.date.today() else today)
        self.greeting = companion_graph.greeting(self.store, today)
        companion_graph.seed_thread(self.app, self.episode_id, self.greeting)
        self.lock = threading.Lock()  # one message at a time: the thread's state is sequential

    def say(self, text: str) -> str:
        with self.lock:
            out = self.app.invoke({"messages": [HumanMessage(text)]},
                                  {"configurable": {"thread_id": self.episode_id, "today": self.today}})
        return out["messages"][-1].content

    def close(self) -> list[str]:
        lines = self.background.wait()  # the end-of-episode pass must see every memory
        lines += companion_graph.close_episode(self.store, self.episode_id, self.today)
        self.background.close()
        return lines


class Start(BaseModel):
    user: str = COMPANION_USER
    persona: str = COMPANION_NAME


class Message(BaseModel):
    text: str


def _plain(value):
    """Neo4j temporal values -> ISO strings, for JSON."""
    if isinstance(value, (Date, DateTime)):
        return value.iso_format()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def create_app(driver, today: dt.date | None = None) -> FastAPI:
    state: dict[str, Chat | None] = {"chat": None}

    @asynccontextmanager
    async def lifespan(_):
        yield
        if state["chat"]:  # server stopping: save the open conversation
            state["chat"].close()

    api = FastAPI(title="Companion", lifespan=lifespan)
    day = today or dt.date.today()

    def chat() -> Chat:
        if state["chat"] is None:
            raise HTTPException(409, "start a session first (POST /api/session)")
        return state["chat"]

    @api.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @api.post("/api/session")
    def start(body: Start):
        log = state["chat"].close() if state["chat"] else []
        state["chat"] = Chat(driver, body.user, body.persona, day)
        c = state["chat"]
        return {"user": c.store.user_name, "persona": c.store.agent, "today": day.isoformat(),
                "greeting": c.greeting, "log": log}

    @api.post("/api/chat")
    def say(body: Message):
        if not body.text.strip():
            raise HTTPException(422, "empty message")
        return {"reply": chat().say(body.text.strip())}

    @api.get("/api/log")
    def log():
        c = chat()
        return {"lines": c.background.drain(), "busy": c.background.busy}

    @api.post("/api/end")
    def end():
        c, state["chat"] = chat(), None
        return {"log": c.close()}

    @api.get("/api/memories")
    def memories():
        """Current memories with what the decay curve needs (the page draws the curves)."""
        store = chat().store
        rows = store.memories(today=day)
        ages = {r["id"]: r["age"] for r in store._q(
            """
            MATCH (m:Memory) WHERE m.id IN $ids
            RETURN m.id AS id, duration.inSeconds(coalesce(m.last_confirmed, m.created_at), $now).seconds / 86400.0 AS age
            """, ids=[r["id"] for r in rows], now=clock.now_for(day))}
        out = []
        for r in rows:
            age = max(ages.get(r["id"], 0.0), 0.0)
            out.append({**_plain(r), "age_days": round(age, 2),
                        "half_life": decay.HALF_LIFE_DAYS.get(r["kind"]),
                        "strength": round(decay.strength(r["importance"], r["kind"], age), 3),
                        "dormant": decay.is_dormant(r["importance"], r["kind"], age),
                        "milestone": r["importance"] >= decay.MILESTONE})
        return {"memories": out, "dormant_below": decay.DORMANT_BELOW}

    @api.get("/api/graph")
    def life_graph():
        """The user, the entities this persona may see, and the current relations between them."""
        store = chat().store
        nodes = store._q(
            """
            MATCH (e:Entity {user: $user}) WHERE e:User OR (""" + entity_visible("e") + """
              AND NOT EXISTS { MATCH (b:Memory {kind: 'boundary'})-[:ABOUT]->(e)
                               WHERE b.invalidated_at IS NULL AND NOT e:User })
            RETURN e.id AS id, e.name AS name, e.kind AS kind, coalesce(e.summary, '') AS summary,
                   COUNT { (e)<-[:ABOUT]-(m:Memory) WHERE m.invalidated_at IS NULL AND """ + visible("m") + """ } AS memories
            """)
        ids = [n["id"] for n in nodes]
        edges = store._q(
            """
            MATCH (a:Entity)-[r:RELATED]->(b:Entity) WHERE a.id IN $ids AND b.id IN $ids
              AND r.until IS NULL AND """ + visible("r") + """
            RETURN a.id AS source, b.id AS target, r.type AS type
            """, ids=ids)
        return {"nodes": nodes, "edges": edges}

    @api.get("/api/related")
    def related(name: str | None = None, method: str = "fastrp"):
        if method not in insights.MIN_SCORE:
            raise HTTPException(422, f"method must be one of {sorted(insights.MIN_SCORE)}")
        return {"related": insights.related(chat().store, name or None, k=3 if name else 12, method=method)}

    return api


def main() -> None:
    import uvicorn
    from neo4j import GraphDatabase

    from config import NEO4J_AUTH, NEO4J_URI

    parser = argparse.ArgumentParser(description="Web UI for the companion.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--today", type=dt.date.fromisoformat, help="simulate a date (YYYY-MM-DD)")
    args = parser.parse_args()
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    try:
        uvicorn.run(create_app(driver, args.today), host="127.0.0.1", port=args.port)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
