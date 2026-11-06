"""Lesson 17 — a small web UI for the companion.

    uv run python -m companion.web                    # http://127.0.0.1:8000
    uv run python -m companion.web --today 2026-11-20 # simulate a date (the page picks user and persona)

One page: the chat, the life graph, how strong each memory still is (Lesson 13 decay), and
what the graph embeddings think belongs together (Lesson 15). Memory extraction runs on the
background worker (Lesson 14), and the page polls /api/log for the ✎ lines.

Lesson 18 adds what memory DOES on each turn: the recall trace (what was found, from which
retriever, what was left out), the exact system prompt, why each write happened, timings, a
persona lens, a look-ahead clock for decay, and the episode diary with its patterns.

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
from companion.store import MemoryStore, entity_visible, format_context, visible
from companion.worker import Background

STATIC = Path(__file__).parent / "static"


class Chat:
    """The one open conversation: its store, LangGraph app, worker and episode."""

    def __init__(self, driver, user: str, persona: str, today: dt.date):
        self.store = MemoryStore(driver, user, agent=persona)
        self.background = Background()
        self.trace: list[dict] = []  # one entry per turn (Lesson 18)
        self.app = companion_graph.build_companion(self.store, self.background, self.trace)
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
    """Neo4j and Python dates -> ISO strings, for JSON."""
    if isinstance(value, (Date, DateTime)):
        return value.iso_format()
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
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
        c = chat()
        reply = c.say(body.text.strip())
        return {"reply": reply, "turn": len(c.trace)}

    @api.get("/api/log")
    def log():
        c = chat()
        return {"lines": c.background.drain(), "busy": c.background.busy, "queued": c.background.queued}

    @api.post("/api/end")
    def end():
        c, state["chat"] = chat(), None
        return {"log": c.close()}

    def store_for(persona: str | None) -> MemoryStore:
        """The open chat's store, or the same user's memory as another persona sees it."""
        store = chat().store
        if not persona or persona == store.agent:
            return store
        return MemoryStore(driver, store.user_name, agent=persona)

    def ahead_of(days: int) -> dt.date:
        if not 0 <= days <= 3650:
            raise HTTPException(422, "ahead must be 0-3650 days")
        return day + dt.timedelta(days=days)

    @api.get("/api/memories")
    def memories(persona: str | None = None, ahead: int = 0):
        """Current memories with what the decay curve needs (the page draws the curves).
        `ahead` looks N days into the future: strength, dormancy and expiry as of then
        (nothing is written; the memories are the ones that exist today)."""
        store, when = store_for(persona), ahead_of(ahead)
        rows = store.memories(today=day)
        ages = {r["id"]: r for r in store._q(
            """
            MATCH (m:Memory) WHERE m.id IN $ids
            RETURN m.id AS id, duration.inSeconds(coalesce(m.last_confirmed, m.created_at), $now).seconds / 86400.0 AS age,
                   m.valid_until IS NOT NULL AND m.valid_until <= $now AS expired_then
            """, ids=[r["id"] for r in rows], now=clock.now_for(when))}
        out = []
        for r in rows:
            age = max(ages.get(r["id"], {}).get("age", 0.0), 0.0)
            out.append({**_plain(r), "age_days": round(age, 2),
                        "expired": bool(ages.get(r["id"], {}).get("expired_then")),
                        "half_life": decay.HALF_LIFE_DAYS.get(r["kind"]),
                        "strength": round(decay.strength(r["importance"], r["kind"], age), 3),
                        "dormant": decay.is_dormant(r["importance"], r["kind"], age),
                        "milestone": r["importance"] >= decay.MILESTONE})
        return {"memories": out, "dormant_below": decay.DORMANT_BELOW, "as_of": when.isoformat()}

    @api.get("/api/trace")
    def trace(n: int | None = None):
        """What memory did on turn `n` (default: the latest): recall, prompt, writes, timings."""
        c = chat()
        if not c.trace:
            return {"count": 0, "turn": None}
        if n is not None and not 1 <= n <= len(c.trace):
            raise HTTPException(404, f"no turn {n}")
        t = c.trace[(n or len(c.trace)) - 1]
        return {"count": len(c.trace), "turn": _plain(t), "highlight": highlight(c.store, t["recall"])}

    def highlight(store: MemoryStore, found: dict) -> list[str]:
        """Entities this recall touched: hopped to, or what its memories are ABOUT."""
        ids = [m["id"] for key in ("core", "semantic", "followups", "upcoming", "recent")
               for m in found.get(key, [])]
        about = store._q("MATCH (m:Memory)-[:ABOUT]->(e:Entity) WHERE m.id IN $ids AND NOT e:User "
                         "RETURN DISTINCT e.id AS id", ids=ids)
        return sorted({e["id"] for e in found["entities"]} | {r["id"] for r in about})

    @api.get("/api/recall")
    def preview(q: str, ahead: int = 0, persona: str | None = None):
        """Recall for any text, without chatting and without counting as an access: try a
        question, a later date, or another persona and compare what memory would hand the LLM."""
        if not q.strip():
            raise HTTPException(422, "empty query")
        store, when = store_for(persona), ahead_of(ahead)
        found = store.recall(q, today=when, touch=False)
        return {"as_of": when.isoformat(), "persona": store.agent, "recall": _plain(found),
                "memory_block": format_context(found, when), "highlight": highlight(store, found)}

    @api.get("/api/episodes")
    def episodes():
        """This persona's diary (Lesson 11) and the patterns consolidation built (Lesson 13)."""
        store = chat().store
        eps = store._q(
            """
            MATCH (ep:Episode {user: $user}) WHERE ep.agent IS NULL OR ep.agent = $agent
            RETURN ep.id AS id, ep.started_at AS started, ep.ended_at AS ended, ep.summary AS summary,
                   ep.mood AS mood, COUNT { (ep)-[:HAS_TURN]->() } AS turns,
                   COUNT { (m:Memory)-[:FROM]->(ep) WHERE """ + visible("m") + """ } AS memories,
                   ep.id = $open AS open
            ORDER BY ep.started_at DESC LIMIT 20
            """, open=chat().episode_id)
        patterns = store._q(
            """
            MATCH (p:Memory {user: $user})-[:SUMMARIZES]->(m:Memory)
            WHERE p.invalidated_at IS NULL AND """ + visible("p") + """
            RETURN p.id AS id, p.text AS text, p.kind AS kind, coalesce(p.evidence, 0) AS evidence,
                   collect(m.text) AS from
            """)
        return {"episodes": _plain(eps), "patterns": patterns}

    @api.get("/api/graph")
    def life_graph(persona: str | None = None):
        """The user, the entities this persona may see, and the current relations between them.
        `persona` shows the graph as another persona would see it (Lesson 16's lens)."""
        store = store_for(persona)
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
            RETURN a.id AS source, b.id AS target, r.type AS type,
                   coalesce(r.private, false) AS private, r.agent AS agent
            """, ids=ids)
        return {"persona": store.agent, "nodes": nodes, "edges": edges}

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
