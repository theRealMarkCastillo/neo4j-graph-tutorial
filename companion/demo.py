"""Scripted two-session demo (Lesson 10):   .venv/bin/python -m companion.demo

Session 1 (2026-11-05): a new user introduces themselves.
Session 2 (2026-11-13): a week later — the companion should follow up on the presentation,
use the sister/dog memories, and update the job fact rather than duplicate it.
Uses a separate user ("Alex (demo)") and wipes it first, so it never touches your own memories.
"""
import datetime as dt

from langchain_core.messages import HumanMessage
from neo4j import GraphDatabase

from config import COMPANION_NAME, NEO4J_AUTH, NEO4J_URI
from companion.extract import summarize_episode
from companion.graph import build_companion, greeting, seed_thread
from companion.store import MemoryStore, format_context

USER = "Alex"
SESSIONS = [
    (dt.date(2026, 11, 5), [
        "Hi! I'm Alex. I moved to Portland last month for a new job as a UX researcher at Lumen Health.",
        "My sister Maya lives in Denver, she's a nurse. We're really close. Oh, and I have a beagle named Biscuit.",
        "Honestly I'm a bit nervous — I'm giving my first big presentation to leadership next Thursday.",
        "One thing: please don't bring up my ex, Jordan. That's still a sore spot.",
    ]),
    (dt.date(2026, 11, 13), [
        "It went really well actually! My VP said it was the clearest research readout she's seen.",
        "Maya's visiting for Thanksgiving. Any ideas for things to do around Portland that work with a dog?",
        "Also, small update: I switched teams, I'm on the Research Ops team now instead of the patient app team.",
    ]),
]


def rule(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * (60 - len(title))}")


def main() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    store = MemoryStore(driver, f"{USER} (demo)")
    store.wipe()
    store.user_name = USER  # display name used in prompts
    app = build_companion(store)

    for n, (today, messages) in enumerate(SESSIONS, 1):
        rule(f"Session {n} — {today}")
        episode_id = store.start_episode(on=today)
        hello = greeting(store, today)
        seed_thread(app, episode_id, hello)
        print(f"{COMPANION_NAME}: {hello}")
        config = {"configurable": {"thread_id": episode_id, "today": today}}
        for text in messages:
            print(f"\n{USER}: {text}")
            for update in app.stream({"messages": [HumanMessage(text)]}, config, stream_mode="updates"):
                node, out = next(iter(update.items()))
                if node == "respond":
                    print(f"{COMPANION_NAME}: {out['messages'][-1].content}")
                elif node == "remember":
                    for line in out["memory_log"]:
                        print(f"   ✎ {line}")
        s = summarize_episode(USER, store.episode_transcript(episode_id))
        store.end_episode(episode_id, s.summary if s else None, s.mood if s else None)
        if s:
            print(f"\n   ✎ episode: {s.summary} (mood: {s.mood})")

    today = SESSIONS[-1][0]
    rule("/memories (current)")
    for m in store.memories():
        print(f"  [{m['kind']}] {m['text']}  ★{m['importance']}" + (f" [{m['status']}]" if m["status"] else ""))
    rule("/people")
    for p in store.recall("people", today=today)["people"]:
        print(f"  {p['name']} — {p['relation']} ({p['kind']})")
    rule("/timeline")
    for t in store.timeline():
        print(f"  {t['date']}  [{t['kind']}] {t['text']}" + (f" [{t['status']}]" if t["status"] else ""))
    rule("Superseded facts")
    for m in store.memories(include_retired=True):
        hist = store.history(m["id"])
        if len(hist) > 1 and not m["retired"]:
            print("  now: " + hist[0]["text"])
            for h in hist[1:]:
                print("  was: " + h["text"])
    rule('recall("what should we do when my sister visits?")')
    print(format_context(store.recall("what should we do when my sister visits?", today=today), today))
    driver.close()


if __name__ == "__main__":
    main()
