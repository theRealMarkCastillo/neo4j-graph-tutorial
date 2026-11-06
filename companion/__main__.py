"""Terminal chat with the companion:   .venv/bin/python -m companion [--user NAME] [--debug]

Commands:  /memories [kind]  /all  /people  /timeline  /history <id>  /forget <id>
           /forget-entity <name or id>  /recall <text>  /consolidate  /ontology  /debug  /new  /help  /quit
"""
import argparse
import datetime as dt

from langchain_core.messages import HumanMessage
from neo4j import GraphDatabase

from config import COMPANION_NAME, COMPANION_USER, NEO4J_AUTH, NEO4J_URI
from companion import ontology
from companion.extract import find_pattern
from companion.graph import build_companion, close_episode, greeting, seed_thread
from companion.store import MemoryStore, format_context

DIM, RESET, BOLD = "\033[2m", "\033[0m", "\033[1m"


def end_session(store: MemoryStore, episode_id: str, today: dt.date) -> None:
    print(f"{DIM}  (saving this conversation...){RESET}")
    for line in close_episode(store, episode_id, today):
        print(f"{DIM}  ✎ {line}{RESET}")


def show_command(store: MemoryStore, cmd: str, arg: str, today: dt.date) -> None:
    if cmd in ("/memories", "/all"):
        rows = store.memories(kind=arg or None, include_retired=cmd == "/all", today=today)
        for m in rows:
            flag = " (retired)" if m["retired"] else (f" [{m['status']}]" if m["status"] else "")
            if m["valid_until"]:
                flag += f" ({'expired' if m['expired'] else 'until'} {m['valid_until'].to_native():%m-%d %H:%M})"
            flag += (" 🔒" if m["sensitive"] else "") + (f" ↻{m['annual_date']}" if m["annual_date"] else "")
            print(f"  {m['id']}  [{m['kind']}] {m['text']}  ★{m['importance']}{flag}")
        print(f"  {len(rows)} memories")
    elif cmd == "/people":
        for p in store.recall(arg or "people", today=today)["people"]:
            print(f"  {p['name']} — {p['relation']} ({p['kind']})  {DIM}{p['id']}{RESET}")
    elif cmd == "/timeline":
        for t in store.timeline():
            flag = " (retired)" if t["retired"] else (f" [{t['status']}]" if t["status"] else "")
            print(f"  {t['date']}  [{t['kind']}] {t['text']}{flag}")
    elif cmd == "/history":
        for i, h in enumerate(store.history(arg)):
            print(f"  {'now ' if i == 0 else 'was '} {h['text']}")
    elif cmd == "/forget":
        print("  deleted" if store.forget(arg) else "  no such memory")
    elif cmd == "/forget-entity":
        gone = store.forget_entity(arg)
        print(f"  forgot {gone['name']}: {gone['memories']} memories deleted, {gone['turns']} turns and "
              f"{gone['episodes']} summaries redacted" if gone else "  no such entity")
    elif cmd == "/ontology":
        print("\n".join("  " + line for line in ontology.tree()))
        for t in store._q("MATCH (t:RelType {proposed: true}) RETURN t.name AS name, t.closest AS hint, t.examples AS ex"):
            print(f"  proposed: {t['name']}  (closest: {t['hint']})  e.g. {t['ex'][-1]}")
    elif cmd == "/consolidate":
        for line in store.consolidate(today, writer=find_pattern) or ["nothing to do"]:
            print(f"  {line}")
    elif cmd == "/recall":
        print(format_context(store.recall(arg, today=today), today))


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a companion that remembers.")
    parser.add_argument("--user", default=COMPANION_USER)
    parser.add_argument("--debug", action="store_true", help="show the memory block each turn")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    store = MemoryStore(driver, args.user)
    app = build_companion(store)
    debug, today = args.debug, dt.date.today()

    def new_session() -> dict:
        episode_id = store.start_episode()
        hello = greeting(store, today)
        seed_thread(app, episode_id, hello)
        print(f"\n{BOLD}{COMPANION_NAME}:{RESET} {hello}\n")
        return {"configurable": {"thread_id": episode_id, "today": today}}

    print(f"{DIM}Chatting as {args.user}. /help for commands.{RESET}")
    config = new_session()
    try:
        while True:
            text = input(f"{BOLD}{args.user}:{RESET} ").strip()
            if not text:
                continue
            if text.startswith("/"):
                cmd, _, arg = text.partition(" ")
                if cmd in ("/quit", "/exit"):
                    break
                if cmd == "/help":
                    print(__doc__)
                elif cmd == "/debug":
                    debug = not debug
                    print(f"  debug {'on' if debug else 'off'}")
                elif cmd == "/new":
                    end_session(store, config["configurable"]["thread_id"], today)
                    config = new_session()
                else:
                    show_command(store, cmd, arg.strip(), today)
                continue
            for update in app.stream({"messages": [HumanMessage(text)]}, config, stream_mode="updates"):
                node, out = next(iter(update.items()))
                if node == "recall" and debug:
                    print(f"{DIM}--- memory ---\n{out['memory_context']}\n--------------{RESET}")
                elif node == "respond":
                    print(f"\n{BOLD}{COMPANION_NAME}:{RESET} {out['messages'][-1].content}\n")
                elif node == "remember":
                    for line in out["memory_log"]:
                        print(f"{DIM}  ✎ {line}{RESET}")
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        end_session(store, config["configurable"]["thread_id"], today)
        driver.close()


if __name__ == "__main__":
    main()
