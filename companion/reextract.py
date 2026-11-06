"""Read stored conversations again with the CURRENT extraction rules.

    python -m companion.reextract --user Mark              every episode, oldest first
    python -m companion.reextract --user Mark --dry-run    show what would be added, write nothing
    python -m companion.reextract --user Mark --last 3     only the three most recent episodes

Memories are extracted when they're said, so a better prompt only helps from then on. After
the prompt learns something new (named things someone likes became entities, Logan -likes->
Minecraft), this runs the end-of-episode pass over each stored transcript. It's safe to run
twice: what's already stored is reinforced, not duplicated (the write path's dedupe), and
only new entities, links and memories are added. Each episode is read on its own date, so
"next Friday" resolves as it did then. Costs one LLM call per episode with 2+ user messages.
"""
from __future__ import annotations

import sys

from neo4j import GraphDatabase

from config import NEO4J_AUTH, NEO4J_URI
from companion.graph import episode_pass
from companion.store import MemoryStore


def episodes(store: MemoryStore, last: int | None = None) -> list[dict]:
    rows = store._q(
        """
        MATCH (ep:Episode {user: $user})
        RETURN ep.id AS id, date(ep.started_at) AS day, ep.agent AS agent ORDER BY ep.started_at
        """
    )
    return rows[-last:] if last else rows


def reextract(driver, user: str, dry_run: bool = False, last: int | None = None) -> list[str]:
    log = []
    for ep in episodes(MemoryStore(driver, user), last):
        store = MemoryStore(driver, user, agent=ep["agent"])  # as the persona that heard it
        day = ep["day"].to_native()
        if dry_run:
            ex = episode_pass(store, ep["id"], day, apply=False, refresh=True)
            if ex is None:
                continue
            log.append(f"{ep['id']} ({day}): entities {[e.name for e in ex.entities]}, "
                       f"links {[f'{r.source} -{r.type}-> {r.target}' for r in ex.relations]}, "
                       f"{len(ex.memories)} memories")
        else:
            log += [f"{ep['id']} ({day}) {l}" for l in episode_pass(store, ep["id"], day, refresh=True)]
    return log


def main() -> None:
    args = sys.argv[1:]
    if "--user" not in args:
        sys.exit(__doc__)
    user = args[args.index("--user") + 1]
    last = int(args[args.index("--last") + 1]) if "--last" in args else None
    with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        for l in reextract(driver, user, dry_run="--dry-run" in args, last=last):
            print(l)


if __name__ == "__main__":
    main()
