"""Read stored conversations again with the CURRENT extraction rules.

    uv run python -m companion.reextract --user Mark              every episode, oldest first
    uv run python -m companion.reextract --user Mark --dry-run    show what would be added, write nothing
    uv run python -m companion.reextract --user Mark --last 3     only the three most recent episodes

Memories are extracted when they're said, so a better prompt only helps from then on. After
the prompt learns something new (named things someone likes became entities, Logan -likes->
Minecraft), this runs the end-of-episode pass over each stored transcript. Existing matches are usually reinforced by the write path, but this is not a guarantee of
idempotence: different extraction output or changed graph state can produce new items.
Review a dry run before applying; apply can also reconcile entities and retire old facts. Each episode is read on its own date, so
"next Friday" resolves as it did then. Refresh uses one LLM call per episode with at least one user message.
"""
from __future__ import annotations

import sys

from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

from config import NEO4J_AUTH, NEO4J_URI
from companion.graph import episode_pass
from companion.store import MemoryStore
from companion.preview import format_preview, preview_extraction


def episodes(store: MemoryStore, last: int | None = None) -> list[dict]:
    rows = store._q(
        """
        MATCH (ep:Episode {user: $user})
        RETURN ep.id AS id, date(ep.started_at) AS day, ep.agent AS agent ORDER BY ep.started_at
        """
    )
    return rows[-last:] if last else rows


def reextract(driver, user: str, dry_run: bool = False, last: int | None = None) -> list[str]:
    if dry_run:
        try:
            return _dry_run(driver, user, last)
        except ClientError as exc:
            if "index" not in str(exc).lower() and "schema" not in str(exc).lower():
                raise
            raise RuntimeError("Dry run requires existing schema/vector indexes; run the normal setup first. "
                               "The preview will not initialize or repair the database.") from exc
    log = []
    for ep in episodes(MemoryStore(driver, user), last):
        store = MemoryStore(driver, user, agent=ep["agent"])
        day = ep["day"].to_native()
        log += [f"{ep['id']} ({day}) {l}" for l in episode_pass(store, ep["id"], day, refresh=True)]
    return log


def _dry_run(driver, user: str, last: int | None) -> list[str]:
    log = [f"current-graph preview for {user}: no database writes",
           "Each episode is compared with the current graph, not with earlier proposals. "
           "A later apply reruns extraction; model output, intervening writes, and earlier applied "
           "episodes can change its result."]
    with MemoryStore.read_view(driver, user) as view:
        selected = episodes(view, last)
    if not selected:
        return [*log, "no episodes to re-extract"]
    for ep in selected:
        with MemoryStore.read_view(driver, user, agent=ep["agent"]) as store:
            day = ep["day"].to_native()
            log.append(f"{ep['id']} ({day}, persona={store.agent})")
            ex = episode_pass(store, ep["id"], day, apply=False, refresh=True)
            if ex is None:
                log.append("  skipped: no eligible user turns")
                continue
            items = preview_extraction(store, ex, ep["id"], day)
            log += format_preview(items) if items else ["  no candidates"]
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
