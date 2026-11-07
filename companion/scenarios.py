"""Lesson 11 — scenario tests for companion memory.

    uv run python -m companion.scenarios              # all scenarios
    uv run python -m companion.scenarios forget sam   # only those whose name contains a word

Each scenario runs scripted sessions through the real LangGraph companion (real LLM calls,
simulated dates), then checks the graph and the companion's replies. Every scenario uses its
own user ("eval <name>") and wipes it first. LLM output varies, so treat a failure as
"look at the transcript", and run more than once before trusting a pass rate.
"""
from __future__ import annotations

import datetime as dt
import sys
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.messages import HumanMessage
from neo4j import GraphDatabase

import config
import observability
from config import NEO4J_AUTH, NEO4J_URI
from companion.graph import build_companion, close_episode, greeting, seed_thread
from companion.embeddings import PROFILE
from companion.store import MemoryStore, format_context

D = dt.date


@dataclass
class Run:
    """What happened in a scenario, for the checks to inspect."""
    store: MemoryStore
    stores: dict[str, MemoryStore] = field(default_factory=dict)  # by persona (Lesson 16)
    greetings: list[str] = field(default_factory=list)    # one per session
    replies: list[list[str]] = field(default_factory=list)  # per session, per message
    extracted: list[list[bool]] = field(default_factory=list)  # did `remember` run for each message?

    def q(self, cypher: str, **params) -> list[dict]:
        return self.store._q(cypher, **params)

    def memories(self) -> list[dict]:
        return self.store.memories()

    def mentions(self, text: str, *words: str) -> bool:
        return any(w.lower() in text.lower() for w in words)


@dataclass
class Scenario:
    name: str
    about: str
    sessions: list[tuple]  # (date, messages) or (date, messages, persona)
    checks: list[tuple[str, Callable[[Run], bool]]]


def about_entity(run: Run, name: str, word: str, relation: str | None = None) -> list[dict]:
    """Current memories ABOUT an entity named `name` (optionally with this relation to the user)
    whose text contains `word`."""
    return run.q(
        """
        MATCH (m:Memory {user: $user})-[:ABOUT]->(e:Entity {name: $name}) WHERE m.invalidated_at IS NULL
          AND toLower(m.text) CONTAINS toLower($word)
          AND ($rel IS NULL OR EXISTS { (:User {id: $user})-[r:RELATED {type: $rel}]->(e) })
        RETURN m.id AS id, m.text AS text, m.due_on AS due_on
        """,
        name=name, word=word, rel=relation,
    )


SCENARIOS = [
    Scenario(
        "cross_turn_facts", "A fact split over two turns ('she' in turn 2) + a small-talk turn skipped",
        [(D(2026, 11, 2), [
            "My best friend from college, Priya, called me today.",
            "She just got engaged! The wedding is on June 12 next year.",
            "haha yeah",
        ])],
        [
            ("Priya exists as a person", lambda r: bool(r.q("MATCH (e:Entity {user:$user, name:'Priya', kind:'person'}) RETURN e"))),
            ("a memory about Priya says she got engaged", lambda r: bool(about_entity(r, "Priya", "engaged"))),
            # by the date field, not the word: "Priya is getting married" (due 2027-06-12) passes
            ("the wedding is dated 2027-06-12", lambda r: any(
                str(m.get("due_on")) == "2027-06-12" or "2027-06-12" in m["text"] for m in about_entity(r, "Priya", ""))),
            ("'haha yeah' skipped the extraction call", lambda r: r.extracted[0] == [True, True, False]),
        ],
    ),
    Scenario(
        "correction", "A plan's date is corrected in the same conversation",
        [(D(2026, 11, 9), [
            "I have a dentist appointment on Thursday.",
            "Oh wait, I just checked — it's actually on Tuesday the 17th.",
        ])],
        [
            ("exactly one open dentist plan", lambda r: len([m for m in r.memories()
                if m["kind"] == "plan" and m["status"] == "open" and "dentist" in m["text"].lower()]) == 1),
            ("it is due 2026-11-17", lambda r: bool(r.q(
                "MATCH (m:Memory {user:$user, kind:'plan', status:'open'}) WHERE m.invalidated_at IS NULL "
                "AND toLower(m.text) CONTAINS 'dentist' AND m.due_on = date('2026-11-17') RETURN m"))),
        ],
    ),
    Scenario(
        "history", "A fact changes; later the user asks about the past",
        [
            (D(2026, 11, 2), ["I live in Austin, I've been here about five years."]),
            (D(2026, 11, 20), ["Big news: I moved to Chicago last week!"]),
            (D(2026, 11, 25), ["Remind me, where did I live before Chicago?"]),
        ],
        [
            ("current memory: lives in Chicago", lambda r: any("chicago" in m["text"].lower() for m in r.memories())),
            ("the Austin memory is retired, not deleted", lambda r: bool(r.q(
                "MATCH (m:Memory {user:$user}) WHERE m.invalidated_at IS NOT NULL AND m.text CONTAINS 'Austin' RETURN m"))),
            # Only current *facts* count: "moved to Austin five years ago" is a true past event.
            ("no current fact says they live in Austin now", lambda r: not any(
                m["kind"] == "fact" and "austin" in m["text"].lower() and "chicago" not in m["text"].lower()
                for m in r.memories())),
            ("asking about the past stored nothing new (no echo)", lambda r: not r.q(
                "MATCH (m:Memory {user:$user})-[:FROM]->(ep:Episode) WHERE date(ep.started_at) = date('2026-11-25') RETURN m")),
            ("the reply answers 'Austin'", lambda r: r.mentions(r.replies[2][0], "Austin")),
        ],
    ),
    Scenario(
        "relationship_change", "Partner becomes an ex and a boundary; nothing about them resurfaces",
        [
            (D(2026, 11, 2), ["My girlfriend Dana and I are planning a trip to Lisbon in March."]),
            (D(2026, 11, 16), ["Dana and I broke up last weekend.", "Please don't bring her up anymore, it hurts."]),
            (D(2026, 11, 30), ["What should I do this weekend? Any trip ideas?"]),
        ],
        [
            ("no active romantic relation to Dana", lambda r: not r.q(
                "MATCH (:User {id:$user})-[r:RELATED]->(:Entity {name:'Dana'}) WHERE r.until IS NULL "
                "AND r.type IN ['partner','girlfriend','boyfriend','dating','spouse'] RETURN r")),
            ("a boundary memory is ABOUT Dana", lambda r: bool(r.q(
                "MATCH (b:Memory {user:$user, kind:'boundary'})-[:ABOUT]->(:Entity {name:'Dana'}) "
                "WHERE b.invalidated_at IS NULL RETURN b"))),
            ("Dana is not in the people list", lambda r: all(
                p["name"] != "Dana" for p in r.store.recall("people", today=D(2026, 11, 30))["people"])),
            ("session 3 never mentions Dana or Lisbon", lambda r: not r.mentions(
                r.greetings[2] + " " + r.replies[2][0], "Dana", "Lisbon")),
        ],
    ),
    Scenario(
        "same_name", "Two different people named Sam stay separate",
        [
            (D(2026, 11, 2), ["My coworker Sam keeps taking credit for my work. So annoying."]),
            (D(2026, 11, 9), ["My cousin Sam is getting married in May!"]),
            (D(2026, 11, 16), ["Sam from work did it again today, took credit for my slides in the team meeting."]),
        ],
        [
            ("two Sam entities", lambda r: r.q(
                "MATCH (e:Entity {user:$user, name:'Sam'}) RETURN count(e) AS n")[0]["n"] == 2),
            ("the wedding is about cousin Sam", lambda r: bool(about_entity(r, "Sam", "marr", "cousin"))),
            ("the slides incident is about coworker Sam", lambda r: bool(about_entity(r, "Sam", "slides", "coworker"))),
            ("no credit/slides memory is about cousin Sam", lambda r: not (
                about_entity(r, "Sam", "credit", "cousin") or about_entity(r, "Sam", "slides", "cousin"))),
        ],
    ),
    Scenario(
        "recurring_date", "A birthday comes up every year",
        [
            (D(2026, 11, 2), ["My mom's birthday is December 2nd. I always forget to plan something."]),
            (D(2026, 11, 28), ["Hey!"]),
        ],
        [
            ("stored with annual_date 12-02", lambda r: any(m["annual_date"] == "12-02" for m in r.memories())),
            ("the 2026-11-28 greeting mentions the birthday", lambda r: r.mentions(r.greetings[1], "birthday")),
            ("recall on 2027-11-25 lists it for 2027-12-02", lambda r: any(
                u["on"] == D(2027, 12, 2) for u in r.store.recall("hi", today=D(2027, 11, 25))["upcoming"])),
        ],
    ),
    Scenario(
        "forget_person", "'Forget everything about Jordan' removes them everywhere",
        [
            (D(2026, 11, 2), [
                "I just adopted a cat named Miso!",
                "Also my ex Jordan texted me again last night.",
                "Actually, please forget everything about Jordan. I don't want you remembering him at all.",
            ]),
            (D(2026, 11, 9), ["Who are the people and pets I've told you about?"]),
        ],
        [
            ("no Jordan entity", lambda r: not r.q("MATCH (e:Entity {user:$user}) WHERE e.name = 'Jordan' RETURN e")),
            ("no memory text mentions Jordan", lambda r: not any("jordan" in m["text"].lower() for m in r.store.memories(include_retired=True))),
            ("no stored turn mentions Jordan", lambda r: not r.q(
                "MATCH (:Episode {user:$user})-[:HAS_TURN]->(t) WHERE toLower(t.text) CONTAINS 'jordan' RETURN t")),
            ("no episode summary mentions Jordan", lambda r: not r.q(
                "MATCH (ep:Episode {user:$user}) WHERE toLower(ep.summary) CONTAINS 'jordan' RETURN ep")),
            ("Miso is still remembered", lambda r: bool(r.q("MATCH (e:Entity {user:$user, name:'Miso'}) RETURN e"))),
            ("session 2 doesn't mention Jordan", lambda r: not r.mentions(r.greetings[1] + r.replies[1][0], "Jordan")),
        ],
    ),
    Scenario(
        "family_multi_hop", "'Anyone in my family...' finds relatives through relation types",
        [
            (D(2026, 11, 2), [
                "My brother Leo is a paramedic in Seattle.",
                "And my aunt Rosa has been a pharmacist for twenty years.",
            ]),
            (D(2026, 11, 9), ["Is anyone in my family in healthcare? I have a question about mixing two medications."]),
        ],
        [
            ("Rosa is linked as a relative", lambda r: bool(r.q(
                "MATCH (:User {id:$user})-[:RELATED]->(:Entity {name:'Rosa'}) RETURN 1"))),
            ("recall surfaces Rosa and Leo", lambda r: {"Rosa", "Leo"} <= {e["name"] for e in r.store.recall(
                "Is anyone in my family in healthcare?", today=D(2026, 11, 9))["entities"]}),
            ("the reply suggests Rosa", lambda r: r.mentions(r.replies[1][0], "Rosa")),
        ],
    ),
    Scenario(
        "sensitive", "A diagnosis is stored as sensitive and only comes up when on-topic",
        [
            (D(2026, 11, 2), [
                "I was diagnosed with ADHD last year and I take medication for it.",
                "On a lighter note, I love hiking — Forest Park is my favorite.",
            ]),
            (D(2026, 11, 9), ["What should I do this weekend?"]),
        ],
        [
            ("the ADHD memory is marked sensitive", lambda r: any(
                m["sensitive"] for m in r.memories() if "adhd" in m["text"].lower())),
            ("it's not in the core profile", lambda r: not any(
                "adhd" in m["text"].lower() for m in r.store.recall("hi", today=D(2026, 11, 9))["core"])),
            ("session 2 greeting/reply don't mention it", lambda r: not r.mentions(
                r.greetings[1] + r.replies[1][0], "ADHD", "medication", "diagnos")),
            ("recall includes it when on-topic", lambda r: "ADHD" in format_context(r.store.recall(
                "I'm struggling to focus at work, my meds don't seem to help", today=D(2026, 11, 9)))),
        ],
    ),

    # ---- Lesson 12: ontology
    Scenario(
        "ontology_synonyms", "'boyfriend' is stored as the canonical partner; the breakup ends it; the pet stays",
        [
            (D(2026, 11, 2), ["My boyfriend Theo and I just adopted a puppy named Waffles!"]),
            (D(2026, 11, 16), ["Theo and I broke up last week. I'm keeping Waffles though."]),
        ],
        [
            ("no non-canonical type stored (boyfriend)", lambda r: not r.q(
                "MATCH (:User {id:$user})-[r:RELATED]->() WHERE r.type IN ['boyfriend','girlfriend','bf'] RETURN r")),
            ("the partner edge to Theo is ended", lambda r: bool(r.q(
                "MATCH (:User {id:$user})-[r:RELATED {type:'partner'}]->(:Entity {name:'Theo'}) WHERE r.until IS NOT NULL RETURN r"))
                and not r.q("MATCH (:User {id:$user})-[r:RELATED {type:'partner'}]->(:Entity {name:'Theo'}) WHERE r.until IS NULL RETURN r")),
            ("Waffles is still an active pet", lambda r: bool(r.q(
                "MATCH (:User {id:$user})-[r:RELATED {type:'pet'}]->(:Entity {name:'Waffles'}) WHERE r.until IS NULL RETURN r"))),
        ],
    ),
    Scenario(
        "ontology_cardinality", "Moving ends the old lives_in even if nobody says 'I left Boston'",
        [
            (D(2026, 11, 2), ["I live in Boston, in a tiny apartment near the river."]),
            (D(2026, 11, 20), ["Settling into my new place in Seattle — the move went smoothly!"]),
        ],
        [
            ("exactly one active lives_in", lambda r: r.q(
                "MATCH (:User {id:$user})-[r:RELATED {type:'lives_in'}]->() WHERE r.until IS NULL RETURN count(r) AS n")[0]["n"] == 1),
            ("and it is Seattle", lambda r: bool(r.q(
                "MATCH (:User {id:$user})-[r:RELATED {type:'lives_in'}]->(:Entity {name:'Seattle'}) WHERE r.until IS NULL RETURN r"))),
            ("Boston is kept as history", lambda r: bool(r.q(
                "MATCH (:User {id:$user})-[r:RELATED {type:'lives_in'}]->(:Entity {name:'Boston'}) WHERE r.until IS NOT NULL RETURN r"))),
        ],
    ),
    Scenario(
        "ontology_inference", "A brother's daughter is inferred to be a niece — nobody said 'niece'",
        [
            (D(2026, 11, 2), ["My brother Leo just had a baby girl! Her name is Ana."]),
            (D(2026, 11, 9), ["Who's the newest member of my family?"]),
        ],
        [
            ("Leo is linked as a brother", lambda r: bool(r.q(
                "MATCH (:User {id:$user})-[:RELATED {type:'brother'}]->(:Entity {name:'Leo'}) RETURN 1"))),
            ("Leo -daughter/child-> Ana is stored", lambda r: bool(r.q(
                "MATCH (:Entity {user:$user, name:'Leo'})-[r:RELATED]->(:Entity {name:'Ana'}) "
                "WHERE r.type IN ['child','daughter'] RETURN r"))),
            ("Ana is the user's niece — stored, or inferred via Leo", lambda r: bool(r.q(
                "MATCH (:User {id:$user})-[r:RELATED]->(:Entity {name:'Ana'}) WHERE r.type IN ['niece','niece_or_nephew'] RETURN r"))
                or any(i["name"] == "Ana" and i["relation"] == "niece_or_nephew" for i in r.store.inferred_relations())),
            ("the reply names Ana", lambda r: r.mentions(r.replies[1][0], "Ana")),
        ],
    ),
    Scenario(
        "personas", "What the user tells one persona privately never reaches another",
        [
            (D(2026, 11, 18), ["I signed up for the Chicago half marathon in March!",
                               "Just between us, I get really bad panic attacks before big races."], "Sage"),
            (D(2026, 11, 19), ["Can you help me plan my training for this week?"], "Coach"),
        ],
        [
            ("the panic attacks are private to Sage", lambda r: bool(r.q(
                """MATCH (m:Memory {user: $user}) WHERE toLower(m.text) CONTAINS 'panic'
                     AND m.private AND m.agent = 'Sage' RETURN m"""))),
            ("Coach knows about the half marathon", lambda r: "marathon" in format_context(
                r.stores["Coach"].recall("my training plan", today=D(2026, 11, 19)), D(2026, 11, 19)).lower()),
            ("nothing Coach sees mentions panic", lambda r: "panic" not in format_context(
                r.stores["Coach"].recall("how do I feel before big races?", today=D(2026, 11, 19)),
                D(2026, 11, 19)).lower()),
            ("Coach's replies use the race and never mention panic or anxiety", lambda r:
                r.mentions(" ".join(r.replies[1]), "marathon", "half", "race")
                and not r.mentions(" ".join(r.greetings[1:] + r.replies[1]), "panic", "anxi")),
        ],
    ),
    Scenario(
        "situational", "Short-lived statuses expire, get a follow-up, and roll up into a habit",
        [
            (D(2026, 11, 18), ["Heading to the gym, talk later!"]),
            (D(2026, 11, 19), ["Morning! I slept terribly though."]),
            (D(2026, 11, 20), ["Just got back from the gym, my legs are dead."]),
            (D(2026, 11, 21), ["Off to the gym again after work, then dinner."]),
            (D(2026, 11, 23), ["Gym again before work today, trying to keep the streak going."]),
        ],
        [
            ("the gym was stored as a status that expires", lambda r: bool(r.q(
                "MATCH (m:Memory {user:$user}) WHERE toLower(m.text) CONTAINS 'gym' AND m.valid_until IS NOT NULL RETURN m"))),
            ("the next day nothing says they're at the gym now", lambda r: not any(
                "gym" in m["text"].lower() and m["kind"] == "status" for m in r.store.memories(today=D(2026, 11, 19)))),
            ("the next day's greeting asks about the gym", lambda r: r.mentions(r.greetings[1], "gym", "workout")),
            ("three gym visits became one lasting habit", lambda r: bool(r.q(
                """MATCH (p:Memory {user:$user}) WHERE p.invalidated_at IS NULL AND p.evidence >= 3
                     AND p.kind IN ['preference', 'fact'] AND toLower(p.text) CONTAINS 'gym' RETURN p"""))),
        ],
    ),
]


def threshold_check() -> tuple[str, bool]:
    """Every scenario run also checks that the thresholds were measured on this extraction
    model's sentences (structure-over-prose spec). Fix a failure by re-running
    `python -m companion.calibrate` and the scenarios on the new model, then updating
    `extraction_model` on the profile."""
    used, measured = config.LLM_EXTRACTION_MODEL, PROFILE.extraction_model
    return (f"thresholds measured with this extraction model ({measured} vs {used})", used == measured)


def run_scenario(driver, sc: Scenario, verbose: bool = True) -> list[tuple[str, bool]]:
    store = MemoryStore(driver, f"eval {sc.name}")
    store.wipe()
    store.user_name = "Riley"  # display name in prompts
    run = Run(store, stores={store.agent: store})
    apps = {store.agent: build_companion(store)}
    for n, (today, messages, *persona) in enumerate(sc.sessions, 1):
        if persona and persona[0] not in run.stores:  # another persona, same user
            other = MemoryStore(driver, f"eval {sc.name}", agent=persona[0])
            other.user_name = "Riley"
            run.stores[persona[0]], apps[persona[0]] = other, build_companion(other)
        store = run.stores[persona[0] if persona else run.store.agent]
        app = apps[store.agent]
        episode_id = store.start_episode(on=today)
        hello = greeting(store, today)
        seed_thread(app, episode_id, hello)
        run.greetings.append(hello)
        run.replies.append([])
        run.extracted.append([])
        if verbose:
            print(f"  -- session {n} ({today})\n  {store.agent}: {hello}")
        config = {"configurable": {"thread_id": episode_id, "today": today}}
        for text in messages:
            if verbose:
                print(f"  Riley: {text}")
            ran = False
            for update in app.stream({"messages": [HumanMessage(text)]}, config, stream_mode="updates"):
                node, out = next(iter(update.items()))
                if node == "respond":
                    run.replies[-1].append(out["messages"][-1].content)
                    if verbose:
                        print(f"  {store.agent}: {out['messages'][-1].content}")
                elif node == "remember":
                    ran = True
                    if verbose:
                        for line in out["memory_log"]:
                            print(f"     ✎ {line}")
            run.extracted[-1].append(ran)
            if verbose and not ran:
                print("     (small talk — extraction skipped)")
        for line in close_episode(store, episode_id, today):
            if verbose:
                print(f"     ✎ {line}")
    results = []
    for desc, check in sc.checks:
        try:
            ok = bool(check(run))
        except Exception as e:  # a crashing check is a failing check
            print(f"     check error: {type(e).__name__}: {e}")
            ok = False
        results.append((desc, ok))
    results.append(threshold_check())
    return results


def main() -> None:
    words = [w.lower() for w in sys.argv[1:]]
    chosen = [s for s in SCENARIOS if not words or any(w in s.name for w in words)]
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    summary, rows = [], []
    # With MLflow tracing on: one run, one trace per scenario (every session, LLM call and
    # graph write in it), and each check's verdict attached to that trace.
    observability.start_eval_run("scenarios", scenarios=",".join(s.name for s in chosen))
    for sc in chosen:
        print(f"\n=== {sc.name}: {sc.about}")
        with observability.span(f"scenario {sc.name}", inputs={"about": sc.about}) as sp:
            results = run_scenario(driver, sc)
            if sp is not None:
                sp.set_outputs({desc: ok for desc, ok in results})
        for desc, ok in results:
            print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
            if sp is not None:
                observability.record_verdict(sp.trace_id, desc, ok, source=f"scenario {sc.name}")
            rows.append({"case": f"{sc.name}: {desc}", "outcome": "passed" if ok else "failed",
                         "trace_id": sp.trace_id if sp is not None else ""})
        summary.append((sc.name, sum(ok for _, ok in results), len(results)))
    observability.finish_eval_run(rows)
    print("\n=== Summary")
    for name, passed, total in summary:
        print(f"  {name:<22} {passed}/{total}")
    print(f"  {'total':<22} {sum(p for _, p, _ in summary)}/{sum(t for _, _, t in summary)}")
    driver.close()


if __name__ == "__main__":
    main()
