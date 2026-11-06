"""Lessons 9 and 11 — LLM calls that turn conversation into structured memory.

Two extraction passes share one output schema (`Extraction`):
- per turn:       the latest exchange, with the few turns before it as read-only context
- end of episode: the whole transcript, to catch what only became clear across turns
"""
from __future__ import annotations

import datetime as dt
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from config import LLM_EXTRACTION_MODEL, chat_model
from companion.models import EpisodeSummary, Extraction

RULES = """\
Remember:
- fact: stable biographical facts (home, job, family, health conditions, history, identity)
- preference: likes, dislikes, values, how they like to be supported
- event: things that happened, with occurred_on as an ISO date when it can be worked out
- plan: upcoming intentions or appointments, with due_on as an ISO date ("Friday" -> the next Friday)
- feeling: notable emotional states and what caused them (these fade, so be selective)
- boundary: topics to avoid or ways they do NOT want to be treated

Rules:
- Each memory text is ONE standalone third-person sentence using names, never "I"/"you"/"he"
  ("{user}'s sister Maya is a nurse in Denver", not "she works there").
- Write dates as absolute dates in the text ("on 2026-11-12"), never relative words like
  "next Thursday", "yesterday" or "last month" — the text must still be true next year.
- Birthdays, anniversaries and other yearly dates: set annual_date to "MM-DD".
- sensitive = true for health (physical or mental), diagnoses, medication, sexuality, religion,
  finances/debt, legal trouble, and trauma.
- ANY request not to mention someone or something — "don't bring her up", "I don't want to
  talk about my dad", "please stop asking about work" — MUST become a `boundary` memory
  (importance 9), with the protected person or topic in `about`. Resolve pronouns from the
  earlier turns ("her" -> the person just discussed). Never skip one.
- Only record what {user} said or clearly confirmed. Ignore the companion's guesses, small talk,
  questions, and anything already stored unless it CHANGED.
- Don't echo: when {user} asks about their own past and the companion answers from memory,
  that is recall, not new information — record nothing for it.
- If a new statement updates or corrects an existing memory (moved cities, new job, a plan's
  date changed, "actually it was Tuesday"), add the new memory and put the old id in
  `supersedes`. If something stopped being true with no replacement, put its id in `invalidate`.
  A change is usually BOTH: an event ("moved to Chicago on 2026-11-13") AND a new current fact
  ("lives in Chicago") whose `supersedes` lists the old fact ("lives in Austin"). Never leave
  an old fact current after the user said it changed.
- If an open plan has now happened or was cancelled, put its id in `completed_plans` (and add an
  `event` memory describing the outcome if one was shared).
- People, pets, places and organizations mentioned go in `entities`. For a KNOWN entity set
  `ref` to its id. If someone shares a name with a known entity but is clearly a different
  person (a coworker Sam vs a cousin Sam), set new=true and give a distinguishing summary.
  In `about`, `source` and `target`, use the known id when two entities share a name.
- Links to {user} or between entities go in `relations` (source 'user' for {user}; types like
  sister, mother, partner, ex, friend, coworker, pet, lives_in, works_at). When a relationship
  ENDS (breakup, divorce, moved away, left a job), add it with ended=true, plus the new one
  if there is one (e.g. partner ended=true, then ex).
- If {user} explicitly asks you to forget a person or thing entirely, put its name or id in
  `forget_entities` and record nothing else about it.
- importance: 9-10 core identity or major life events, 6-8 meaningful, 3-5 useful detail, 1-2 trivia.
- If nothing is worth remembering, return empty lists. Never invent details.

Known entities (id: name (kind, relation to {user}) — summary):
{entities}

Existing memories you may update (id: [kind] text):
{candidates}"""

TURN_SYSTEM = """\
You maintain the long-term biographical memory of an AI companion for {user}.
Record what is worth remembering about {user}'s life from the LATEST exchange.
Earlier turns are given only as context — to work out who "she", "it" or "that" refers to and
to combine details spread across turns. Do not re-record something from earlier turns unless
the latest exchange adds to or changes it.
Today is {today}.

""" + RULES

EPISODE_SYSTEM = """\
You maintain the long-term biographical memory of an AI companion for {user}.
Below is a whole conversation that has just ended. Memories were already extracted turn by
turn (they are in the existing memories list). Your job is a careful second pass:
- add what was MISSED — facts that only became clear across several turns, or details that
  were spread out;
- fix mistakes in what was stored, using `supersedes` (e.g. a pronoun resolved to the wrong
  person, or a date that was later corrected);
- don't repeat memories that are already stored correctly. Returning empty lists is fine.
Only {user}'s own lines are a source of facts. The companion's lines often repeat what it
already remembers (including old, replaced facts) — never record those. If {user}'s lines only
ask questions or chat, return empty lists.
Today is {today}.

""" + RULES

CONTEXT_TURNS = 6  # earlier messages shown to the per-turn extractor


def _structured(llm, schema: type[BaseModel], messages) -> BaseModel:
    """Tool-calling structured output, falling back to plain JSON for endpoints without tools."""
    try:
        return llm.with_structured_output(schema, method="function_calling").invoke(messages)
    except Exception:
        hint = SystemMessage(
            "Reply with ONLY a JSON object matching this JSON Schema, no prose:\n"
            + json.dumps(schema.model_json_schema())
        )
        text = llm.invoke([*messages, hint]).content
        match = re.search(r"\{.*\}", text, re.S)
        return schema.model_validate_json(match.group(0) if match else "{}")


def _system(template: str, user: str, candidates: list[dict], entities: list[str], today: dt.date) -> str:
    return template.format(
        user=user,
        today=f"{today.isoformat()} ({today.strftime('%A')})",
        entities="\n".join(entities) or "(none yet)",
        candidates="\n".join(
            f"{c['id']}: [{c['kind']}{'/' + c['status'] if c.get('status') else ''}] {c['text']}"
            for c in candidates) or "(none yet)",
    )


def _run(system: str, content: str) -> Extraction:
    llm = chat_model(LLM_EXTRACTION_MODEL, temperature=0)
    try:
        return _structured(llm, Extraction, [SystemMessage(system), HumanMessage(content)])
    except Exception as e:  # never let memory extraction break the conversation
        print(f"  (memory extraction failed: {type(e).__name__}: {e})")
        return Extraction()


# Phrases that almost always set a boundary. Used as a tripwire, not as the extractor.
BOUNDARY_HINT = re.compile(
    r"\b(don'?t|do not|never|stop|please don'?t)\b.{0,25}\b(bring|mention|talk|ask|raise|remind)"
    r"|\bdon'?t want to (talk|hear|discuss)", re.I)


def extract_memories(user: str, user_msg: str, assistant_msg: str, candidates: list[dict],
                     entities: list[str], today: dt.date | None = None,
                     earlier: list[tuple[str, str]] = ()) -> Extraction:
    """Per-turn pass. `earlier` = (speaker, text) pairs before the latest exchange.

    Boundaries are safety-critical, so a regex tripwire double-checks the LLM: if the message
    looks like a boundary and none was extracted, extract once more with an explicit hint."""
    today = today or dt.date.today()
    context = "\n".join(f"{who}: {text}" for who, text in earlier[-CONTEXT_TURNS:])
    content = (f"Earlier in this conversation (context only):\n{context}\n\n" if context else "") + \
              f"LATEST EXCHANGE:\n{user}: {user_msg}\n\nCompanion: {assistant_msg}"
    system = _system(TURN_SYSTEM, user, candidates, entities, today)
    ex = _run(system, content)
    if BOUNDARY_HINT.search(user_msg) and not any(m.kind == "boundary" for m in ex.memories):
        retry = _run(system, content + f"\n\nNOTE: {user}'s latest message sets a BOUNDARY. "
                     "Record it as a boundary memory, with who or what it protects in `about`.")
        boundaries = [m for m in retry.memories if m.kind == "boundary"]
        ex.memories += boundaries
        ex.entities += [e for e in retry.entities if e.name not in {x.name for x in ex.entities}]
    return ex


def extract_episode(user: str, transcript: str, candidates: list[dict], entities: list[str],
                    today: dt.date | None = None) -> Extraction:
    """End-of-episode pass over the whole transcript."""
    if not transcript.strip():
        return Extraction()
    today = today or dt.date.today()
    return _run(_system(EPISODE_SYSTEM, user, candidates, entities, today), transcript)


def summarize_episode(user: str, transcript: str, today: dt.date | None = None) -> EpisodeSummary | None:
    if not transcript.strip():
        return None
    today = today or dt.date.today()
    llm = chat_model(LLM_EXTRACTION_MODEL, temperature=0)
    messages = [
        SystemMessage(f"Summarize this conversation between {user} and their AI companion "
                      f"for the companion's diary. Write in third person about {user}. "
                      f"Today is {today.isoformat()}; write absolute dates, never "
                      f"'next Thursday' or 'yesterday'. Do not mention topics {user} asked "
                      f"the companion to avoid or forget, or sensitive health details — just "
                      f"note that a boundary was set or a personal topic came up. '[forgotten]' "
                      f"marks text the user had erased: ignore it and don't mention it."),
        HumanMessage(transcript),
    ]
    try:
        return _structured(llm, EpisodeSummary, messages)
    except Exception as e:
        print(f"  (episode summary failed: {type(e).__name__}: {e})")
        return None
