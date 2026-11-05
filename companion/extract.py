"""Lesson 9 — LLM calls that turn conversation into structured memory."""
from __future__ import annotations

import datetime as dt
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from config import LLM_EXTRACTION_MODEL, chat_model
from companion.models import EpisodeSummary, Extraction

EXTRACT_SYSTEM = """\
You maintain the long-term biographical memory of an AI companion for {user}.
Read the latest exchange and record only what is worth remembering about {user}'s life.
Today is {today}.

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
- For a boundary, list the person or topic it protects in `about` (e.g. the ex's name).
- Only record what {user} said or clearly confirmed. Ignore the companion's guesses, small talk,
  questions, and anything already stored unless it CHANGED.
- If a new statement updates an existing memory, add the new memory and put the old id in
  `supersedes` (e.g. moved cities, new job, relationship changed). If something stopped being
  true with no replacement, put its id in `invalidate`.
- If an open plan has now happened or was cancelled, put its id in `completed_plans` (and add an
  `event` memory describing the outcome if one was shared).
- People, pets, places and organizations mentioned go in `entities` (reuse the exact names of
  known entities). Their link to {user} or to each other goes in `relations`
  (source 'user' for {user}; types like sister, mother, partner, friend, coworker, pet, lives_in,
  works_at, attends).
- importance: 9-10 core identity or major life events, 6-8 meaningful, 3-5 useful detail, 1-2 trivia.
- If nothing is worth remembering, return empty lists. Never invent details.

Known entities: {entities}

Existing memories you may update (id: [kind] text):
{candidates}"""


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


def extract_memories(user: str, user_msg: str, assistant_msg: str, candidates: list[dict],
                     entities: list[str], today: dt.date | None = None) -> Extraction:
    today = today or dt.date.today()
    system = EXTRACT_SYSTEM.format(
        user=user,
        today=f"{today.isoformat()} ({today.strftime('%A')})",
        entities=", ".join(entities) or "(none yet)",
        candidates="\n".join(
            f"{c['id']}: [{c['kind']}{'/' + c['status'] if c.get('status') else ''}] {c['text']}"
            for c in candidates) or "(none yet)",
    )
    exchange = f"{user}: {user_msg}\n\nCompanion: {assistant_msg}"
    llm = chat_model(LLM_EXTRACTION_MODEL, temperature=0)
    try:
        return _structured(llm, Extraction, [SystemMessage(system), HumanMessage(exchange)])
    except Exception as e:  # never let memory extraction break the conversation
        print(f"  (memory extraction failed: {type(e).__name__}: {e})")
        return Extraction()


def summarize_episode(user: str, transcript: str) -> EpisodeSummary | None:
    if not transcript.strip():
        return None
    llm = chat_model(LLM_EXTRACTION_MODEL, temperature=0)
    messages = [
        SystemMessage(f"Summarize this conversation between {user} and their AI companion "
                      f"for the companion's diary. Write in third person about {user}. "
                      f"Today is {dt.date.today().isoformat()}; write absolute dates, never "
                      f"'next Thursday' or 'yesterday'. Do not mention topics {user} asked "
                      f"the companion to avoid — just note that a boundary was set."),
        HumanMessage(transcript),
    ]
    try:
        return _structured(llm, EpisodeSummary, messages)
    except Exception as e:
        print(f"  (episode summary failed: {type(e).__name__}: {e})")
        return None
