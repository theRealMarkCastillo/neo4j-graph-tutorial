"""Lesson 10 — the companion as a LangGraph:

    START -> recall -> respond -> remember -> END

- recall:   hybrid graph + vector retrieval from Neo4j (long-term memory)
- respond:  the LLM, given the persona, the memory block and recent turns
            (short-term memory = this thread's messages, kept by the checkpointer)
- remember: log both turns to the Episode, extract memories, write them to the graph

Per-call settings travel in config["configurable"]: thread_id (= episode id), today.
"""
from __future__ import annotations

import datetime as dt

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from config import COMPANION_NAME, chat_model
from companion.extract import extract_memories
from companion.store import MemoryStore, format_context

PERSONA = """\
You are {name}, a warm, thoughtful AI companion and assistant for {user}. Today is {today}.

You have a long-term memory of {user}'s life, shown below. Use it the way a good friend would:
- Weave memories in naturally. Never list them or say "according to my memory".
- Follow up on open plans when it fits, especially ones that are due or just passed.
- Respect every boundary strictly — do not raise those topics, even indirectly.
- Only rely on memories shown here. If you don't know something, ask instead of guessing.
- Be honest that you are an AI if it comes up. Be supportive, but for a crisis or medical,
  legal or financial decisions, encourage real-world help.
- Match {user}'s tone and keep replies conversational and fairly short.

=== What you remember about {user} ===
{memory}"""

RECENT_TURNS = 16  # short-term window sent to the LLM


class CompanionState(MessagesState):
    memory_context: str
    memory_log: list[str]


def _today(config: RunnableConfig) -> dt.date:
    return config["configurable"].get("today") or dt.date.today()


def build_companion(store: MemoryStore):
    llm = chat_model()

    def recall(state: CompanionState, config: RunnableConfig):
        query = state["messages"][-1].content
        found = store.recall(query, today=_today(config))
        return {"memory_context": format_context(found, _today(config))}

    def respond(state: CompanionState, config: RunnableConfig):
        today = _today(config)
        system = SystemMessage(PERSONA.format(
            name=COMPANION_NAME, user=store.user_name,
            today=f"{today.isoformat()} ({today.strftime('%A')})",
            memory=state["memory_context"],
        ))
        recent = state["messages"][-RECENT_TURNS:]
        while recent and not isinstance(recent[0], HumanMessage):  # start the window on a user turn
            recent = recent[1:]
        return {"messages": [llm.invoke([system, *recent])]}

    def remember(state: CompanionState, config: RunnableConfig):
        episode_id = config["configurable"]["thread_id"]
        user_msg = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)).content
        reply = state["messages"][-1].content
        store.log_turn(episode_id, store.user_name, user_msg)
        store.log_turn(episode_id, COMPANION_NAME, reply)
        extraction = extract_memories(
            store.user_name, user_msg, reply,
            candidates=store.extraction_candidates(user_msg),
            entities=store.known_entities(),
            today=_today(config),
        )
        return {"memory_log": store.apply(extraction, episode_id)}

    g = StateGraph(CompanionState)
    g.add_node("recall", recall)
    g.add_node("respond", respond)
    g.add_node("remember", remember)
    g.add_edge(START, "recall")
    g.add_edge("recall", "respond")
    g.add_edge("respond", "remember")
    g.add_edge("remember", END)
    return g.compile(checkpointer=InMemorySaver())


def greeting(store: MemoryStore, today: dt.date) -> str:
    """Opening line for a new session, so the companion can follow up proactively."""
    found = store.recall("how are things going, plans and recent events", today=today)
    if not any(found[k] for k in ("core", "semantic", "people", "episodes")):
        return f"Hi {store.user_name}, I'm {COMPANION_NAME}. I'd love to get to know you — how's your day going?"
    system = PERSONA.format(name=COMPANION_NAME, user=store.user_name,
                            today=f"{today.isoformat()} ({today.strftime('%A')})",
                            memory=format_context(found, today))
    prompt = (f"{store.user_name} just opened a new chat. Greet them in one or two sentences. "
              "If an open plan is due or has just passed, ask how it went; otherwise pick up "
              "naturally from your last conversation.")
    return chat_model().invoke([SystemMessage(system), HumanMessage(prompt)]).content


def seed_thread(app, thread_id: str, text: str) -> None:
    """Put the greeting into the thread's short-term memory as the companion's first turn."""
    app.update_state({"configurable": {"thread_id": thread_id}},
                     {"messages": [AIMessage(text)]}, as_node="respond")
