"""Lesson 7 — a LangGraph agent that answers questions with tools over the team graph.

    uv run python team_agent.py                       # interactive chat
    uv run python team_agent.py "Who can back up Eve?"  # one question, then exit

The graph is the classic tool loop, built by hand so every piece is visible:

    START -> agent --(tool calls?)--> tools -> agent ... --(no tool calls)--> END
"""
import json
import sys

from fastembed import TextEmbedding
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from neo4j import GraphDatabase, unit_of_work

from config import LOCAL_EMBED_MODEL, NEO4J_AUTH, NEO4J_URI, chat_model

driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
_embedder = None


def read(query: str, limit: int | None = None, **params) -> list[dict]:
    """Run Cypher in a READ transaction: the server rejects any write inside it.

    `limit` streams at most that many rows (the rest is never fetched), and the 10 s
    timeout stops runaway queries such as an accidental cartesian product."""
    @unit_of_work(timeout=10)
    def work(tx):
        result = tx.run(query, **params)
        return [r.data() for r in (result.fetch(limit) if limit else result)]

    with driver.session(database="neo4j") as session:
        return session.execute_read(work)


# --- Fixed, parameterized tools: predictable and safe (prefer these) -----------------

@tool
def find_experts(skill: str) -> str:
    """List people who have a skill, strongest first, with their current projects.
    skill: a skill name as stored, e.g. Python, TypeScript, ML, UX, Rust (case-insensitive)."""
    rows = read(
        """
        MATCH (p:Person)-[h:HAS_SKILL]->(s:Skill) WHERE toLower(s.name) = toLower($skill)
        OPTIONAL MATCH (p)-[:WORKS_ON]->(pr:Project {status:'active'})
        RETURN p.name AS person, h.level AS level, collect(pr.name) AS active_projects
        ORDER BY level DESC
        """,
        skill=skill,
    )
    return json.dumps(rows) if rows else f"Nobody has the skill '{skill}'."


@tool
def person_profile(name: str) -> str:
    """Everything directly connected to one person: role, skills, projects, manager, contacts.
    name: the person's first name as stored, e.g. "Eve"."""
    rows = read(
        """
        MATCH (p:Person {name: $name})
        OPTIONAL MATCH (p)-[h:HAS_SKILL]->(s)
        WITH p, collect(s.name + ' (level ' + toString(h.level) + ')') AS skills
        OPTIONAL MATCH (p)-[:WORKS_ON]->(pr)
        WITH p, skills, collect(pr.name) AS projects
        OPTIONAL MATCH (m)-[:MANAGES]->(p)
        WITH p, skills, projects, collect(m.name) AS managers
        OPTIONAL MATCH (p)-[:KNOWS]-(k)
        RETURN p.name AS name, p.role AS role, p.description AS bio, skills, projects,
               managers, collect(k.name) AS knows
        """,
        name=name,
    )
    return json.dumps(rows[0]) if rows else f"No person named '{name}'."


@tool
def staffing_gaps() -> str:
    """Skills that active projects require but at most one person has (single points of failure)."""
    return json.dumps(read(
        """
        MATCH (pr:Project {status:'active'})-[:REQUIRES]->(s:Skill)
        OPTIONAL MATCH (s)<-[:HAS_SKILL]-(p:Person)
        WITH pr, s, collect(p.name) AS who WHERE size(who) < 2
        RETURN pr.name AS project, s.name AS skill, who
        """
    ))


@tool
def semantic_search(text: str) -> str:
    """Find people/projects whose descriptions are closest in meaning to the text."""
    global _embedder
    _embedder = _embedder or TextEmbedding(LOCAL_EMBED_MODEL)
    vector = next(_embedder.embed([text])).tolist()
    return json.dumps(read(
        """
        CALL db.index.vector.queryNodes('embedded_vec', 4, $v) YIELD node, score
        RETURN labels(node)[0] AS type, node.name AS name, round(score, 3) AS score,
               node.description AS description
        """,
        v=vector,
    ))


# --- Free-form Cypher: flexible, so it runs read-only with a row cap ----------------

@tool
def run_cypher(query: str) -> str:
    """Run a READ-ONLY Cypher query for questions the other tools can't answer.
    At most 25 rows come back: use count() / aggregation for totals.
    Schema: (:Person {name, role, description})-[:HAS_SKILL {level}]->(:Skill {name}),
    (:Person)-[:WORKS_ON]->(:Project {name, status, description}),
    (:Project)-[:REQUIRES]->(:Skill), (:Person)-[:MANAGES]->(:Person),
    (:Person)-[:KNOWS {since}]->(:Person), (:Person)-[:HAS_ROLE]->(:Role {name}).
    Person.role and HAS_ROLE hold the same job title (Lesson 3 copied one into the other)."""
    try:
        return json.dumps(read(query, limit=25), default=str)
    except Exception as e:  # syntax errors and attempted writes come back to the model
        return f"Query failed: {e}"


TOOLS = [find_experts, person_profile, staffing_gaps, semantic_search, run_cypher]
SYSTEM = SystemMessage(
    "You answer questions about a software team using the tools, which read a Neo4j graph. "
    "Prefer the specific tools; use run_cypher only when none fits. "
    "Base every claim on tool results and name the people, skills and projects involved."
)


def build_graph():
    llm = chat_model(temperature=0).bind_tools(TOOLS)

    def agent(state: MessagesState):
        return {"messages": [llm.invoke([SYSTEM, *state["messages"]])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=InMemorySaver())  # remembers the chat within a thread


def ask(app, question: str, thread: str = "cli"):
    """Stream one turn, printing each tool call and the final answer."""
    config = {"configurable": {"thread_id": thread}}
    for update in app.stream({"messages": [HumanMessage(question)]}, config, stream_mode="updates"):
        for node, payload in update.items():
            msg = payload["messages"][-1]
            if node == "agent" and msg.tool_calls:
                for call in msg.tool_calls:
                    print(f"  → {call['name']}({json.dumps(call['args'])})")
            elif node == "tools":
                for m in payload["messages"]:
                    print(f"  ← {m.name}: {m.content[:160]}{'…' if len(m.content) > 160 else ''}")
            elif node == "agent":
                print(f"\nAgent: {msg.content}\n")


if __name__ == "__main__":
    app = build_graph()
    try:
        if len(sys.argv) > 1:
            ask(app, " ".join(sys.argv[1:]))
        else:
            print("Ask about the team (Ctrl-D or 'quit' to exit).")
            while (q := input("You: ").strip()) not in ("quit", "exit"):
                if q:
                    ask(app, q)
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        driver.close()
