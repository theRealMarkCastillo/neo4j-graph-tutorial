"""Lesson 7 — LangGraph agent with graph tools.

Test an agent in two layers:
1. The tools are plain functions over the graph: test them directly, no LLM (fast, exact).
2. The agent's *choices* need the LLM: run real questions and grade the trajectory
   (which tools it called) and the answer with code. These are evals: opt in with -m llm.
"""
import json

import pytest

import team_agent as ta


def test_find_experts_is_case_insensitive_and_ranked(driver):
    rows = json.loads(ta.find_experts.invoke({"skill": "python"}))
    people = [r["person"] for r in rows]
    assert {"Alice", "Eve"} <= set(people)
    levels = [r["level"] for r in rows]
    assert levels == sorted(levels, reverse=True)


def test_tools_explain_empty_results(driver):
    # A tool should tell the model *why* it found nothing, so the model can recover.
    assert "Nobody has the skill" in ta.find_experts.invoke({"skill": "Basket weaving"})
    assert "No person named" in ta.person_profile.invoke({"name": "Zed"})


def test_person_profile(driver):
    p = json.loads(ta.person_profile.invoke({"name": "Eve"}))
    assert "Search Revamp" in p["projects"]
    assert any(s.startswith("ML") for s in p["skills"])


def test_run_cypher_refuses_writes(driver):
    out = ta.run_cypher.invoke({"query": "CREATE (:Person {name:'Mallory'}) RETURN 1"})
    assert out.startswith("Query failed")
    assert driver.execute_query("MATCH (p:Person {name:'Mallory'}) RETURN count(p) AS n").records[0]["n"] == 0


def test_run_cypher_caps_rows(driver):
    out = json.loads(ta.run_cypher.invoke({"query": "UNWIND range(1, 100) AS i RETURN i"}))
    assert len(out) == 25


def test_run_cypher_reports_syntax_errors(driver):
    assert ta.run_cypher.invoke({"query": "MATCH (n RETURN n"}).startswith("Query failed")


# --- LLM evals ---------------------------------------------------------------------------

def run_agent(question: str) -> tuple[list[str], str]:
    """Ask once; return (tool names called, final answer)."""
    from langchain_core.messages import HumanMessage

    app = ta.build_graph()
    out = app.invoke({"messages": [HumanMessage(question)]},
                     {"configurable": {"thread_id": f"pytest-{question}"}})
    tools = [c["name"] for m in out["messages"] for c in (getattr(m, "tool_calls", None) or [])]
    return tools, out["messages"][-1].content


# (question, a tool it should call, words the answer must contain)
AGENT_CASES = [
    ("Who knows Python best?", "find_experts", ["Alice"]),
    ("What does Eve work on?", "person_profile", ["Search Revamp"]),
    ("Which skills on active projects have only one person covering them?", "staffing_gaps", []),
    ("Who could help with a recommendation engine?", "semantic_search", ["Eve"]),
]


@pytest.mark.llm
@pytest.mark.parametrize("question,tool,words", AGENT_CASES, ids=[c[1] for c in AGENT_CASES])
def test_agent_picks_the_right_tool(driver, llm_ready, question, tool, words):
    tools, answer = run_agent(question)
    assert tool in tools, f"called {tools}"
    for w in words:
        assert w in answer, answer
