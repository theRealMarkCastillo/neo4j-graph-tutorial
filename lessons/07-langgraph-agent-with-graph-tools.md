# Lesson 7 — A LangGraph agent with graph tools

Lesson 6 described giving an agent tools over the graph. Here we build it with
[LangGraph](https://langchain-ai.github.io/langgraph/) and any **OpenAI-compatible** chat endpoint.

## Setup

In `.env` (see [.env.example](../.env.example)):

```bash
LLM_BASE_URL=https://api.deepinfra.com/v1   # or OpenAI, OpenRouter, Ollama, LM Studio, vLLM...
LLM_MODEL=google/gemma-4-31B-it              # must support tool calling
LLM_API_KEY=...
```

`config.chat_model()` builds a LangChain `ChatOpenAI` pointed at that endpoint, so swapping
providers is an `.env` change, not a code change. Recorded outputs used DeepInfra with
`google/gemma-4-31B-it`.

## LangGraph in one picture

A LangGraph app is a **state machine**: a typed state, **nodes** (functions that return state
updates) and **edges** (which node runs next, possibly chosen by a function).

```
START ──► agent ──(tool calls?)──► tools ──┐
            ▲                              │
            └──────────────────────────────┘
          (no tool calls) ──► END
```

```python
g = StateGraph(MessagesState)                    # state = the message list
g.add_node("agent", agent)                       # LLM with tools bound
g.add_node("tools", ToolNode(TOOLS))             # runs whatever tools the LLM asked for
g.add_edge(START, "agent")
g.add_conditional_edges("agent", tools_condition)  # -> "tools" if tool calls, else END
g.add_edge("tools", "agent")
app = g.compile(checkpointer=InMemorySaver())    # remembers messages per thread_id
```

We wire the graph by hand to show the parts. LangChain 1.0's `create_agent(model, tools,
checkpointer=...)` builds this same loop in one call, and is what you'd usually reach for.

The **checkpointer** is LangGraph's short-term memory: every turn on the same `thread_id`
sees the earlier messages. (Lessons 8–10 add *long-term* memory in Neo4j.)

## The tools — [`team_agent.py`](../team_agent.py)

| Tool | Kind | What it does |
|---|---|---|
| `find_experts(skill)` | fixed, parameterized | people with a skill, strongest first, plus their active projects |
| `person_profile(name)` | fixed | role, bio, skills, projects, manager, contacts |
| `staffing_gaps()` | fixed | required skills only 0–1 people have (Lesson 1 query 9) |
| `semantic_search(text)` | vector | Lesson 5's `embedded_vec` index (*semantic* = by meaning, [sense 2](../docs/glossary.md#semantic)) |
| `run_cypher(query)` | free-form | read-only Cypher, schema in the docstring, 25-row cap |

Every tool runs inside `session.execute_read(...)`, so Neo4j itself rejects writes:

```python
>>> run_cypher.invoke({"query": "CREATE (:Person {name:'Hacker'})"})
Query failed: {neo4j_code: Neo.ClientError.Statement.AccessMode}
{message: Writing in read access mode not allowed. Attempted write to neo4j} ...
```

The tool's **docstring is the tool description** the LLM sees, which is why `run_cypher`
carries the schema.

## Run it

```bash
uv run python team_agent.py                                   # interactive
uv run python team_agent.py "Who could back up Eve if she's out for a month?"
```

## Output ([full log](outputs/07-team-agent.txt))

```
You: Who could back up Eve if she's out for a month?
  → person_profile({"name": "Eve"})
  ← person_profile: {"name": "Eve", "role": "Data Scientist", ... "skills": ["Python (le…
  → find_experts({"skill": "ML"})
  → find_experts({"skill": "Python"})
  ← find_experts: [{"person": "Eve", "level": 5, "active_projects": ["Search Revamp"]}]
  ← find_experts: [{"person": "Alice", "level": 5, ...}, {"person": "Eve", "level": 4, ...}, {"person": "Gia", …
  → run_cypher({"query": "MATCH (p:Person)-[:HAS_SKILL]->(s:Skill) WHERE s.name IN ['ML', 'Machine Learning', 'Data Science'] RETURN p.name, s.name"})
  ← run_cypher: [{"p.name": "Eve", "s.name": "ML"}]

Agent: ... Alice is the strongest candidate for backup. She is also working on the Search
Revamp project and possesses a higher proficiency in Python (level 5) than Eve. Gia also has
Python skills (level 3) ... However, Eve appears to be the only person with the ML skill ...
making her a single point of failure for the machine learning aspects of the Search Revamp project.
```

```
You: Which active projects are at risk because only one person has a required skill, and who could we train?
  → staffing_gaps({})
  ← staffing_gaps: [{"project": "Search Revamp", "skill": "ML", "who": ["Eve"]}, {"project": "Mobile App", "skill": "UX", "who": ["Bob"]}]
  → find_experts({"skill": "ML"})
  → find_experts({"skill": "UX"})
  ...
Agent: ... For ML (Search Revamp): Alice ... proficient in Python, which is commonly used for ML.
For UX (Mobile App): Cara ... already building React Native components and a shared design system.
```

```
You: How many people does Dev manage, and what are their roles?
  → run_cypher({"query": "MATCH (m:Person {name: 'Dev'})-[:MANAGES]->(e:Person)
                OPTIONAL MATCH (e)-[:HAS_ROLE]->(r:Role) RETURN e.name AS employee, r.name AS role"})
  ← run_cypher: [{"employee": "Finn", "role": "Senior Engineer"}, {"employee": "Cara", "role": "Engineer"}, {"employee": "Alice", "role": "Engineer"}]

Agent: Dev manages 3 people: Finn (Senior Engineer), Cara (Engineer), Alice (Engineer)
```

What to notice:
- The model **calls tools in parallel** (two `find_experts` at once) — `ToolNode` runs them together.
- It **chains**: profile → experts → a custom Cypher check.
- For a question no fixed tool covers, it **wrote correct Cypher**, including the `:Role` nodes
  from Lesson 3, using only the schema in the docstring.

## Concepts and design choices

*Where this fits:* Not the memory pipeline itself: an agent that *uses* a graph, for contrast with the companion's workflow. The whole pipeline, step by step, is in [design decisions → the pipeline as concepts](../docs/design-decisions.md#the-pipeline-as-concepts).

**Terms with a specific meaning here** (every sense is in the [glossary](../docs/glossary.md#words-with-several-meanings)):

| Term | Means here | Not to be confused with |
|---|---|---|
| [agent](../docs/glossary.md#agent) | sense 1: an LLM choosing tools in a loop | the `agent` property naming a persona (Lesson 16) |
| [graph](../docs/glossary.md#graph) | sense 2: the LangGraph state graph (control flow) | the Neo4j data graph |
| [semantic](../docs/glossary.md#semantic) | sense 2: the `semantic_search` tool is vector search | semantic memory |
| checkpointer | short-term memory: one conversation's messages per `thread_id` | long-term memory (Neo4j, from Lesson 8) |

**Design choices** (with alternatives in [design-decisions.md](../docs/design-decisions.md)):

| We chose | Because | It costs | Instead, you could | |
|---|---|---|---|---|
| A ReAct agent with fixed tools plus a read-only Cypher tool | the model picks the right lookup; damage is limited by permissions | the model can pick the wrong tool (a trajectory eval checks) | a fixed workflow (the companion, D19); text-to-Cypher only | [D19](../docs/design-decisions.md#d19) |
| Retrieval on demand (the model calls a tool) | no cost when the question doesn't need the graph | the model has to know to look | retrieve every turn in code (the companion, D10) | [D10](../docs/design-decisions.md#d10) |
| In-memory checkpointer | no setup | conversations vanish on restart | SQLite / Postgres checkpointers (`langgraph-checkpoint-postgres`); LangChain 1.0's `create_agent` builds the same loop in one call | [D15](../docs/design-decisions.md#d15) |

## Test it — [`tests/test_07_agent.py`](../tests/test_07_agent.py)

Test an agent in two layers:

1. **Tools are plain functions over the graph.** Call them directly, with no LLM:
   - `find_experts` is case-insensitive and ranked;
   - empty results explain themselves ("Nobody has the skill…"), so the model can recover;
   - `run_cypher` refuses writes, reports syntax errors and caps rows at 25.

   The cap streams only 25 rows (`result.fetch(25)`) under a 10-second timeout. The first
   version fetched everything and sliced afterwards, so a runaway query could exhaust memory
   before the cap applied.
2. **The agent's choices need the LLM** (`pytest -m llm`). Grade the **trajectory**: which tool
   it called for each question, plus a few words the answer must contain.

   ```python
   AGENT_CASES = [("Who knows Python best?", "find_experts", ["Alice"]),
                  ("What does Eve work on?", "person_profile", ["Search Revamp"]), ...]
   ```

## Takeaways

- Start with **fixed, parameterized tools**; add free-form Cypher behind a read-only transaction.
- Put the **schema in the tool description** so the model can write queries.
- LangGraph's checkpointer gives **short-term** memory per thread; long-term memory needs a store.

## Next

This agent forgets everything when the process ends. [Lesson 8](08-memory-design-for-companions.md) designs long-term memory for a companion, in the same database.

<!-- nav -->
---

[← Lesson 6: Graphs in AI and AI agents (concepts)](06-graphs-in-ai-and-agents.md) · [All lessons](../README.md#lessons) · [Lesson 8: Designing memory for an AI assistant / companion →](08-memory-design-for-companions.md)
