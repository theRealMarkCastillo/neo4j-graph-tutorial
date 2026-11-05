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

The **checkpointer** is LangGraph's short-term memory: every turn on the same `thread_id`
sees the earlier messages. (Lessons 8–10 add *long-term* memory in Neo4j.)

## The tools — [`team_agent.py`](../team_agent.py)

| Tool | Kind | What it does |
|---|---|---|
| `find_experts(skill)` | fixed, parameterized | people with a skill, strongest first, plus their active projects |
| `person_profile(name)` | fixed | role, bio, skills, projects, manager, contacts |
| `staffing_gaps()` | fixed | required skills only 0–1 people have (Lesson 1 query 9) |
| `semantic_search(text)` | vector | Lesson 5's `embedded_vec` index |
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
.venv/bin/python team_agent.py                                   # interactive
.venv/bin/python team_agent.py "Who could back up Eve if she's out for a month?"
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

## Takeaways

- Start with **fixed, parameterized tools**; add free-form Cypher behind a read-only transaction.
- Put the **schema in the tool description** so the model can write queries.
- LangGraph's checkpointer gives **short-term** memory per thread; long-term memory needs a store.
