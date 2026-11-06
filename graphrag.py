"""Lesson 5.5 — GraphRAG: vector search finds entry points, the graph supplies connected context.

Usage:  uv run python graphrag.py "who could help with a recommendation engine?"
If LLM_MODEL (and LLM_BASE_URL / LLM_API_KEY) are set in .env, the LLM writes the answer;
otherwise the script prints the retrieved context and stops.
"""
import sys

from fastembed import TextEmbedding
from neo4j import GraphDatabase

import config
from config import LLM_MODEL, LOCAL_EMBED_MODEL as MODEL, NEO4J_AUTH as AUTH, NEO4J_URI as URI

# Step 1 finds the top-k semantically similar nodes; step 2 walks 1 hop out from each and
# turns every relationship into a plain-English fact the LLM can read.
RETRIEVAL_QUERY = """
CALL db.index.vector.queryNodes('embedded_vec', $k, $vector) YIELD node, score
OPTIONAL MATCH (node)-[r]-(other)
WHERE NOT other:Assignment
WITH node, score,
     collect(DISTINCT
       CASE WHEN startNode(r) = node
            THEN node.name + ' ' + type(r) + ' ' + other.name
            ELSE other.name + ' ' + type(r) + ' ' + node.name END
       + coalesce(' (level ' + toString(r.level) + ')', '')) AS facts
RETURN node.name AS name, labels(node)[0] AS type, node.description AS description,
       score, facts
ORDER BY score DESC
"""


def retrieve(question: str, k: int = 3) -> list[dict]:
    vector = next(TextEmbedding(MODEL).embed([question])).tolist()
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        records, _, _ = driver.execute_query(RETRIEVAL_QUERY, vector=vector, k=k)
    return [r.data() for r in records]


def build_context(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        facts = "\n".join(f"  - {f}" for f in h["facts"])
        blocks.append(f"[{h['type']}] {h['name']} (match {h['score']:.2f})\n"
                      f"  {h['description']}\n{facts}")
    return "\n\n".join(blocks)


SYSTEM = """You answer questions about a team, using only the knowledge-graph facts inside <context>.

- Recommend the people, projects or skills from the context that fit best, and say which facts support each one.
- Partial matches are useful: "Eve trains ranking models" is relevant to a recommendation engine.
- Only if nothing in the context is relevant, say that the graph doesn't cover it. Never invent people or facts."""


def answer(question: str, context: str) -> str | None:
    """Plain OpenAI SDK call: no framework, just messages in, text out."""
    if not config.llm_configured():
        print("(No LLM configured: set LLM_MODEL, LLM_BASE_URL and LLM_API_KEY in .env "
              "to get a generated answer.)\n")
        return None
    import openai

    try:
        response = config.llm_client().chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}"},
            ],
        )
    except openai.APIError as e:
        print(f"(LLM call failed: {type(e).__name__}: {e})\n")
        return None
    return response.choices[0].message.content


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Who could help us build a recommendation engine?"
    print(f"Q: {question}\n")
    context = build_context(retrieve(question))
    print("=== Retrieved graph context ===")
    print(context, "\n")
    text = answer(question, context)
    if text:
        print("=== LLM answer ===")
        print(text)
