"""Lesson 5.5 — GraphRAG: vector search finds entry points, the graph supplies connected context.

Usage:  .venv/bin/python graphrag.py "who could help with a recommendation engine?"
If ANTHROPIC_API_KEY is set in .env, Claude writes the answer; otherwise the script
prints the retrieved context and the prompt it would send.
"""
import sys

from fastembed import TextEmbedding
from neo4j import GraphDatabase

from config import ANTHROPIC_MODEL, EMBED_MODEL as MODEL, NEO4J_AUTH as AUTH, NEO4J_URI as URI

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


def answer_with_claude(question: str, context: str) -> str | None:
    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.beta.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=16000,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # re-runs on another model if this one declines
            system="Answer using only the team knowledge graph context provided. "
                   "Name the people, projects and skills that support your answer.",
            messages=[{"role": "user",
                       "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}"}],
        )
    except TypeError:  # the SDK raises this when no API key / auth token / profile is set
        print("(No Anthropic credentials found: add ANTHROPIC_API_KEY to .env to get a generated answer.)\n")
        return None
    except anthropic.APIError as e:
        print(f"(Claude call failed: {type(e).__name__}: {e})\n")
        return None
    if response.stop_reason == "refusal":
        return "(Claude declined to answer.)"
    return "".join(b.text for b in response.content if b.type == "text")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Who could help us build a recommendation engine?"
    print(f"Q: {question}\n")
    context = build_context(retrieve(question))
    print("=== Retrieved graph context ===")
    print(context, "\n")
    answer = answer_with_claude(question, context)
    if answer:
        print("=== Claude's answer ===")
        print(answer)
