"""Lesson 5.2 — give nodes text descriptions, embed them locally, store vectors in Neo4j."""
from fastembed import TextEmbedding
from neo4j import GraphDatabase

from config import EMBED_MODEL as MODEL, NEO4J_AUTH as AUTH, NEO4J_URI as URI

DESCRIPTIONS = {
    ("Project", "Search Revamp"): "Rebuilding product search with semantic ranking, query understanding and machine-learning relevance models.",
    ("Project", "Mobile App"): "Cross-platform mobile application with a redesigned user interface, smoother onboarding and offline support.",
    ("Project", "Legacy Migration"): "Moved the old billing monolith off on-prem servers onto cloud infrastructure. Finished last year.",
    ("Person", "Alice"): "Backend engineer who loves distributed systems and Python, and mentors juniors on testing.",
    ("Person", "Bob"): "Product designer focused on accessibility, user research and interaction design.",
    ("Person", "Cara"): "Frontend engineer building React Native components and a shared design system.",
    ("Person", "Dev"): "Engineering manager who runs hiring, quarterly planning and career growth for the team.",
    ("Person", "Eve"): "Data scientist training ranking models and running A/B experiments on search quality.",
    ("Person", "Finn"): "Senior engineer exploring Rust for high-performance, low-latency services.",
    ("Person", "Gia"): "Systems programmer with a background in compilers and Rust developer tooling.",
}

model = TextEmbedding(MODEL)
keys = list(DESCRIPTIONS)
vectors = list(model.embed([DESCRIPTIONS[k] for k in keys]))
rows = [
    {"label": label, "name": name, "text": DESCRIPTIONS[(label, name)], "embedding": vec.tolist()}
    for (label, name), vec in zip(keys, vectors)
]

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    # Clean up the toy example from 5.1.
    driver.execute_query("MATCH (t:Toy) DETACH DELETE t")
    driver.execute_query("DROP INDEX toy_vec IF EXISTS")

    # One extra label (:Embedded) lets a single vector index cover people AND projects.
    driver.execute_query(
        """
        UNWIND $rows AS row
        MATCH (n {name: row.name}) WHERE row.label IN labels(n)
        SET n:Embedded, n.description = row.text
        WITH n, row
        CALL db.create.setNodeVectorProperty(n, 'embedding', row.embedding)
        """,
        rows=rows,
    )
    print(f"Embedded {len(rows)} nodes ({len(rows[0]['embedding'])} dims)")

    driver.execute_query(
        """
        CREATE VECTOR INDEX embedded_vec IF NOT EXISTS
        FOR (n:Embedded) ON (n.embedding)
        OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
        """
    )
    driver.execute_query("CALL db.awaitIndexes(60)")
    print("Vector index 'embedded_vec' is online")
