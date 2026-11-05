"""Graph + vector memory for an AI assistant / companion (Lessons 8-10).

    companion/schema.py   constraints, indexes, vector indexes
    companion/models.py   the memory types the extractor produces
    companion/store.py    MemoryStore: write (resolve, dedupe, supersede) and recall (hybrid)
    companion/extract.py  LLM calls: memory extraction and episode summaries
    companion/graph.py    LangGraph: recall -> respond -> remember
    companion/__main__.py terminal chat:  .venv/bin/python -m companion
    companion/demo.py     scripted two-session demo used to record the lesson outputs
"""
