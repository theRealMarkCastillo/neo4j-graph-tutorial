"""Shared settings for the lesson scripts, loaded from .env (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"])

# Local embedding model (no API key needed): Lessons 5-7, and the companion's default.
LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LOCAL_EMBED_DIMS = 384
# Lesson 18: a cross-encoder that judges whether a recalled memory is about the message at all
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

# One OpenAI-compatible chat endpoint for every lesson that calls an LLM (5, 7-12):
# OpenAI, DeepInfra, OpenRouter, Ollama, LM Studio, vLLM...
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or None
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip() or "not-needed"  # local servers ignore it
LLM_EXTRACTION_MODEL = os.getenv("LLM_EXTRACTION_MODEL", "").strip() or LLM_MODEL
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Lesson 19: the companion's embeddings (Lessons 8-18). Leave EMBED_BASE_URL empty for the local
# model above; set it to an OpenAI-compatible /embeddings endpoint to use a remote model. Each
# model needs measured thresholds: see companion/embeddings.py.
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "").strip() or None
EMBED_MODEL = (os.getenv("EMBED_MODEL", "").strip() if EMBED_BASE_URL else "") or LOCAL_EMBED_MODEL
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "").strip() or LLM_API_KEY  # same provider: same key

COMPANION_USER = os.getenv("COMPANION_USER", "Alex")
COMPANION_NAME = os.getenv("COMPANION_NAME", "Sage")


def llm_configured() -> bool:
    return bool(LLM_MODEL)


def llm_client():
    """The plain OpenAI SDK client for the endpoint in .env (Lesson 5).

    `chat_model()` below wraps this same SDK for LangChain/LangGraph (Lessons 7-12)."""
    from openai import OpenAI

    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=120, max_retries=2)


def chat_model(model: str | None = None, temperature: float | None = None):
    """A LangChain chat model (built on the OpenAI SDK) pointed at the endpoint in .env."""
    from langchain_openai import ChatOpenAI

    if not (model or LLM_MODEL):
        raise SystemExit("Set LLM_MODEL (and LLM_BASE_URL / LLM_API_KEY) in .env first.")
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=model or LLM_MODEL,
        temperature=LLM_TEMPERATURE if temperature is None else temperature,
        timeout=120,
        max_retries=2,
    )


# Optional MLflow tracing (observability.py): on only when MLFLOW_TRACKING_URI is set.
import observability  # noqa: E402

observability.setup()
