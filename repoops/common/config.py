import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)

DATABASE_URL = env("DATABASE_URL", "postgresql+psycopg://repoops:repoops@localhost:5432/repoops")

OPENAI_BASE_URL = env("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = env("OPENAI_API_KEY", "")
OPENAI_MODEL = env("OPENAI_MODEL", "gpt-4.1-mini")

# Optional “escalation model” (GPT-5 etc.)
OPENAI_STRONG_BASE_URL = env("OPENAI_STRONG_BASE_URL", OPENAI_BASE_URL) or OPENAI_BASE_URL
OPENAI_STRONG_API_KEY = env("OPENAI_STRONG_API_KEY", "")
OPENAI_STRONG_MODEL = env("OPENAI_STRONG_MODEL", "gpt-5")


# --- RAG / Embeddings ---
EMBEDDINGS_PROVIDER = env("EMBEDDINGS_PROVIDER", "openai_compatible")  # openai_compatible|hash
EMBEDDINGS_BASE_URL = env("EMBEDDINGS_BASE_URL", OPENAI_BASE_URL)
EMBEDDINGS_API_KEY = env("EMBEDDINGS_API_KEY", OPENAI_API_KEY) or ""
EMBEDDINGS_MODEL = env("EMBEDDINGS_MODEL", "text-embedding-3-small")
EMBEDDINGS_TIMEOUT_S = int(env("EMBEDDINGS_TIMEOUT_S", "60") or "60")
EMBEDDINGS_DIM = int(env("EMBEDDINGS_DIM", "1536") or "1536")

# Retrieval tuning
RAG_CANDIDATES_SEMANTIC = int(env("RAG_CANDIDATES_SEMANTIC", "80") or "80")
RAG_CANDIDATES_KEYWORD = int(env("RAG_CANDIDATES_KEYWORD", "200") or "200")
RAG_TOPK_CONTEXT = int(env("RAG_TOPK_CONTEXT", "12") or "12")

# Rerank
RAG_RERANK = (env("RAG_RERANK", "1") or "1") not in ("0", "false", "False")
RAG_RERANK_MODEL = env("RAG_RERANK_MODEL", "gpt-4.1-mini")  # cheap + stable
RAG_RERANK_MAX_DOCS = int(env("RAG_RERANK_MAX_DOCS", "50") or "50")
RAG_RERANK_TIMEOUT_S = int(env("RAG_RERANK_TIMEOUT_S", "45") or "45")
RAG_RERANK_ON_AMBIGUITY = (env("RAG_RERANK_ON_AMBIGUITY", "1") or "1") not in ("0", "false", "False")

# Optional filtering
RAG_FILTER_LANGS = env("RAG_FILTER_LANGS", "")  # e.g. "py,ts,tsx"

# Lexical search
LEXICAL_PRIMARY = (env("LEXICAL_PRIMARY", "zoekt") or "zoekt").lower()
ZOEK_URL = env("ZOEK_URL", "") or ""
ZOEK_TIMEOUT_S = int(env("ZOEK_TIMEOUT_S", "6") or "6")
ZOEK_K = int(env("ZOEK_K", "120") or "120")

# If running inside Docker and ZOEK_URL points to localhost, redirect to service name.
if ZOEK_URL.startswith("http://localhost") and Path("/.dockerenv").exists():
    ZOEK_URL = ZOEK_URL.replace("localhost", "zoekt-web", 1)

# Fusion + context assembly
RRF_K = int(env("RRF_K", "60") or "60")
RAG_MAX_FILES = int(env("RAG_MAX_FILES", "6") or "6")
RAG_MAX_SNIPS_PER_FILE = int(env("RAG_MAX_SNIPS_PER_FILE", "3") or "3")
RAG_MAX_CONTEXT_CHARS = int(env("RAG_MAX_CONTEXT_CHARS", "22000") or "22000")

# TypeScript graph expansion (tsserver/TS language service)
TS_GRAPH_ENABLED = (env("TS_GRAPH_ENABLED", "1") or "1") not in ("0", "false", "False")
TS_GRAPH_MAX_FILES = int(env("TS_GRAPH_MAX_FILES", "4000") or "4000")
TS_GRAPH_MAX_REFS = int(env("TS_GRAPH_MAX_REFS", "120") or "120")
TS_GRAPH_MAX_DEFS = int(env("TS_GRAPH_MAX_DEFS", "80") or "80")
TS_GRAPH_MAX_IMPORTS = int(env("TS_GRAPH_MAX_IMPORTS", "80") or "80")


WORKSPACE_ROOT = env("WORKSPACE_ROOT", "/workspaces") or "/workspaces"

EXEC_ALLOWLIST = [
    item.strip()
    for item in (env("EXEC_ALLOWLIST", "pnpm,npm,yarn,pytest,python,make,git") or "").split(",")
    if item.strip()
]
EXEC_ALLOWLIST_SET = set(EXEC_ALLOWLIST)
