from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Optional, Iterable, Dict, Any, List

from sqlalchemy import text
from repoops.db.session import engine
from repoops.indexing.chunker import iter_code_files, chunk_file
from repoops.indexing.embeddings import embed_text, embed_many
from repoops.common.config import EMBEDDINGS_DIM

def _to_vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

def ensure_pgvector():
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS repo_index (
                  repo_id TEXT PRIMARY KEY,
                  commit_sha TEXT NOT NULL,
                  embedding_model TEXT NOT NULL,
                  embedding_dim INT NOT NULL,
                  updated_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS code_chunks (
                  id SERIAL PRIMARY KEY,
                  repo_id TEXT NOT NULL,
                  commit_sha TEXT NOT NULL,
                  path TEXT NOT NULL,
                  start_line INT NOT NULL,
                  end_line INT NOT NULL,
                  kind TEXT NOT NULL,
                  symbol TEXT NOT NULL,
                  lang TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  content TEXT NOT NULL,
                  created_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """
            )
        )

        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS code_embeddings (
                  chunk_id INT PRIMARY KEY REFERENCES code_chunks(id) ON DELETE CASCADE,
                  embedding vector({EMBEDDINGS_DIM}) NOT NULL,
                  embedding_model TEXT NOT NULL,
                  embedding_dim INT NOT NULL,
                  created_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """
            )
        )

        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_repo_commit_path ON code_chunks(repo_id, commit_sha, path);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_repo_path ON code_chunks(repo_id, path);"))
        conn.execute(
            text(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_chunk_identity
                ON code_chunks(repo_id, commit_sha, path, start_line, end_line, content_hash);
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_embeddings_vec
                ON code_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                """
            )
        )

def get_repo_index(repo_id: str) -> Optional[Dict[str, Any]]:
    ensure_pgvector()
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT repo_id, commit_sha, embedding_model, embedding_dim, updated_at FROM repo_index WHERE repo_id = :rid"),
            {"rid": repo_id},
        ).fetchone()
        if not row:
            return None
        return {
            "repo_id": row[0],
            "commit_sha": row[1],
            "embedding_model": row[2],
            "embedding_dim": int(row[3]),
            "updated_at": str(row[4]),
        }

def delete_repo_semantic(repo_id: str) -> None:
    ensure_pgvector()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM code_embeddings
                WHERE chunk_id IN (SELECT id FROM code_chunks WHERE repo_id = :rid)
                """
            ),
            {"rid": repo_id},
        )
        conn.execute(text("DELETE FROM code_chunks WHERE repo_id = :rid"), {"rid": repo_id})
        conn.execute(text("DELETE FROM repo_index WHERE repo_id = :rid"), {"rid": repo_id})

def _git_head_commit(repo_root: str) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    out = (proc.stdout or "").strip()
    return out if out else "unknown"

def _build_meta_header(path: str, lang: str, kind: str, symbol: str, text: str) -> str:
    """
    Inject cheap, deterministic metadata into the embedded text
    to improve semantic search for React SPA patterns without DB migrations.
    """
    low = (text or "").lower()

    hook = bool(symbol and symbol.startswith("use")) or ("useeffect" in low or "usememo" in low or "usecallback" in low)
    context = ("createcontext(" in low) or ("context.provider" in low) or ("context.consumer" in low)
    reduxish = ("@reduxjs/toolkit" in low) or ("createSlice" in text) or ("configureStore" in text) or ("useSelector" in text) or ("useDispatch" in text)
    zustandish = ("zustand" in low) or ("create(" in low and "set" in low and "get" in low and "zustand" in low)

    # Very rough JSX signal
    jsxish = ("<" in text and (".tsx" in path or "jsx" in lang)) or ("return (" in low and "<" in text)

    flags: list[str] = []
    if hook:
        flags.append("hook=true")
    if context:
        flags.append("context=true")
    if reduxish:
        flags.append("redux=true")
    if zustandish:
        flags.append("zustand=true")
    if jsxish:
        flags.append("jsx=true")

    flag_str = " ".join(flags)
    return f"/* META path={path} lang={lang} kind={kind} symbol={symbol} {flag_str} */\n"

def upsert_repo_semantic(
    repo_id: str,
    repo_root: str,
    *,
    max_files: int = 600,
    commit_sha: Optional[str] = None,
    embed_batch_size: int = 64,
    embed_max_chars: int = 12000,
    embed_min_batch: int = 8,
) -> Dict[str, Any]:
    ensure_pgvector()
    head = commit_sha or _git_head_commit(repo_root)

    # Hard rule: keep ONLY current commit per repo_id
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM code_embeddings
                WHERE chunk_id IN (SELECT id FROM code_chunks WHERE repo_id = :rid)
                """
            ),
            {"rid": repo_id},
        )
        conn.execute(text("DELETE FROM code_chunks WHERE repo_id = :rid"), {"rid": repo_id})

    # Determine embedding model/dim once
    sample = embed_text("ping")
    embedding_model = sample.model_id
    embedding_dim = sample.dim

    indexed_files = 0
    indexed_chunks = 0
    embed_failures = 0
    truncated = 0

    def _cap_text(text: str) -> str:
        nonlocal truncated
        text = text or ""
        if len(text) <= embed_max_chars:
            return text
        truncated += 1
        return text[:embed_max_chars] + "\n...<truncated_for_embedding>..."

    def _embed_many_robust(texts: List[str]) -> List[Any]:
        nonlocal embed_failures
        try:
            return embed_many(texts)
        except Exception:
            embed_failures += 1

        if len(texts) <= embed_min_batch:
            out = []
            for text_item in texts:
                try:
                    out.append(embed_text(text_item))
                except Exception:
                    embed_failures += 1
                    out.append(None)
            return out

        mid = len(texts) // 2
        left = _embed_many_robust(texts[:mid])
        right = _embed_many_robust(texts[mid:])
        return left + right

    with engine.begin() as conn:
        pending: list[tuple[int, str]] = []

        def flush_batch():
            nonlocal indexed_chunks, pending
            if not pending:
                return

            ids = [chunk_id for (chunk_id, chunk_text) in pending]
            texts = [_cap_text(chunk_text) for (chunk_id, chunk_text) in pending]

            results = _embed_many_robust(texts)
            if len(results) != len(ids):
                raise RuntimeError(f"embed_many returned {len(results)} results for {len(ids)} texts")

            rows = []
            for cid, emb in zip(ids, results):
                if emb is None:
                    continue
                rows.append(
                    dict(
                        cid=cid,
                        emb=_to_vector_literal(emb.vector),
                        model=emb.model_id,
                        dim=emb.dim,
                    )
                )

            if rows:
                conn.execute(
                    text(
                        """
                        INSERT INTO code_embeddings (chunk_id, embedding, embedding_model, embedding_dim)
                        VALUES (:cid, CAST(:emb AS vector), :model, :dim)
                        """
                    ),
                    rows,
                )
                indexed_chunks += len(rows)

            pending = []

        for rel in iter_code_files(repo_root):
            chunks = chunk_file(repo_root, rel)

            for chunk in chunks:
                res = conn.execute(
                    text(
                        """
                        INSERT INTO code_chunks (repo_id, commit_sha, path, start_line, end_line, kind, symbol, lang, content_hash, content)
                        VALUES (:repo_id, :commit_sha, :path, :start_line, :end_line, :kind, :symbol, :lang, :content_hash, :content)
                        ON CONFLICT DO NOTHING
                        RETURNING id
                        """
                    ),
                    dict(
                        repo_id=repo_id,
                        commit_sha=head,
                        path=chunk.path,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        kind=chunk.kind,
                        symbol=chunk.symbol,
                        lang=chunk.lang,
                        content_hash=chunk.content_hash,
                        content=chunk.text,
                    ),
                )
                row = res.fetchone()
                if row:
                    header = _build_meta_header(chunk.path, chunk.lang, chunk.kind, chunk.symbol, chunk.text)
                    pending.append((int(row[0]), header + chunk.text))
                    if len(pending) >= embed_batch_size:
                        flush_batch()

            indexed_files += 1
            if indexed_files >= max_files:
                break

        flush_batch()

        conn.execute(
            text(
                """
                INSERT INTO repo_index (repo_id, commit_sha, embedding_model, embedding_dim, updated_at)
                VALUES (:rid, :sha, :model, :dim, NOW())
                ON CONFLICT (repo_id) DO UPDATE
                SET commit_sha = EXCLUDED.commit_sha,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_dim = EXCLUDED.embedding_dim,
                    updated_at = NOW()
                """
            ),
            dict(rid=repo_id, sha=head, model=embedding_model, dim=embedding_dim),
        )

        conn.execute(text("ANALYZE code_chunks;"))
        conn.execute(text("ANALYZE code_embeddings;"))

    return {
        "repo_id": repo_id,
        "commit_sha": head,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "indexed_files": indexed_files,
        "indexed_chunks": indexed_chunks,
        "mode": "full",
        "embed_batch_size": embed_batch_size,
        "embed_max_chars": embed_max_chars,
        "embed_failures": embed_failures,
        "truncated": truncated,
    }

def semantic_search(repo_id: str, query: str, *, k: int = 40) -> List[Dict[str, Any]]:
    ensure_pgvector()
    emb = embed_text(query)
    qemb = _to_vector_literal(emb.vector)

    with engine.begin() as conn:
        res = conn.execute(
            text(
                """
                SELECT
                    c.path, c.start_line, c.end_line, c.symbol, c.lang,
                    (1 - (e.embedding <=> CAST(:qemb AS vector))) as score
                FROM code_embeddings e
                JOIN code_chunks c ON c.id = e.chunk_id
                WHERE c.repo_id = :repo_id
                ORDER BY e.embedding <=> CAST(:qemb AS vector)
                LIMIT :k
                """
            ),
            dict(repo_id=repo_id, qemb=qemb, k=int(k)),
        ).fetchall()

    return [
        dict(
            path=row[0],
            start_line=int(row[1]),
            end_line=int(row[2]),
            symbol=row[3],
            lang=row[4],
            score=float(row[5]),
        )
        for row in res
    ]
