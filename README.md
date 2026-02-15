# RepoOps Autopilot

RepoOps Autopilot is a local-first backend service that takes a repo URL plus a task description and produces verified code edits using retrieval and LLM orchestration, **focused on React single-page application (SPA) frontends**. It clones a repo into a per-run workspace, builds retrieval context, asks the model for minimal edits, applies guardrails, runs verification, and stores artifacts in Postgres.

**Target scope:** React SPA frontend projects (Vite/Webpack), TypeScript/JavaScript, common frontend patterns (components, hooks, routing, state managers).

**Core components**
1. `apps/api`: FastAPI service for creating runs and fetching results.
2. `apps/worker`: background worker that executes runs.
3. `repoops/indexing`: chunking, embeddings, and pgvector index.
4. `repoops/retrieval`: hybrid search (keyword plus semantic) with optional rerank.
5. `repoops/core`: orchestration graph, guardrails, verification, task checks.
6. `repoops/tools/ts_graph`: TypeScript graph expansion via tsserver/TS language service.

## Architecture (Best-Results Target)
1. **Lexical:** Zoekt (primary) with `rg` fallback.
2. **Semantic:** OpenAI embeddings + pgvector.
3. **Structural:** tsserver graph expansion (defs/refs/import closure).
4. **Fusion:** RRF across lexical + semantic.
5. **Rerank:** optional, only when retrieval is ambiguous.

## How It Works
1. Create a run via `POST /runs`.
2. Worker claims the run, clones the repo into `WORKSPACE_ROOT/<run_id>/repo`.
3. Semantic index is built using pgvector.
4. Retrieval builds context using keyword and semantic search.
5. The LLM proposes edits and guardrails validate them.
6. Edits are applied, verification runs, and artifacts are saved.

## Requirements
1. Docker and Docker Compose.
2. Postgres with pgvector (provided via `docker-compose.yml`).
3. An OpenAI-compatible LLM endpoint for edits.
4. An embeddings provider (OpenAI-compatible or hash fallback).

## Quick Start (Docker Compose)
1. Create a `.env` file with your credentials.
2. Start services:

```bash
docker compose up -d --build db api worker
```

The API will be available at `http://localhost:8000`.
The worker runs in the background to process runs. To follow logs:

```bash
docker compose logs -f api worker
```

### With Zoekt (Lexical Search)
1. Set `ZOEK_URL=http://localhost:6070` in `.env`.
2. Start services with the Zoekt profile:

```bash
docker compose --profile zoekt up --build
```

Zoekt indexes the shared `/workspaces` volume on a loop and serves results on port `6070`.
You can tune indexing via:

1. `ZOEK_INDEX_PATHS` (default `/workspaces`)
2. `ZOEK_INDEX_INTERVAL` (default `60` seconds)
3. `ZOEK_INDEX_ARGS` (extra `zoekt-index` flags)

## Environment Configuration
These are the most important variables (see `repoops/common/config.py` for all options).

1. `DATABASE_URL`
2. `OPENAI_BASE_URL`
3. `OPENAI_API_KEY`
4. `OPENAI_MODEL`
5. `OPENAI_STRONG_BASE_URL` (optional)
6. `OPENAI_STRONG_API_KEY` (optional)
7. `OPENAI_STRONG_MODEL` (optional)
8. `EMBEDDINGS_PROVIDER`
9. `EMBEDDINGS_BASE_URL`
10. `EMBEDDINGS_API_KEY`
11. `EMBEDDINGS_MODEL`
12. `WORKSPACE_ROOT`
13. `LEXICAL_PRIMARY` (default `zoekt`)
14. `ZOEK_URL` (Zoekt webserver URL, enables Zoekt)
15. `RRF_K`
16. `RAG_MAX_FILES`
17. `RAG_MAX_SNIPS_PER_FILE`
18. `RAG_MAX_CONTEXT_CHARS`
19. `TS_GRAPH_ENABLED`
20. `TS_GRAPH_MAX_FILES`
21. `TS_GRAPH_MAX_REFS`
22. `TS_GRAPH_MAX_DEFS`
23. `TS_GRAPH_MAX_IMPORTS`

When running via Docker Compose, ensure `DATABASE_URL` uses host `db`, not `localhost`.
Default driver is `psycopg` (psycopg3). If you prefer psycopg2, set `DATABASE_URL` to use `postgresql+psycopg2://...` and install that driver instead.

## Zoekt (Lexical Search)
Zoekt is optional but recommended for large repos. To enable it:
1. Run a local Zoekt webserver/indexer.
2. Set `ZOEK_URL` (for example `http://localhost:6070`).
3. Keep `LEXICAL_PRIMARY=zoekt`.

If `ZOEK_URL` is not set or Zoekt is unavailable, the system falls back to `rg`.

## API Endpoints
1. `POST /runs`
2. `GET /runs/{run_id}`
3. `GET /runs/{run_id}/artifacts`

## API Usage (Docker)
All examples assume the API is running at `http://localhost:8000`.

### Create a run
```bash
curl -X POST "http://localhost:8000/runs" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/ingaAdejanova/movies-fe.git",
    "ref": "master",
    "task_text": "Add a loading state to the main App component."
  }'
```

### Get run status
```bash
curl "http://localhost:8000/runs/<run_id>"
```

### Fetch artifacts
```bash
curl "http://localhost:8000/runs/<run_id>/artifacts"
```

Artifacts include `notes`, `diff`, `test_log`, `repo_overview`, `rag_trace`, `rag_eval`, and `pr_text`.

## Artifacts
Each run saves artifacts such as:
1. `notes`
2. `diff`
3. `test_log`
4. `repo_overview`
5. `rag_trace`
6. `rag_eval`
7. `pr_text`

## Local Development (No Docker)
1. Install dependencies from `requirements.txt`.
2. Ensure Postgres and pgvector are running locally.
3. Start the API:

```bash
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```

4. Start the worker:

```bash
python -m apps.worker.worker
```

## Notes
1. The worker runs verification commands based on detected scripts in `package.json` or `requirements.txt`.
2. The LLM router supports a base model and an optional stronger model for retries.
3. Guardrails restrict edits to safe paths and limit the scope of changes.

## Evaluation (Local)
We include a lightweight retrieval evaluation harness so you can compare models/settings.

```bash
python -m repoops.eval.run_eval --dataset repoops/eval/data/sample_tasks.jsonl --repo-root .
```

See `repoops/eval/README.md` for dataset format and options.

### Evaluation via Docker (No Local Python Dependencies)
1. Start the database:

```bash
docker compose up -d db
```

2. Run eval using the Docker profile:

```bash
docker compose --profile eval run --rm eval
```

To pass custom options:

```bash
docker compose --profile eval run --rm eval --dataset repoops/eval/data/sample_tasks.jsonl --repo-root /app --use-llm
```

To evaluate a remote repo without manual cloning:

```bash
docker compose --profile eval run --rm eval \
  --repo-url https://github.com/ingaAdejanova/movies-fe.git \
  --ref master \
  --dataset repoops/eval/data/sample_tasks.jsonl \
  --index
```

To evaluate a repo-specific dataset with zoek:
```bash
ZOEK_URL=http://zoekt-web:6070 LEXICAL_PRIMARY=zoekt \
docker compose --profile eval run --rm --build eval \
  --repo-url https://github.com/ingaAdejanova/movies-fe.git \
  --ref master \
  --dataset repoops/eval/data/movies_fe_tasks.jsonl \
  --index
```

If you want to use Zoekt inside Docker:
1. Start Zoekt services (indexer + webserver):

```bash
docker compose --profile zoekt up -d --build zoekt-index zoekt-web
```

2. (Optional) Restrict indexing to a path:
   - `ZOEK_INDEX_PATHS=/workspaces/repo` indexes the bind-mounted repo.
   - `ZOEK_INDEX_PATHS=/workspaces/eval/<repo-slug>` indexes an eval clone.

3. You can override at runtime:
   `docker compose --profile eval run --rm -e ZOEK_URL=http://zoekt-web:6070 -e LEXICAL_PRIMARY=zoekt eval ...`

Note: the Zoekt services are built locally from source for ARM/macOS compatibility, so the first build may take a few minutes.
The Zoekt webserver is started with `-rpc` to expose the JSON API used by the eval harness.

To force a one-off index pass:

```bash
docker compose --profile zoekt run --rm zoekt-index-once
# or restrict it:
docker compose --profile zoekt run --rm zoekt-index-once -index /data/index /workspaces/eval/<repo-slug>
```

### Generate a Repo-Specific Eval Dataset
For more realistic results, generate a dataset from the target repo:

```bash
python -m repoops.eval.generate_dataset --repo-root /path/to/repo --out repoops/eval/data/generated_tasks.jsonl
```

Then run eval against it:

```bash
python -m repoops.eval.run_eval --dataset repoops/eval/data/generated_tasks.jsonl --repo-root /path/to/repo --index
```
