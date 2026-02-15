# Evaluation (Local)

This folder provides a lightweight retrieval evaluation harness so you can compare models or retrieval settings with fixed test data.

## Dataset Format (JSONL)
Each line is a JSON object with:
1. `id` (string)
2. `task` (string)
3. `expected_paths` (array of repo-relative paths)
4. `tags` (optional array)

Example:
```json
{"id":"t1","task":"Improve the retrieval fusion logic.","expected_paths":["repoops/retrieval/hybrid.py"]}
```

## Run Evaluation
From the repo root:

```bash
python -m repoops.eval.run_eval --dataset repoops/eval/data/sample_tasks.jsonl --repo-root .
```

To rebuild the semantic index before running:

```bash
python -m repoops.eval.run_eval --dataset repoops/eval/data/sample_tasks.jsonl --repo-root . --index
```

To enable LLM rerank during eval (uses your configured LLM):

```bash
python -m repoops.eval.run_eval --dataset repoops/eval/data/sample_tasks.jsonl --repo-root . --use-llm
```

## Clone a Repo Automatically
You can also evaluate a remote repo without manually cloning it:

```bash
python -m repoops.eval.run_eval --dataset repoops/eval/data/sample_tasks.jsonl --repo-url https://github.com/ingaAdejanova/movies-fe.git --ref main --index
```

For Docker, set `--clone-root /workspaces/eval` (default).

## What You Get
- Per-task retrieval metrics: recall@k, precision@k, MRR, nDCG
- Aggregate averages
- Model/config metadata (embeddings model, LLM model, lexical engine)

Use different env values (e.g. `EMBEDDINGS_MODEL`, `RAG_RERANK_MODEL`, `LEXICAL_PRIMARY`) to compare results across runs.

## Generate a Repo-Specific Dataset
You can generate a task set tailored to a specific repo:

```bash
python -m repoops.eval.generate_dataset --repo-root /path/to/repo --out repoops/eval/data/generated_tasks.jsonl
```

For Docker/eval clones:

```bash
python -m repoops.eval.generate_dataset --repo-root /workspaces/eval/<repo-slug> --out repoops/eval/data/generated_tasks.jsonl
```
