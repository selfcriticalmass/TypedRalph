# CodeAct Agent System

Terminal-native CodeAct agent with two execution modes:

- `free`: natural-language request -> Python generation -> subprocess execution loop
- `schema`: typed function search/execute loop over a Pydantic-backed registry

## Features

- Textual TUI with mode selection, backend selection, trace log, schema browser, and CFG/logits status
- Strict subprocess isolation for generated code and registered function execution
- Pydantic function registry and argument validation
- Hybrid function retrieval using lexical/type signals plus BGE-M3 docstring embeddings
- Parquet embedding cache with schema-hash invalidation
- First-class local Gemma 4 31B workflow inside the uv-managed `.venv`
- First-class OpenRouter workflow using Gemma 4 31B as the default remote model
- Explicit backend capability reporting for CFG and full-logits availability

## Project Layout

```text
codeact-agent/
├── agent.py
├── config.py
├── embed.py
├── executor.py
├── inference.py
├── pyproject.toml
├── schema.py
├── search.py
├── ui/
│   ├── __init__.py
│   └── app.py
└── README.md
```

## Install

```bash
uv sync
```

## Run

```bash
uv run codeact-agent
```

If `config.json` exists in the project root, the TUI loads it automatically.

For one-off commands without installing globally, `uv run python -m ui.app` works too.

## Example Config

```json
{
  "mode": "free",
  "inference": {
    "backend": "local_gemma",
    "model_name": "google/gemma-4-31B-it",
    "local_model_id": "google/gemma-4-31B-it",
    "local_cache_root": ".venv/huggingface",
    "download_if_missing": true,
    "cfg_enabled": true,
    "context_length": 8192,
    "temperature": 0.2
  },
  "retrieval": {
    "embedding_model": "BAAI/bge-m3",
    "cache_path": "cache/vectors.parquet",
    "top_k": 5
  },
  "execution": {
    "workspace_root": ".",
    "artifact_dir": "artifacts",
    "free_mode_timeout": 30,
    "schema_mode_timeout": 15,
    "max_iterations": 8
  },
  "schema": {
    "registry_path": "./registry.py"
  }
}
```

## Function Registry

Schema mode expects a Python module or file that exposes `FUNCTION_REGISTRY`. Entries may be:

- `FunctionSchema` instances
- top-level Python callables, which are auto-converted into `FunctionSchema`

Functions must be importable at module scope and callable with keyword arguments because execution happens in a subprocess.

Example `registry.py`:

```python
from schema import FunctionSchema


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


FUNCTION_REGISTRY = [
    FunctionSchema.from_callable(add, tags=["math", "arithmetic"]),
]
```

## Secrets

API keys resolve in this order:

1. environment variables
2. `.env` in the workspace root
3. runtime entry in the TUI
4. optional keyring lookup

Relevant variables:

- `OPENROUTER_API_KEY`
- `OPENROUTER_BASE_URL`
- `HF_TOKEN`
- `CODEACT_EMBED_API_KEY`
- `CODEACT_EMBED_BASE_URL`

The TUI never prints API keys and does not pass them on the command line.

## Dependency Management

This project uses `uv` as the source of truth for dependencies.

- Add or update packages with `uv add <package>`
- Sync the environment with `uv sync`
- Run the app with `uv run codeact-agent`

`requirements.txt` is intentionally not used so dependency metadata only lives in one place.

## Backend Notes

- `local_gemma` is the primary local backend.
- On first use, the app checks for Gemma assets inside `.venv` and downloads them if they are missing.
- Downloaded model assets are resolved through Hugging Face with the cache rooted under `.venv/huggingface`.
- The local loader uses `AutoProcessor.from_pretrained(...)` and `AutoModelForCausalLM.from_pretrained(...)` against that uv-local cache.
- Local transformers inference is treated as CFG-capable because it can expose generation scores and supports custom logits processors.
- `openrouter` is the primary API backend and defaults to the same Gemma model id.
- OpenRouter is shown as CFG-disabled unless explicitly overridden because it does not expose the full logits surface the local path can provide.

## Local Gemma Bootstrap

The local workflow is designed around `uv`, Hugging Face, and a project-local `.venv`.

1. `uv sync` creates the environment and installs the runtime.
2. When you run the app with the `local_gemma` backend selected, it checks whether Gemma model assets are already available inside `.venv`.
3. If they are missing and `download_if_missing` is enabled, it downloads the model into `.venv/models/...`.
4. The backend then loads the tokenizer and weights from that local path and uses transformers generation with custom logits processor hooks enabled.

If the Gemma repo requires Hugging Face authentication, set `HF_TOKEN` before running the app.

## Retrieval Notes

- Semantic retrieval uses `BAAI/bge-m3` through `FlagEmbedding.BGEM3FlagModel`.
- Vector generation and Parquet cache management live in `embed.py`.
- Hybrid search that consumes those vectors lives in `search.py`.
- Cache entries are written to Parquet at `cache/vectors.parquet`.
- Dense and sparse vectors are stored as JSON-encoded columns so the cache remains portable and simple to inspect.
- The cache is invalidated when the function registry schema hash changes.

## Current Scope

This implementation keeps the two execution paths separate and enforces subprocess execution for both. The schema loop is prompt-constrained rather than provider-native function calling, so the action surface remains bounded even when the underlying backend does not offer strict tool APIs.
