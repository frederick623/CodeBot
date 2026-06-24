# CodeBot

CodeBot is a **Rust-based AI coding assistant daemon** that runs locally alongside your editor. It indexes your workspace, retrieves semantically relevant code, and drives a local (or remote) LLM to answer questions, generate code, and apply patches — all while keeping repository content strictly sandboxed from instructions.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Key Technologies](#key-technologies)
- [Source Layout](#source-layout)
- [API Endpoints](#api-endpoints)
- [Core Workflows](#core-workflows)
  - [Chat](#chat)
  - [Plan → Verify → Implement](#plan--verify--implement)
  - [Workspace Indexing](#workspace-indexing)
- [Configuration](#configuration)
- [Security Design](#security-design)
- [Building and Running](#building-and-running)

---

## Architecture Overview

```
Editor Extension  ──HTTP──►  codebot-daemon (127.0.0.1:9099)
                                    │
                    ┌───────────────┼────────────────────┐
                    ▼               ▼                    ▼
               Indexer          Retriever           Orchestrator
            (tree-sitter)   (vector + symbol)   (plan/implement)
                    │               │                    │
                    └───────────────┴────────────────────┘
                                    │
                              Store (SQLite)
                           ┌────────┴────────┐
                       chunks/files      sqlite-vec
                       symbols/edges    (ANN KNN index)
                                    │
                              EmbedClient
                           (fastembed / ONNX)
                                    │
                              LlmClient
                         (OpenAI-compatible API)
```

The daemon listens on `127.0.0.1:9099`. An editor extension (not included in this repo) communicates with it via HTTP JSON calls. The daemon indexes the workspace in the background and keeps the index fresh through a filesystem watcher.

---

## Key Technologies

| Technology | Role |
|---|---|
| **Rust + Tokio** | Async runtime; all I/O and model calls are non-blocking |
| **Axum 0.7** | HTTP server framework; all endpoints are typed JSON handlers |
| **SQLite (rusqlite)** | Persistent storage for files, chunks, symbols, call/import edges, plans, and tasks |
| **sqlite-vec** | Native approximate nearest-neighbour (ANN/KNN) vector search inside SQLite via the `vec0` virtual table |
| **fastembed / ONNX Runtime** | In-process text embeddings (BGE-Small-EN-V1.5) and reranking (BGE-Reranker-Base); no external microservice needed; weights are fetched from Hugging Face on first run |
| **tree-sitter** | AST-aware code parsing and chunking; used for symbol extraction, import/call graph building, syntax error detection, and the name verification gate |
| **Supported languages** | TypeScript/JavaScript, Rust, Python, C, C++, Java, C#, Go |
| **notify / notify-debouncer-full** | Cross-platform filesystem watcher for incremental re-indexing |
| **ignore** | Gitignore-aware workspace walker (skips `.gitignore`-ed and hidden files) |
| **blake3** | Fast content hashing; drives incremental indexing — a file is only re-embedded if its hash changed |
| **similar + diffy** | Unified-diff generation (patch previews) and tolerant diff parsing from model output |
| **reqwest** | Async HTTP client for OpenAI-compatible LLM API calls |
| **serde / serde_json** | JSON serialization for all API types and structured LLM output |
| **tracing / tracing-subscriber** | Structured JSON logging |
| **dashmap** | Concurrent hashmap for workspace registry |
| **parking_lot** | Fast Mutex around synchronous ONNX inference calls |

---

## Source Layout

```
src/
├── main.rs          # Entry point: server setup, route handlers, diff extraction utilities
├── api.rs           # Request/response DTOs for all HTTP endpoints
├── embed_client.rs  # In-process embedding + reranking via fastembed/ONNX
├── indexer.rs       # Workspace indexer: AST chunking, symbol/call/import graph, incremental hashing
├── llm.rs           # OpenAI-compatible LLM client (chat + structured JSON output)
├── orchestrator.rs  # Planning gate and implement loop (plan → generate → verify → retry)
├── prompt.rs        # Prompt assembly with context budgeting and untrusted-data fencing
├── retrieval.rs     # Hybrid retrieval: active file + symbol lookup + vector KNN + reranking
├── storage.rs       # SQLite store: file index, chunk/embedding store, symbol graph, plan/task tables
├── tools.rs         # Tool executor: file read, ripgrep search, patch preview/apply, test runner
├── verifier.rs      # Deterministic verification cascade: parse → names → references
└── watcher.rs       # Filesystem watcher that feeds changed paths to the incremental indexer
```

### Module Responsibilities

#### `main.rs`
Bootstraps all components, wires them into shared `AppState`, registers HTTP routes, and contains the utility functions for extracting unified diffs from LLM responses (fence scrubbing, section splitting, target path resolution).

#### `api.rs`
Pure data layer: all `Request` / `Response` structs for the HTTP API, serialized/deserialized with serde. No logic.

#### `embed_client.rs`
Wraps `fastembed`'s `TextEmbedding` (BGE-Small-EN-V1.5) and `TextRerank` (BGE-Reranker-Base) behind a `Clone`-able async interface. ONNX inference is CPU-bound and runs in `spawn_blocking` to keep it off the async worker threads. Models are loaded once at startup.

#### `indexer.rs`
- **Full index**: walks the workspace respecting `.gitignore`, skips files > 1 MB and files whose paths match secret-detection heuristics.
- **Incremental index**: computes a BLAKE3 hash; if the hash matches the stored value, the file is skipped.
- **AST chunking**: parses each file with tree-sitter, extracts top-level definitions as chunks (one chunk per function/class/etc.); falls back to a sliding line-window for unsupported file types.
- **Symbol graph**: extracts defined names, imported modules, and call sites; inserts them as nodes and typed edges (`import`, `call`) into the store for the symbol-lookup retrieval path.

#### `llm.rs`
Thin wrapper around the OpenAI `/v1/chat/completions` API. Provides two call modes:
- `chat()` — free-text completion (temperature 0.1).
- `complete_json()` — structured output constrained by a JSON schema via `response_format` (temperature 0, deterministic).

#### `orchestrator.rs`
Runs the two-phase code-generation workflow:
1. **Plan** — sends a system-level prompt that constrains the model to emitting only a JSON interface spec (module names, function signatures, parameter types). Every identifier is normalized to `snake_case` and frozen into a `plan_symbols` registry in the store.
2. **Implement** — for each pending task in the plan, calls the LLM to generate a function body, then runs the verifier cascade. If the cascade auto-corrects the output, the correction is stored. If it fails, the task is retried (up to `max_attempts`). Terminal statuses: `verified` or `failed`.

#### `prompt.rs`
Assembles the final user message from retrieved chunks under a character budget. Repository content is always wrapped in `<repo_context>` / `<file>` tags and sanitized (closing tags and fake instruction blocks are neutralized) so it cannot be mistaken for instructions. The system prompt enforces the instruction/data boundary.

#### `retrieval.rs`
Three-stage hybrid retrieval, de-duplicated by chunk key, then reranked:
1. **Active file** — the currently open file (or selection) is always included.
2. **Symbol lookup** — CamelCase/underscore identifiers from the user prompt are matched against the symbol index.
3. **Vector KNN** — the prompt is embedded and the top-20 nearest chunks are fetched from the `vec0` table.
4. **Reranking** — all candidates are scored against the prompt with the cross-encoder reranker; top 8 are returned.

#### `storage.rs`
SQLite schema (WAL mode):
- `files` — path, content hash, size, timestamps.
- `chunks` — file, line range, optional symbol name, content.
- `symbols` — file, name, kind (function/class/etc.), line range.
- `edges` — directed graph edges with a type label (`import`, `call`).
- `chunk_embeddings` — `vec0` virtual table for ANN KNN.
- `plan_symbols` — frozen interface registry per plan.
- `tasks` — implementation work units with status lifecycle.

Above ~50 000 chunks the `vec0` index handles KNN natively; below that, brute-force cosine is used instead.

#### `tools.rs`
Utility executor for the daemon:
- `read_file` — reads a workspace file (path-traversal checked).
- `search` — ripgrep-backed pattern search.
- `make_patch` — generates a unified-diff preview from a proposed full-file replacement using `similar`.
- `apply_full` — writes the new content to disk (only after user approval is signalled by the extension).
- `run_tests` — runs an arbitrary shell command and captures stdout/stderr.

#### `verifier.rs`
Deterministic, cheapest-first cascade run before any generated code is accepted:
1. **Parse** — tree-sitter syntax check; bad syntax short-circuits immediately.
2. **Names** — every function definition must appear in the plan's frozen registry. Case/style drift (e.g., `verifyToken` vs `verify_token`) is auto-corrected; unknown names are flagged for model retry.
3. **References** — call sites that drift from a registry name are canonicalized in-place.

The cascade loops up to 8 times to reach a fixpoint after auto-corrections, then returns a `CascadeReport` with the verdict, any violations, applied fixes, and the corrected source.

#### `watcher.rs`
Starts a debounced filesystem watcher (400 ms debounce) on the workspace root. File create/modify events trigger `index_file` (cheap if the hash is unchanged); remove events call `remove_file` to purge the file's chunks and embeddings from the store.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — returns `{"status":"ok"}` |
| `POST` | `/workspace/open` | Open a workspace: triggers a background full index and starts the filesystem watcher |
| `POST` | `/chat` | Conversational Q&A: hybrid retrieval → prompt assembly → LLM → diff extraction |
| `POST` | `/plan` | Interface-first planning gate: returns a frozen module/function spec and plan ID |
| `POST` | `/verify` | Run the deterministic cascade on a code snippet against a plan's registry |
| `POST` | `/implement` | Drive all pending tasks for a plan through generate→verify→retry |
| `GET` | `/index/status` | Returns the current count of indexed files and chunks |

---

## Core Workflows

### Chat

```
POST /chat
{
  "workspace_path": "/path/to/repo",
  "active_file": "src/foo.rs",        // optional
  "selection": {"start": 0, "end": 100}, // optional byte range
  "open_files": ["src/bar.rs"],       // optional
  "user_prompt": "How does the auth middleware work?"
}
```

1. Hybrid retrieval fetches the most relevant code chunks.
2. Chunks are fenced as untrusted data and assembled under a 24 000-character context budget.
3. The LLM is called with the sandboxed system prompt.
4. Unified diffs in the response are extracted and returned as `patches[]` for the editor extension to preview.

### Plan → Verify → Implement

```
POST /plan  { "workspace_path": "...", "goal": "Add a JWT auth module" }
→ { "plan_id": "...", "modules": [...], "frozen_names": [...] }

POST /verify  { "plan_id": "...", "path": "src/auth.rs", "code": "fn verifyToken..." }
→ { "ok": false, "stage": "names", "violations": [...], "corrected": "fn verify_token..." }

POST /implement  { "plan_id": "...", "path": "src/auth.rs", "max_attempts": 3 }
→ { "tasks": [{ "name": "verify_token", "status": "verified", "code": "..." }] }
```

### Workspace Indexing

```
POST /workspace/open  { "workspace_path": "/path/to/repo" }
```

This triggers an **asynchronous** full walk of the workspace. Files are parsed with tree-sitter, chunked, embedded in batches of 32, and stored in SQLite. A filesystem watcher is started afterwards for incremental updates. The response returns immediately with `"indexed": false`; poll `/index/status` to track progress.

---

## Configuration

The daemon reads two environment variables at startup:

| Variable | Default | Description |
|---|---|---|
| `CODEBOT_LLM_BASE` | `http://127.0.0.1:8000` | Base URL of any OpenAI-compatible LLM server |
| `CODEBOT_LLM_MODEL` | `gpt-oss-20b-mxfp4-q8` | Model name sent in API calls |

The SQLite database is written to `.assistant.db` in the working directory. Embedding model weights are cached by fastembed in its default Hugging Face cache directory.

---

## Security Design

Several layers prevent prompt injection from repository content:

1. **Strict instruction/data separation** — only `user_prompt` (in `/chat`) and `goal` (in `/plan`) are ever treated as instructions. All retrieved code is wrapped in `<repo_context>` / `<file>` XML tags and the system prompt explicitly forbids following instructions found inside them.
2. **Fence sanitization** — `</file>`, `<repo_context>`, and `</repo_context>` are neutralized inside chunk content before they reach the model, preventing tag breakout.
3. **Planning constraint** — the planning call uses `response_format: json_schema` with `strict: true` at temperature 0, so the model cannot emit prose or code bodies — only a machine-readable interface spec.
4. **Deterministic name gate** — generated code is verified against a registry that the model cannot modify; name drift is corrected without asking the model.
5. **Path traversal guard** — `tools.rs` canonicalizes all file paths and rejects anything that escapes the workspace root.
6. **Secret skipping** — the indexer skips files whose paths match common secret-file patterns (e.g., `.env`, key files).

---

## Building and Running

```bash
# Build
cargo build --release

# Run (point at a local OpenAI-compatible server on port 8000)
CODEBOT_LLM_BASE=http://127.0.0.1:8000 \
CODEBOT_LLM_MODEL=my-model \
./target/release/codebot-daemon

# Run tests
cargo test
```

The daemon listens on `127.0.0.1:9099`. Embedding model weights (~100 MB) are downloaded from Hugging Face on first startup and cached locally.
