# CodeBot

CodeBot is a **lightweight, prompt-driven IDE** (in the spirit of OpenCode / Claude Code) built around a Rust daemon and a small egui desktop GUI. It is designed for **weak local LLMs**: the program — not the model — does the heavy lifting of indexing the project, resolving symbols and call graphs, choosing what to read, remembering past turns, and turning the model's `SEARCH/REPLACE` blocks into real, syntax-checked file edits. Repository content stays strictly sandboxed from instructions.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Key Technologies](#key-technologies)
- [Source Layout](#source-layout)
- [API Endpoints](#api-endpoints)
- [Core Workflows](#core-workflows)
  - [Agent Turn (Ask / Agent mode)](#agent-turn-ask--agent-mode)
  - [Chat](#chat)
  - [Plan → Verify → Implement](#plan--verify--implement)
  - [Workspace Indexing](#workspace-indexing)
- [Configuration](#configuration)
- [Desktop GUI](#desktop-gui)
- [Security Design](#security-design)
- [Building and Running](#building-and-running)

---

## Architecture Overview

```
Editor Extension  ──HTTP──►  codebot-daemon (localhost:9099)
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

The daemon listens on `localhost:9099`. An editor extension (not included in this repo) communicates with it via HTTP JSON calls. The daemon indexes the workspace in the background and keeps the index fresh through a filesystem watcher.

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
├── main.rs          # Headless daemon entry point (logging + server::run)
├── lib.rs           # Library crate shared by the daemon and the GUI
├── server.rs        # HTTP API: routes, AppState, agent run registry, settings, git endpoints
├── agent.rs         # Program-driven agent loop: memory + project map + retrieval → plan → answer/edit
├── edits.rs         # SEARCH/REPLACE block parser and fuzzy, indentation-tolerant applier
├── api.rs           # Request/response DTOs for the legacy HTTP endpoints
├── embed_client.rs  # In-process embedding + reranking via fastembed/ONNX
├── indexer.rs       # Workspace indexer: file filter, AST chunking, symbol/call/import graph, pruning
├── llm.rs           # OpenAI-compatible LLM client (history chat, lenient JSON, model listing)
├── orchestrator.rs  # Planning gate and implement loop (plan → generate → verify → retry)
├── prompt.rs        # Prompt assembly with context budgeting and untrusted-data fencing
├── retrieval.rs     # Root-scoped hybrid retrieval: symbols + vector KNN + call-graph expansion + rerank
├── storage.rs       # SQLite store: files/chunks/embeddings, symbol graph, settings, messages, memory
├── tools.rs         # Tool executor: safe file IO, literal search, patch apply, git changes/revert
├── verifier.rs      # Deterministic verification cascade: parse → names → references
├── watcher.rs       # Filesystem watcher that feeds changed paths to the incremental indexer
└── bin/gui/
    ├── main.rs      # egui desktop app: chat transcript, prompt box, Changes pane, settings, log
    └── markdown.rs  # Markdown → egui renderer (headings, tables, lists, code) + glyph normalisation
```

### Module Responsibilities

#### `main.rs` / `server.rs`
`main.rs` only initialises logging and calls `server::run`. `server.rs` bootstraps all components into a shared `AppState`, registers the HTTP routes, keeps the in-flight agent runs (`/agent/run` → `/agent/status`), hot-swaps the LLM client when settings change, and contains the unified-diff extraction utilities used by `/chat`.

#### `agent.rs`
The core of the IDE. One turn runs as a *program-driven* loop so a small model only ever has to do one easy thing at a time:

1. **Gather (program).** Recent memory notes, conversation history, a compact *project map* (every indexed file with the symbols it defines) and hybrid retrieval seeded by the prompt (symbols → one-hop call-graph neighbours → vector KNN → rerank).
2. **Plan (bounded model call, JSON).** Decide `intent` (`answer` | `edit`), up to 4 files to read and up to 3 literal search terms. In **Ask mode**, explicit indexed paths (or unique basenames) bypass the model planner when all named files are readable and fit without truncation: at most 4 files, each within the file-read budget, together using at most half the context budget. Related hybrid retrieval is still included. Ambiguous paths, larger files and Agent mode retain model planning. A failed planner falls back to a **read-only answer** using retrieval, never automatic edits.
3. **Answer or edit (model).** Ask mode answers in Markdown. Agent mode requests `SEARCH/REPLACE` blocks (never unified diffs, which weak models get wrong), applies them in memory, parse-checks the result with tree-sitter, and retries once with precise feedback plus the live file contents before writing to disk and re-indexing.
4. **Remember (program).** The turn and a one-line memory note are persisted per workspace.

#### Planning latency and progress

- The agent's file-selection planner has a **45-second total deadline**, including any compatibility retry, and sends **`max_tokens: 1024`**. For model IDs containing `gpt-oss`, it also requests **`reasoning_effort: "low"`**; whether the server honors these options depends on the backend.
- Strict JSON-schema output is tried first. Only an explicit unsupported `response_format`, `json_schema`, or `reasoning_effort` error (HTTP 400/422) permits one plain-chat fallback, without structured-output/reasoning options. The token cap and original deadline remain in force. Authentication, connection, rate-limit, server, timeout, malformed and truncated-output errors do not trigger a second planning call.
- The GUI's existing step feed now receives start/end timing and **waiting updates every 5 seconds** for retrieval, planning, answering and edit generation. Planning reports model ID and input byte counts, and compatibility retries are visible. Waiting updates indicate elapsed time, not proof that the backend is generating tokens.
- Final answers/edits retain the existing 600-second request timeout and non-streaming transport; these are status heartbeats, not token streaming. This planner deadline does not change the separate `/plan` interface-spec workflow. Cancelling a timed-out HTTP request stops the client waiting; stopping generation on the inference server depends on that server.

#### `edits.rs`
Parses `path` + `<<<<<<< SEARCH / ======= / >>>>>>> REPLACE` blocks and applies them with three strategies — exact match, indentation-tolerant line match, then fuzzy match (similarity ≥ 0.88) — re-indenting the replacement to the matched block. Empty `SEARCH` creates a new file. CRLF endings are preserved.

#### `api.rs`
Pure data layer: all `Request` / `Response` structs for the HTTP API, serialized/deserialized with serde. No logic.

#### `embed_client.rs`
Wraps `fastembed`'s `TextEmbedding` (BGE-Small-EN-V1.5) and `TextRerank` (BGE-Reranker-Base) behind a `Clone`-able async interface. ONNX inference is CPU-bound and runs in `spawn_blocking` to keep it off the async worker threads. Models are loaded once, on their first embedding or reranking request, so the daemon can serve health checks immediately.

#### `indexer.rs`
- **Full index**: walks the workspace respecting `.gitignore`, skips files > 1 MB and files whose paths match secret-detection heuristics.
- **Incremental index**: computes a BLAKE3 hash; if the hash matches the stored value, the file is skipped.
- **AST chunking**: parses each file with tree-sitter, extracts top-level definitions as chunks (one chunk per function/class/etc.); falls back to a sliding line-window for unsupported file types.
- **Symbol graph**: extracts defined names, imported modules, and call sites; inserts them as nodes and typed edges (`import`, `call`) into the store for the symbol-lookup retrieval path.

#### `llm.rs`
Thin wrapper around the OpenAI `/v1/chat/completions` API. Provides these call modes:
- `chat()` — free-text completion (temperature 0.1).
- `complete_json()` — structured output constrained by a JSON schema via `response_format` (temperature 0, deterministic).
- `complete_plan_json()` — bounded file-selection planning with progress and a narrowly scoped compatibility fallback.

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
| `GET` | `/index/status?workspace_path=` | Indexed file/chunk counts for a workspace (whole store when omitted) |
| `POST` | `/agent/run` | Start an agent turn `{workspace_path, prompt, mode: "ask"|"agent"}` → `{run_id}` |
| `GET` | `/agent/status?run_id=` | Poll a run: `{steps[], done, result{answer, edited[], failed[], context_used[]}, error}` |
| `GET` / `DELETE` | `/history?workspace_path=` | Read / clear the persisted conversation for a workspace |
| `GET` / `POST` | `/settings` | Read / update `llm_base`, `llm_model`, `llm_api_key` (write-only), `context_chars` — applied immediately |
| `GET` | `/models` | Model IDs reported by the configured LLM server (`/v1/models`) |
| `GET` | `/git/changes?workspace_path=` | Working-tree changes vs `HEAD`: `[{path, status, diff, additions, deletions}]` |
| `POST` | `/git/revert` | `{workspace_path, file}` — restore a tracked file from `HEAD` or delete an untracked one |

---

## Core Workflows

### Agent Turn (Ask / Agent mode)

```
POST /agent/run   { "workspace_path": "/path/to/repo", "prompt": "make greet use an f-string", "mode": "agent" }
→ { "run_id": "…" }

GET /agent/status?run_id=…
→ { "steps": ["gathering context…", "plan: edit greet", "reading src/greet.py", "applying 2 edit block(s)", "wrote src/greet.py"],
    "done": true,
    "result": { "answer": "…markdown…", "edited": [{"path": "src/greet.py", "created": false}], "failed": [], "context_used": [...] } }
```

The run executes in the background; the GUI streams `steps` into the transcript and refreshes the Changes pane when `done` flips. `mode: "ask"` never writes files.

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

This triggers an **asynchronous** full walk of the workspace. Only source/config/doc files pass the `is_indexable` filter (lockfiles, licences, minified bundles, `.map`, binaries and `.gitignore`d paths are skipped); entries for files that no longer exist are pruned. Files are parsed with tree-sitter, chunked, embedded in batches of 32, and stored in SQLite under their absolute path, so one `.assistant.db` can hold several workspaces and every query is scoped to the workspace root. A filesystem watcher is started afterwards for incremental updates. The response returns immediately with `"indexed": false`; poll `/index/status?workspace_path=…` to track progress.

---

## Configuration

LLM settings are resolved as **stored settings → environment → defaults**, and can be changed at runtime from the GUI (⚙ Settings) or `POST /settings` without restarting.

| Setting / Variable | Default | Description |
|---|---|---|
| `llm_base` / `CODEBOT_LLM_BASE` | `http://localhost:8000` | Base URL of any OpenAI-compatible server (llama.cpp, Ollama, LM Studio, vLLM, …) |
| `llm_model` / `CODEBOT_LLM_MODEL` | `gpt-oss-20b-mxfp4-q8` | Model name sent in API calls; pick from `/models` in the GUI |
| `llm_api_key` / `CODEBOT_LLM_API_KEY` (or `OPENAI_API_KEY`) | *(none)* | Bearer token, only needed for hosted endpoints |
| `context_chars` | `24000` | Character budget for repository context per model call — lower it for small-context models |

The SQLite database is written to `.assistant.db` in the daemon's working directory. Embedding model weights are cached by fastembed in its default Hugging Face cache directory.

**Working with a weak local model.** `response_format: json_schema` is attempted first and falls back to plain prompting with JSON extraction when the server rejects it; model calls time out after 10 minutes; edits are requested as short `SEARCH/REPLACE` blocks and matched fuzzily, so minor copy errors still apply; syntax is verified before anything is written.

---

## Desktop GUI

```bash
cargo run --bin gui          # connects to a running daemon on :9099 or starts one in-process
```

- **Transcript** — Markdown answers rendered natively (headings, tables, lists, fenced code, inline code/bold/italic). Typographic characters the bundled fonts lack (e.g. non-breaking hyphens) are normalised so no "□" boxes appear.
- **Prompt** — `Enter` sends, `Ctrl+Enter` inserts a newline. Toggle **Agent** (may edit files) / **Ask** (read-only).
- **Changes pane (right)** — every file that differs from git `HEAD`, with `A`/`M`/`D` badges, `+/-` line counts, an expandable coloured diff and a per-file **Revert**. Refreshes after each run and every 5 s.
- **Settings** — LLM base URL, model (with a dropdown from `/models`), API key, context budget; saved to the store and applied immediately.
- **Status / Log** — indexed file and chunk counts for the open workspace, plus agent steps and server events.

---

## Security Design

Several layers prevent prompt injection from repository content:

1. **Strict instruction/data separation** — only `user_prompt` (in `/chat`) and `goal` (in `/plan`) are ever treated as instructions. All retrieved code is wrapped in `<repo_context>` / `<file>` XML tags and the system prompt explicitly forbids following instructions found inside them.
2. **Fence sanitization** — `</file>`, `<repo_context>`, and `</repo_context>` are neutralized inside chunk content before they reach the model, preventing tag breakout.
3. **Planning constraint** — the planning call uses `response_format: json_schema` with `strict: true` at temperature 0, so the model cannot emit prose or code bodies — only a machine-readable interface spec.
4. **Deterministic name gate** — generated code is verified against a registry that the model cannot modify; name drift is corrected without asking the model.
5. **Path traversal guard** — `tools.rs` rejects absolute paths and `..` components lexically, then canonicalizes and rejects anything that escapes the workspace root (also applied to `/git/revert`).
6. **Secret skipping** — the indexer skips files whose paths match common secret-file patterns (e.g., `.env`, key files).

---

## Building and Running

```bash
# Build
cargo build --release

# Run the headless daemon (point at a local OpenAI-compatible server on port 8000)
CODEBOT_LLM_BASE=http://localhost:8000 \
CODEBOT_LLM_MODEL=my-model \
./target/release/codebot-daemon

# Or run the desktop GUI (starts the daemon in-process if none is listening)
./target/release/gui

# Run tests
cargo test
```

Typical local setup: start `llama-server -m model.gguf --port 8000` (or `ollama serve` with `CODEBOT_LLM_BASE=http://localhost:11434`), launch the GUI, type the project path in **Workspace** and press **Open**, wait for the file/chunk counter to settle, then ask or instruct.

The daemon listens on `localhost:9099`. Embedding model weights (~100 MB) are downloaded from Hugging Face on first startup and cached locally.
