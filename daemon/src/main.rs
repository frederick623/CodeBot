mod api;
mod embed_client;
mod indexer;
mod llm;
mod prompt;
mod retrieval;
mod storage;
mod tools;
mod watcher;

use anyhow::Result;
use axum::{extract::State, routing::{get, post}, Json, Router};
use dashmap::DashMap;
use std::{path::PathBuf, sync::Arc};
use uuid::Uuid;

use api::*;
use embed_client::EmbedClient;
use indexer::Indexer;
use llm::LlmClient;
use prompt::{build_prompt, ContextBudget};
use retrieval::Retriever;
use storage::Store;

#[derive(Clone)]
struct AppState {
    store: Store,
    embed: EmbedClient,
    llm: LlmClient,
    retriever: Arc<Retriever>,
    indexer: Arc<Indexer>,
    workspaces: Arc<DashMap<String, String>>, // path -> workspace_id
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().json().with_env_filter("info").init();

    let store = Store::open(&PathBuf::from(".assistant.db"))?;
    let embed = EmbedClient::new("http://127.0.0.1:8088");
    let llm = LlmClient::new("http://127.0.0.1:11434", "qwen2.5-coder:7b"); // any OpenAI-compatible server
    let retriever = Arc::new(Retriever::new(store.clone(), embed.clone()));
    let indexer = Arc::new(Indexer::new(store.clone(), embed.clone()));

    let state = AppState {
        store, embed, llm, retriever, indexer,
        workspaces: Arc::new(DashMap::new()),
    };

    let app = Router::new()
        .route("/health", get(|| async { Json(serde_json::json!({"status":"ok"})) }))
        .route("/workspace/open", post(open_workspace))
        .route("/chat", post(chat))
        .route("/index/status", get(index_status))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:9099").await?;
    tracing::info!("daemon listening on 127.0.0.1:9099");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn open_workspace(
    State(st): State<AppState>,
    Json(req): Json<OpenWorkspaceReq>,
) -> Json<OpenWorkspaceResp> {
    let id = Uuid::new_v4().to_string();
    st.workspaces.insert(req.workspace_path.clone(), id.clone());

    let root = PathBuf::from(&req.workspace_path);
    let idx = st.indexer.clone();
    let root_clone = root.clone();

    // Initial index in the background, then start the watcher for incremental updates.
    tokio::spawn(async move {
        if let Err(e) = idx.index_workspace(&root_clone).await {
            tracing::error!("initial index failed: {e}");
        }
        if let Err(e) = watcher::spawn_watcher(root_clone, idx) {
            tracing::error!("watcher failed: {e}");
        }
    });

    Json(OpenWorkspaceResp { workspace_id: id, indexed: false })
}

async fn chat(State(st): State<AppState>, Json(req): Json<ChatReq>) -> Json<ChatResp> {
    let trace_id = Uuid::new_v4().to_string();
    tracing::info!(trace_id, prompt = %req.user_prompt, "chat request");

    let root = req.workspace_path.clone();

    // Retrieval: hybrid + rerank.
    let retrieved = match st.retriever.retrieve(&req, &root).await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(trace_id, "retrieval error: {e}");
            vec![]
        }
    };

    // Prompt assembly: file content embedded strictly as fenced DATA.
    let built = build_prompt(
        &req.user_prompt,
        &retrieved,
        ContextBudget { max_chars: 24_000 },
    );

    // Model call.
    let answer = match st.llm.chat(&built.system, &built.user).await {
        Ok(a) => a,
        Err(e) => {
            tracing::error!(trace_id, "llm error: {e}");
            format!("Model call failed: {e}")
        }
    };

    // Patches: parse unified diffs out of the answer if present.
    // (Left minimal here; the tool loop in tools.rs builds previews that the
    //  extension shows before any write, satisfying user-approved patching.)
    let patches = extract_diffs(&answer);

    Json(ChatResp {
        answer,
        patches,
        context_used: built.files_used,
        trace_id,
    })
}

async fn index_status(State(st): State<AppState>) -> Json<IndexStatus> {
    let (files, chunks) = st.store.counts().unwrap_or((0, 0));
    Json(IndexStatus {
        files, chunks, pending: 0,
        last_indexed_ms: 0,
    })
}

/// Tolerant unified-diff extractor.
///
/// Handles diffs whether or not they are wrapped in ```diff/```patch fences,
/// splits multi-file diffs into one `Patch` per file, tolerates `git`-style
/// extended headers (`diff --git`, `index`, `new file mode`, rename lines), and
/// recovers the target path from the `+++`/`---` headers. Each section is parsed
/// with `diffy` for validation; sections that diffy rejects are kept verbatim as
/// long as they structurally look like a diff, so partial/streamed model output
/// is never silently dropped.
fn extract_diffs(answer: &str) -> Vec<Patch> {
    let scrubbed = scrub_fences(answer);
    extract_unified_sections(&scrubbed)
        .into_iter()
        .filter_map(|section| {
            let body = trim_to_diff(&section);
            if body.is_empty() {
                return None;
            }
            let file = diff_target_path(body).unwrap_or_else(|| "unknown".to_string());
            Some(Patch { file, diff: body.to_string() })
        })
        .collect()
}

/// Keep the contents of ```diff/```patch fences, drop other fenced code blocks
/// (so source examples aren't mistaken for diffs), and keep surrounding prose.
fn scrub_fences(answer: &str) -> String {
    let mut out = String::new();
    let mut fence: Option<bool> = None; // Some(keep_contents) while inside a fence
    for line in answer.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("```").or_else(|| trimmed.strip_prefix("~~~")) {
            match fence {
                None => {
                    let lang = rest.trim().to_ascii_lowercase();
                    fence = Some(lang == "diff" || lang == "patch");
                }
                Some(_) => fence = None,
            }
            continue;
        }
        match fence {
            Some(false) => {} // inside a non-diff code block: skip
            _ => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    out
}

/// Split text into per-file unified-diff sections. A section begins at a
/// `diff --git` line or a `--- ` header and extends across its hunks; non-diff
/// prose between sections is discarded.
fn extract_unified_sections(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut sections = Vec::new();
    let mut cur: Option<String> = None;
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let starts_file = line.starts_with("diff --git ")
            || (line.starts_with("--- ")
                && lines.get(i + 1).map_or(false, |n| n.starts_with("+++ ")));
        if starts_file {
            if let Some(s) = cur.take() {
                sections.push(s);
            }
            cur = Some(String::new());
        }
        if let Some(buf) = cur.as_mut() {
            if is_diff_line(line) {
                buf.push_str(line);
                buf.push('\n');
            } else if !line.trim().is_empty() {
                // A non-blank, non-diff line terminates the current section.
                sections.push(cur.take().unwrap());
            } else {
                // Blank line: keep it as hunk context only if a hunk is open.
                buf.push('\n');
            }
        }
        i += 1;
    }
    if let Some(s) = cur.take() {
        sections.push(s);
    }
    sections
}

/// Lines that legitimately appear inside a unified diff section.
fn is_diff_line(line: &str) -> bool {
    line.starts_with("diff --git ")
        || line.starts_with("index ")
        || line.starts_with("new file mode")
        || line.starts_with("deleted file mode")
        || line.starts_with("old mode")
        || line.starts_with("new mode")
        || line.starts_with("similarity index")
        || line.starts_with("rename ")
        || line.starts_with("copy ")
        || line.starts_with("--- ")
        || line.starts_with("+++ ")
        || line.starts_with("@@")
        || line.starts_with('+')
        || line.starts_with('-')
        || line.starts_with(' ')
        || line.starts_with('\\') // "\ No newline at end of file"
}

/// Drop leading git extended-header lines so the body starts at `--- `/`@@`,
/// which is what `diffy` expects. Returns the trimmed slice.
fn trim_to_diff(section: &str) -> &str {
    let mut offset = 0;
    for line in section.lines() {
        if line.starts_with("--- ") || line.starts_with("@@") {
            return section[offset..].trim_end_matches('\n');
        }
        offset += line.len() + 1;
    }
    section.trim_end_matches('\n')
}

/// Resolve the patch target path. Prefer `diffy`'s parsed headers; fall back to
/// sniffing the `+++`/`---`/`diff --git` lines directly.
fn diff_target_path(body: &str) -> Option<String> {
    if let Ok(patch) = diffy::Patch::from_str(body) {
        if let Some(p) = patch.modified().filter(|p| *p != "/dev/null") {
            return Some(clean_path(p));
        }
        if let Some(p) = patch.original().filter(|p| *p != "/dev/null") {
            return Some(clean_path(p));
        }
    }
    for line in body.lines() {
        if let Some(rest) = line.strip_prefix("+++ ") {
            if rest.trim() != "/dev/null" {
                return Some(clean_path(rest));
            }
        }
    }
    for line in body.lines() {
        if let Some(rest) = line.strip_prefix("--- ") {
            if rest.trim() != "/dev/null" {
                return Some(clean_path(rest));
            }
        }
        if let Some(rest) = line.strip_prefix("diff --git ") {
            // "a/path b/path" -> take the b/ side.
            if let Some(b) = rest.split_whitespace().nth(1) {
                return Some(clean_path(b));
            }
        }
    }
    None
}

/// Strip `a/`/`b/` prefixes and any trailing tab-delimited timestamp.
fn clean_path(raw: &str) -> String {
    let mut p = raw.trim();
    if let Some((path, _ts)) = p.split_once('\t') {
        p = path.trim();
    }
    p.strip_prefix("a/")
        .or_else(|| p.strip_prefix("b/"))
        .unwrap_or(p)
        .to_string()
}