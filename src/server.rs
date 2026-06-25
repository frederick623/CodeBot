//! HTTP server bootstrap and route handlers.
//!
//! Extracted from `main.rs` so both the headless daemon and the GUI binary can
//! start the server through [`run`]. The GUI uses this to bring the server up
//! in-process when `/health` is not already answering.
use anyhow::Result;
use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use dashmap::DashMap;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use uuid::Uuid;

use crate::api::*;
use crate::embed_client::EmbedClient;
use crate::indexer::{detect_lang, Indexer};
use crate::llm::LlmClient;
use crate::orchestrator::Orchestrator;
use crate::prompt::{build_prompt, fence_context, ContextBudget};
use crate::retrieval::Retriever;
use crate::storage::Store;
use crate::tools::ToolExecutor;
use crate::watcher;

/// Address the daemon listens on. Shared so the GUI knows where to connect.
pub const DEFAULT_ADDR: &str = "127.0.0.1:9099";

#[derive(Clone)]
pub struct AppState {
    store: Store,
    embed: EmbedClient,
    llm: LlmClient,
    retriever: Arc<Retriever>,
    indexer: Arc<Indexer>,
    orchestrator: Arc<Orchestrator>,
    workspaces: Arc<DashMap<String, String>>, // path -> workspace_id
}

/// Build the shared application state: opens the store, loads the in-process
/// embedding/rerank models, and wires up retrieval, indexing and orchestration.
/// This blocks while model weights are loaded, so call it once at startup.
pub fn build_state() -> Result<AppState> {
    let store = Store::open(&PathBuf::from(".assistant.db"))?;
    let embed = EmbedClient::new()?; // loads embedding + rerank models in-process
    // Any OpenAI-compatible server. Overridable via env so the daemon can point
    // at a different host/model without recompiling.
    let llm_base = std::env::var("CODEBOT_LLM_BASE")
        .unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());
    let llm_model = std::env::var("CODEBOT_LLM_MODEL")
        .unwrap_or_else(|_| "gpt-oss-20b-mxfp4-q8".to_string());
    tracing::info!(%llm_base, %llm_model, "llm configured");
    let llm = LlmClient::new(llm_base, llm_model);
    let retriever = Arc::new(Retriever::new(store.clone(), embed.clone()));
    let indexer = Arc::new(Indexer::new(store.clone(), embed.clone()));
    let orchestrator = Arc::new(Orchestrator::new(store.clone(), llm.clone()));

    Ok(AppState {
        store,
        embed,
        llm,
        retriever,
        indexer,
        orchestrator,
        workspaces: Arc::new(DashMap::new()),
    })
}

/// Construct the router with every endpoint wired to `state`.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(|| async { Json(serde_json::json!({"status":"ok"})) }))
        .route("/workspace/open", post(open_workspace))
        .route("/chat", post(chat))
        .route("/plan", post(plan))
        .route("/verify", post(verify))
        .route("/implement", post(implement))
        .route("/apply", post(apply))
        .route("/index/status", get(index_status))
        .with_state(state)
}

/// Build state, bind [`DEFAULT_ADDR`], and serve until shutdown.
pub async fn run() -> Result<()> {
    let state = build_state()?;
    let app = router(state);
    let listener = tokio::net::TcpListener::bind(DEFAULT_ADDR).await?;
    tracing::info!("daemon listening on {DEFAULT_ADDR}");
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
    //  client shows before any write, satisfying user-approved patching.)
    let patches = extract_diffs(&answer);

    Json(ChatResp {
        answer,
        patches,
        context_used: built.files_used,
        trace_id,
    })
}

/// Interface-first planning gate. Gathers repo context for the goal (fenced as
/// untrusted DATA, same path as chat), then runs the orchestrator's structured
/// planning call and freezes the resulting registry.
async fn plan(State(st): State<AppState>, Json(req): Json<PlanReq>) -> Json<PlanResp> {
    let trace_id = Uuid::new_v4().to_string();
    tracing::info!(trace_id, goal = %req.goal, "plan request");

    // Reuse the hybrid retriever to ground planning in the existing code.
    let chat_req = ChatReq {
        workspace_path: req.workspace_path.clone(),
        active_file: None,
        selection: None,
        open_files: vec![],
        user_prompt: req.goal.clone(),
    };
    let retrieved = st
        .retriever
        .retrieve(&chat_req, &req.workspace_path)
        .await
        .unwrap_or_default();
    let (context, _used) = fence_context(&retrieved, &ContextBudget { max_chars: 16_000 });

    match st.orchestrator.plan(&req.goal, &context).await {
        Ok(out) => Json(PlanResp {
            plan_id: out.plan_id,
            modules: out.spec.modules,
            frozen_names: out.frozen_names,
            trace_id,
        }),
        Err(e) => {
            tracing::error!(trace_id, "plan error: {e}");
            Json(PlanResp {
                plan_id: String::new(),
                modules: vec![],
                frozen_names: vec![],
                trace_id,
            })
        }
    }
}

/// Deterministic verification cascade. Parses the candidate `code` with
/// tree-sitter and runs the composed gate (parse → names → references) against
/// the frozen registry for `plan_id`, returning the failing stage and its
/// violations plus a fully auto-corrected source where every fault was
/// deterministically fixable. Unknown language or empty registry → no-op pass.
async fn verify(State(st): State<AppState>, Json(req): Json<VerifyReq>) -> Json<VerifyResp> {
    let trace_id = Uuid::new_v4().to_string();
    let registry: Vec<String> = st
        .store
        .get_plan_symbols(&req.plan_id)
        .unwrap_or_default()
        .into_iter()
        .map(|s| s.name)
        .collect();

    let report = match detect_lang(Path::new(&req.path)) {
        Some(lang) if !registry.is_empty() => crate::verifier::run_cascade(&req.code, lang, &registry),
        _ => crate::verifier::CascadeReport {
            ok: true,
            stage: None,
            violations: vec![],
            fixes: vec![],
            corrected: None,
        },
    };

    Json(VerifyResp {
        ok: report.ok,
        stage: report.stage,
        violations: report
            .violations
            .into_iter()
            .map(|v| ViolationDto {
                checker: v.checker,
                message: v.message,
                line: v.line,
                suggestion: v.suggestion,
            })
            .collect(),
        fixes: report.fixes,
        corrected: report.corrected,
        trace_id,
    })
}

/// Execution loop: drive every pending task for a plan through
/// generate→verify→retry. Verification (the AST name gate) is the gate; the
/// store owns task state, so `verified` is only reached when a deterministic
/// checker passes. Returns the terminal state of each task that was processed.
async fn implement(State(st): State<AppState>, Json(req): Json<ImplementReq>) -> Json<ImplementResp> {
    let trace_id = Uuid::new_v4().to_string();
    tracing::info!(trace_id, plan_id = %req.plan_id, "implement request");

    let tasks = match st
        .orchestrator
        .implement(&req.plan_id, &req.path, req.max_attempts)
        .await
    {
        Ok(outcomes) => outcomes
            .into_iter()
            .map(|o| TaskOutcomeDto {
                name: o.name,
                status: o.status,
                attempts: o.attempts,
                code: o.code,
                violations: o.violations,
            })
            .collect(),
        Err(e) => {
            tracing::error!(trace_id, "implement error: {e}");
            vec![]
        }
    };

    Json(ImplementResp { plan_id: req.plan_id, tasks, trace_id })
}

async fn index_status(State(st): State<AppState>) -> Json<IndexStatus> {
    let (files, chunks) = st.store.counts().unwrap_or((0, 0));
    Json(IndexStatus {
        files,
        chunks,
        pending: 0,
        last_indexed_ms: 0,
    })
}

/// Apply a unified diff to a workspace file, gated by explicit client approval.
///
/// Reads the current file (the "before"), applies the supplied diff with
/// `diffy`, and computes a clean before→after unified diff (via `similar`) that
/// reflects exactly what is written — independent of how the model formatted the
/// incoming diff. When `dry_run` is set the result is returned without touching
/// disk, so the client can confirm the diff applies first. All file access is
/// sandboxed to the workspace by [`ToolExecutor`].
async fn apply(Json(req): Json<ApplyReq>) -> Json<ApplyResp> {
    let trace_id = Uuid::new_v4().to_string();
    tracing::info!(trace_id, file = %req.file, dry_run = req.dry_run, "apply request");

    let tools = ToolExecutor::new(req.workspace_path.clone());
    let after = (|| -> anyhow::Result<String> {
        let before = tools.read_file(&req.file).unwrap_or_default();
        let patch = diffy::Patch::from_str(&req.diff)
            .map_err(|e| anyhow::anyhow!("parse diff: {e}"))?;
        diffy::apply(&before, &patch).map_err(|e| anyhow::anyhow!("apply diff: {e}"))
    })();

    let fail = |e: String| ApplyResp {
        ok: false,
        file: req.file.clone(),
        diff: String::new(),
        error: Some(e),
        dry_run: req.dry_run,
    };

    match after {
        Ok(after) => {
            // Clean diff from the real on-disk "before" vs the new "after",
            // computed before any write so the comparison is exact.
            let preview = tools.make_patch(&req.file, &after).map(|p| p.diff).unwrap_or_default();
            if req.dry_run {
                return Json(ApplyResp { ok: true, file: req.file, diff: preview, error: None, dry_run: true });
            }
            match tools.apply_full(&req.file, &after) {
                Ok(()) => Json(ApplyResp { ok: true, file: req.file, diff: preview, error: None, dry_run: false }),
                Err(e) => Json(fail(e.to_string())),
            }
        }
        Err(e) => {
            tracing::error!(trace_id, "apply error: {e}");
            Json(fail(e.to_string()))
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a unique temp workspace containing `file` with `content`.
    fn temp_workspace(file: &str, content: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("codebot_apply_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(file), content).unwrap();
        dir
    }

    /// A dry-run apply returns the before→after diff but leaves the file untouched.
    #[tokio::test]
    async fn apply_dry_run_does_not_write() {
        let before = "line1\nline2\nline3\n";
        let after = "line1\nline2-modified\nline3\n";
        let dir = temp_workspace("foo.txt", before);
        let diff = diffy::create_patch(before, after).to_string();

        let resp = apply(Json(ApplyReq {
            workspace_path: dir.display().to_string(),
            file: "foo.txt".to_string(),
            diff,
            dry_run: true,
        }))
        .await
        .0;

        assert!(resp.ok, "dry run should succeed: {:?}", resp.error);
        assert!(resp.dry_run);
        assert!(resp.diff.contains("line2-modified"));
        // File on disk is unchanged.
        assert_eq!(std::fs::read_to_string(dir.join("foo.txt")).unwrap(), before);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A real apply writes the transformed content and reports a clean diff.
    #[tokio::test]
    async fn apply_writes_changes() {
        let before = "alpha\nbeta\ngamma\n";
        let after = "alpha\nBETA\ngamma\n";
        let dir = temp_workspace("bar.txt", before);
        let diff = diffy::create_patch(before, after).to_string();

        let resp = apply(Json(ApplyReq {
            workspace_path: dir.display().to_string(),
            file: "bar.txt".to_string(),
            diff,
            dry_run: false,
        }))
        .await
        .0;

        assert!(resp.ok, "apply should succeed: {:?}", resp.error);
        assert!(!resp.dry_run);
        assert_eq!(std::fs::read_to_string(dir.join("bar.txt")).unwrap(), after);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A diff whose context does not match the file fails cleanly without writing.
    #[tokio::test]
    async fn apply_rejects_mismatched_diff() {
        let before = "one\ntwo\nthree\n";
        let dir = temp_workspace("baz.txt", before);
        // Build a diff against unrelated content so the hunk context cannot match.
        let diff = diffy::create_patch("XXXX\nYYYY\nZZZZ\n", "XXXX\nCHANGED\nZZZZ\n").to_string();

        let resp = apply(Json(ApplyReq {
            workspace_path: dir.display().to_string(),
            file: "baz.txt".to_string(),
            diff,
            dry_run: false,
        }))
        .await
        .0;

        assert!(!resp.ok);
        assert!(resp.error.is_some());
        assert_eq!(std::fs::read_to_string(dir.join("baz.txt")).unwrap(), before);

        std::fs::remove_dir_all(&dir).ok();
    }
}
