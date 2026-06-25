use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct OpenWorkspaceReq {
    pub workspace_path: String,
}

#[derive(Debug, Serialize)]
pub struct OpenWorkspaceResp {
    pub workspace_id: String,
    pub indexed: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Selection {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Deserialize)]
pub struct ChatReq {
    pub workspace_path: String,
    pub active_file: Option<String>,
    pub selection: Option<Selection>,
    #[serde(default)]
    pub open_files: Vec<String>,
    /// The ONLY field whose content is ever treated as instructions.
    pub user_prompt: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Patch {
    pub file: String,
    pub diff: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatResp {
    pub answer: String,
    #[serde(default)]
    pub patches: Vec<Patch>,
    pub context_used: Vec<String>,
    pub trace_id: String,
}

#[derive(Debug, Deserialize)]
pub struct PlanReq {
    pub workspace_path: String,
    /// The single instruction text driving the planning gate.
    pub goal: String,
}

#[derive(Debug, Serialize)]
pub struct PlanResp {
    pub plan_id: String,
    pub modules: Vec<crate::orchestrator::ModuleSpec>,
    /// Canonical names actually frozen into the registry (post-normalization
    /// and dedup) — the source of truth downstream steps must conform to.
    pub frozen_names: Vec<String>,
    pub trace_id: String,
}

#[derive(Debug, Deserialize)]
pub struct VerifyReq {
    pub plan_id: String,
    /// File path; used only to detect the language by extension.
    pub path: String,
    pub code: String,
}

#[derive(Debug, Serialize)]
pub struct ViolationDto {
    /// The cascade stage that produced this violation (e.g. "parse", "names").
    pub checker: String,
    /// Human-readable description of the fault.
    pub message: String,
    pub line: usize,
    /// Canonical target to rename to, when the fault is deterministically fixable.
    pub suggestion: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct VerifyResp {
    pub ok: bool,
    /// The failing cascade stage when `ok` is false; null when the code passed.
    pub stage: Option<String>,
    pub violations: Vec<ViolationDto>,
    /// Deterministic corrections the cascade applied, for transparency.
    pub fixes: Vec<String>,
    /// Source with all deterministic corrections applied; null if unchanged.
    pub corrected: Option<String>,
    pub trace_id: String,
}

#[derive(Debug, Deserialize)]
pub struct ImplementReq {
    pub plan_id: String,
    /// File path; used only to detect the target language by extension.
    pub path: String,
    /// Retry budget per task. Defaults to 3 when omitted.
    #[serde(default = "default_max_attempts")]
    pub max_attempts: u32,
}

fn default_max_attempts() -> u32 {
    3
}

#[derive(Debug, Serialize)]
pub struct TaskOutcomeDto {
    pub name: String,
    /// Terminal status: `verified` or `failed`.
    pub status: String,
    pub attempts: i64,
    pub code: Option<String>,
    /// Unresolved violations when `failed`; empty when verified.
    pub violations: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ImplementResp {
    pub plan_id: String,
    pub tasks: Vec<TaskOutcomeDto>,
    pub trace_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IndexStatus {
    pub files: usize,
    pub chunks: usize,
    pub pending: usize,
    pub last_indexed_ms: i64,
}

#[derive(Debug, Deserialize)]
pub struct ApplyReq {
    pub workspace_path: String,
    /// Workspace-relative target file.
    pub file: String,
    /// The unified diff to apply to the current file content.
    pub diff: String,
    /// When true, verify the diff applies and return the resulting before→after
    /// diff without writing anything to disk.
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApplyResp {
    pub ok: bool,
    pub file: String,
    /// Clean before→after unified diff, for display; empty when the apply failed.
    pub diff: String,
    /// Failure reason when `ok` is false.
    pub error: Option<String>,
    /// Echoes whether this was a preview (no write happened).
    #[serde(default)]
    pub dry_run: bool,
}