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

#[derive(Debug, Serialize)]
pub struct Patch {
    pub file: String,
    pub diff: String,
}

#[derive(Debug, Serialize)]
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
pub struct NameViolationDto {
    pub found: String,
    pub line: usize,
    /// Canonical name to rename to, when the only fault was casing/style.
    pub suggestion: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct VerifyResp {
    pub ok: bool,
    pub violations: Vec<NameViolationDto>,
    /// Source with auto-correctable renames applied; null if nothing changed.
    pub corrected: Option<String>,
    pub trace_id: String,
}

#[derive(Debug, Serialize)]
pub struct IndexStatus {
    pub files: usize,
    pub chunks: usize,
    pub pending: usize,
    pub last_indexed_ms: i64,
}