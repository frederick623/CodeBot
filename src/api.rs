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

#[derive(Debug, Serialize)]
pub struct IndexStatus {
    pub files: usize,
    pub chunks: usize,
    pub pending: usize,
    pub last_indexed_ms: i64,
}