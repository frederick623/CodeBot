use crate::api::{ChatReq};
use crate::embed_client::EmbedClient;
use crate::storage::{ChunkRow, Store};
use anyhow::Result;
use std::collections::HashMap;

pub struct Retriever {
    store: Store,
    embed: EmbedClient,
}

#[derive(Clone)]
pub struct Retrieved {
    pub file: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub source: &'static str, // "active" | "symbol" | "vector"
}

impl Retriever {
    pub fn new(store: Store, embed: EmbedClient) -> Self {
        Self { store, embed }
    }

    pub async fn retrieve(&self, req: &ChatReq, root: &str) -> Result<Vec<Retrieved>> {
        let mut pool: HashMap<String, Retrieved> = HashMap::new();

        // 1. Active file + selection always included.
        if let Some(active) = &req.active_file {
            let full = format!("{}/{}", root, active);
            if let Ok(content) = std::fs::read_to_string(&full) {
                let snippet = if let Some(sel) = &req.selection {
                    content.get(sel.start..sel.end).unwrap_or(&content).to_string()
                } else {
                    content
                };
                let key = format!("{}#active", active);
                pool.insert(key, Retrieved {
                    file: active.clone(), start_line: 0, end_line: 0,
                    content: snippet, source: "active",
                });
            }
        }

        // 2. Exact symbol search on identifiers mentioned in the prompt.
        for token in extract_identifiers(&req.user_prompt) {
            for row in self.store.symbol_lookup(&token)? {
                pool.entry(chunk_key(&row)).or_insert_with(|| to_ret(row, "symbol"));
            }
        }

        // 3. Semantic vector search.
        let qvec = self.embed.embed(&[req.user_prompt.clone()]).await?;
        if let Some(q) = qvec.first() {
            for (row, _score) in self.store.knn(q, 20)? {
                pool.entry(chunk_key(&row)).or_insert_with(|| to_ret(row, "vector"));
            }
        }

        // 4. Rerank everything against the prompt, keep the top window.
        let candidates: Vec<Retrieved> = pool.into_values().collect();
        if candidates.is_empty() { return Ok(vec![]); }

        let docs: Vec<String> = candidates.iter().map(|c| c.content.clone()).collect();
        let ranked = self.embed.rerank(&req.user_prompt, &docs, 8).await?;

        Ok(ranked.into_iter().map(|r| candidates[r.index].clone()).collect())
    }
}

fn chunk_key(r: &ChunkRow) -> String {
    format!("{}:{}-{}", r.file, r.start_line, r.end_line)
}
fn to_ret(r: ChunkRow, source: &'static str) -> Retrieved {
    Retrieved {
        file: r.file, start_line: r.start_line, end_line: r.end_line,
        content: r.content, source,
    }
}

/// Heuristic: pull identifier-like tokens from the user's natural language prompt.
fn extract_identifiers(prompt: &str) -> Vec<String> {
    prompt
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() > 2 && t.chars().any(|c| c.is_uppercase() || c == '_'))
        .map(|s| s.to_string())
        .collect()
}