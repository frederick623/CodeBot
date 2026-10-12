use crate::api::ChatReq;
use crate::embed_client::EmbedClient;
use crate::storage::{root_prefix, ChunkRow, Store};
use anyhow::Result;
use std::collections::HashMap;

pub struct Retriever {
    store: Store,
    embed: EmbedClient,
}

#[derive(Clone, Debug)]
pub struct Retrieved {
    /// Workspace-relative path (absolute paths never reach the model).
    pub file: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub source: &'static str, // "active" | "symbol" | "vector" | "graph" | "requested"
}

impl Retriever {
    pub fn new(store: Store, embed: EmbedClient) -> Self {
        Self { store, embed }
    }

    pub async fn retrieve(&self, req: &ChatReq, root: &str) -> Result<Vec<Retrieved>> {
        self.retrieve_with(req, root, 8).await
    }

    /// Hybrid retrieval returning the `top` best chunks after reranking.
    pub async fn retrieve_with(&self, req: &ChatReq, root: &str, top: usize) -> Result<Vec<Retrieved>> {
        let prefix = root_prefix(root);
        let mut pool: HashMap<String, Retrieved> = HashMap::new();

        // 1. Active file + selection always included.
        if let Some(active) = &req.active_file {
            let full = format!("{}{}", prefix, active);
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

        // 2. Exact symbol search on identifiers mentioned in the prompt, then
        //    one hop along the call graph: a weak model cannot ask for the
        //    callers/callees itself, so the program brings them in.
        let mut seeds: Vec<ChunkRow> = Vec::new();
        for token in extract_identifiers(&req.user_prompt) {
            for row in self.store.symbol_lookup(root, &token)? {
                if !pool.contains_key(&chunk_key(&row)) {
                    seeds.push(row.clone());
                    pool.insert(chunk_key(&row), to_ret(row, "symbol", &prefix));
                }
            }
        }
        for seed in seeds.iter().take(6) {
            let Some(sym) = &seed.symbol else { continue };
            let (callees, callers) = self.store.graph_neighbours(root, &seed.file, sym)?;
            for callee in callees.iter().take(8) {
                for row in self.store.symbol_lookup(root, callee)? {
                    pool.entry(chunk_key(&row)).or_insert_with(|| to_ret(row, "graph", &prefix));
                }
            }
            for caller in callers.iter().take(8) {
                // `<file>::<symbol>` → find the defining chunk via the symbol table.
                if let Some((file, name)) = caller.rsplit_once("::") {
                    if name == "<file>" { continue; }
                    for row in self.store.symbol_lookup(root, name)? {
                        if row.file == file {
                            pool.entry(chunk_key(&row)).or_insert_with(|| to_ret(row, "graph", &prefix));
                        }
                    }
                }
            }
        }

        // 3. Semantic vector search.
        let qvec = self.embed.embed(&[req.user_prompt.clone()]).await?;
        if let Some(q) = qvec.first() {
            for (row, _score) in self.store.knn(root, q, 20)? {
                pool.entry(chunk_key(&row)).or_insert_with(|| to_ret(row, "vector", &prefix));
            }
        }

        // 4. Rerank everything against the prompt, keep the top window.
        let candidates: Vec<Retrieved> = pool.into_values().collect();
        if candidates.is_empty() { return Ok(vec![]); }

        let docs: Vec<String> = candidates.iter().map(|c| c.content.clone()).collect();
        let ranked = self.embed.rerank(&req.user_prompt, &docs, top).await?;

        Ok(ranked.into_iter().map(|r| candidates[r.index].clone()).collect())
    }
}

fn chunk_key(r: &ChunkRow) -> String {
    format!("{}:{}-{}", r.file, r.start_line, r.end_line)
}
fn to_ret(r: ChunkRow, source: &'static str, prefix: &str) -> Retrieved {
    let file = r.file.strip_prefix(prefix).unwrap_or(&r.file).to_string();
    Retrieved {
        file, start_line: r.start_line, end_line: r.end_line,
        content: r.content, source,
    }
}

/// Heuristic: pull identifier-like tokens from the user's natural language
/// prompt. Lookup is exact-match against the symbol table, so plain words are
/// cheap to include too — they only hit when a definition really carries that
/// name.
fn extract_identifiers(prompt: &str) -> Vec<String> {
    let mut out: Vec<String> = prompt
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() > 3 || (t.len() > 2 && t.chars().any(|c| c.is_uppercase() || c == '_')))
        .map(|s| s.to_string())
        .collect();
    out.sort();
    out.dedup();
    out
}