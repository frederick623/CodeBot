//! In-process embedding + reranking.
//!
//! Inference runs inside the daemon via `fastembed` (ONNX Runtime); there is no
//! separate microservice. Model weights are fetched from Hugging Face on first
//! use and cached locally. The public API is unchanged from the previous HTTP
//! client so `retrieval` and `indexer` need no edits.
use anyhow::Result;
use std::sync::Arc;

use fastembed::{
    EmbeddingModel, RerankInitOptions, RerankerModel, TextEmbedding, TextInitOptions, TextRerank,
};
use parking_lot::Mutex;

const EMBED_BATCH: usize = 32;

// fastembed's inference methods take `&mut self`, so the shared models live
// behind a Mutex. Calls run inside `spawn_blocking`, so holding the (non-async)
// lock across the synchronous inference is fine.
#[derive(Clone)]
pub struct EmbedClient {
    embed: Arc<Mutex<TextEmbedding>>,
    rerank: Arc<Mutex<TextRerank>>,
}

pub struct RerankItem { pub index: usize, pub score: f32 }

impl EmbedClient {
    /// Load both models. This blocks while weights are downloaded/loaded, so it
    /// is called once during daemon startup.
    pub fn new() -> Result<Self> {
        let embed = TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::BGESmallENV15))?;
        let rerank = TextRerank::try_new(RerankInitOptions::new(RerankerModel::BGERerankerBase))?;
        Ok(Self {
            embed: Arc::new(Mutex::new(embed)),
            rerank: Arc::new(Mutex::new(rerank)),
        })
    }

    pub async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let model = self.embed.clone();
        let docs: Vec<String> = inputs.to_vec();
        // CPU-bound inference: keep it off the async runtime's worker threads.
        let vecs = tokio::task::spawn_blocking(move || model.lock().embed(docs, Some(EMBED_BATCH)))
            .await??;
        Ok(vecs)
    }

    pub async fn rerank(&self, query: &str, docs: &[String], top_k: usize)
        -> Result<Vec<RerankItem>> {
        let model = self.rerank.clone();
        let query = query.to_string();
        let documents: Vec<String> = docs.to_vec();
        let items = tokio::task::spawn_blocking(move || -> Result<Vec<RerankItem>> {
            // fastembed returns results sorted by descending relevance score.
            let results = model.lock().rerank(query, documents, false, None)?;
            Ok(results
                .into_iter()
                .take(top_k)
                .map(|r| RerankItem { index: r.index, score: r.score })
                .collect())
        })
        .await??;
        Ok(items)
    }
}