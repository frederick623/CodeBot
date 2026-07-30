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
use tokio::sync::OnceCell;

const EMBED_BATCH: usize = 32;

// fastembed's inference methods take `&mut self`, so the shared models live
// behind a Mutex. Calls run inside `spawn_blocking`, so holding the (non-async)
// lock across the synchronous inference is fine.
#[derive(Clone)]
pub struct EmbedClient {
    models: Arc<OnceCell<Arc<Models>>>,
}

struct Models {
    embed: Mutex<TextEmbedding>,
    rerank: Mutex<TextRerank>,
}

pub struct RerankItem {
    pub index: usize,
    pub score: f32,
}

impl EmbedClient {
    /// Create the client without loading models so the daemon can begin serving
    /// health checks while the model cache is initialized on first use.
    pub fn new() -> Self {
        Self {
            models: Arc::new(OnceCell::new()),
        }
    }

    async fn models(&self) -> Result<Arc<Models>> {
        Ok(self
            .models
            .get_or_try_init(|| async {
                tokio::task::spawn_blocking(|| -> Result<Arc<Models>> {
                    tracing::info!("loading embedding and reranking models");
                    let embed = TextEmbedding::try_new(TextInitOptions::new(
                        EmbeddingModel::BGESmallENV15,
                    ))?;
                    let rerank = TextRerank::try_new(RerankInitOptions::new(
                        RerankerModel::BGERerankerBase,
                    ))?;
                    Ok(Arc::new(Models {
                        embed: Mutex::new(embed),
                        rerank: Mutex::new(rerank),
                    }))
                })
                .await?
            })
            .await?
            .clone())
    }

    pub async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let models = self.models().await?;
        let docs: Vec<String> = inputs.to_vec();
        // CPU-bound inference: keep it off the async runtime's worker threads.
        let vecs =
            tokio::task::spawn_blocking(move || models.embed.lock().embed(docs, Some(EMBED_BATCH)))
                .await??;
        Ok(vecs)
    }

    pub async fn rerank(
        &self,
        query: &str,
        docs: &[String],
        top_k: usize,
    ) -> Result<Vec<RerankItem>> {
        let models = self.models().await?;
        let query = query.to_string();
        let documents: Vec<String> = docs.to_vec();
        let items = tokio::task::spawn_blocking(move || -> Result<Vec<RerankItem>> {
            // fastembed returns results sorted by descending relevance score.
            let results = models.rerank.lock().rerank(query, documents, false, None)?;
            Ok(results
                .into_iter()
                .take(top_k)
                .map(|r| RerankItem {
                    index: r.index,
                    score: r.score,
                })
                .collect())
        })
        .await??;
        Ok(items)
    }
}

#[cfg(test)]
mod tests {
    use super::EmbedClient;

    #[test]
    fn construction_does_not_initialize_models() {
        let client = EmbedClient::new();
        assert!(client.models.get().is_none());
    }
}
