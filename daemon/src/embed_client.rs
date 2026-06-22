use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct EmbedClient {
    base: String,
    http: reqwest::Client,
}

#[derive(Serialize)]
struct EmbedReq<'a> { inputs: &'a [String], normalize: bool }
#[derive(Deserialize)]
struct EmbedResp { vectors: Vec<Vec<f32>> }

#[derive(Serialize)]
struct RerankReq<'a> { query: &'a str, documents: &'a [String], top_k: Option<usize> }
#[derive(Deserialize)]
pub struct RerankItem { pub index: usize, pub score: f32 }
#[derive(Deserialize)]
struct RerankResp { results: Vec<RerankItem> }

impl EmbedClient {
    pub fn new(base: impl Into<String>) -> Self {
        Self { base: base.into(), http: reqwest::Client::new() }
    }

    pub async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let r: EmbedResp = self
            .http
            .post(format!("{}/embed", self.base))
            .json(&EmbedReq { inputs, normalize: true })
            .send().await?.error_for_status()?
            .json().await?;
        Ok(r.vectors)
    }

    pub async fn rerank(&self, query: &str, docs: &[String], top_k: usize)
        -> Result<Vec<RerankItem>> {
        let r: RerankResp = self
            .http
            .post(format!("{}/rerank", self.base))
            .json(&RerankReq { query, documents: docs, top_k: Some(top_k) })
            .send().await?.error_for_status()?
            .json().await?;
        Ok(r.results)
    }
}