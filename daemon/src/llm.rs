use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone)]
pub struct LlmClient {
    base: String,
    model: String,
    http: reqwest::Client,
}

#[derive(Serialize)]
struct ChatMessage<'a> { role: &'a str, content: &'a str }
#[derive(Serialize)]
struct ChatBody<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
    stream: bool,
}
#[derive(Deserialize)]
struct Choice { message: RespMessage }
#[derive(Deserialize)]
struct RespMessage { content: String }
#[derive(Deserialize)]
struct ChatCompletion { choices: Vec<Choice> }

impl LlmClient {
    pub fn new(base: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base: base.into(),
            model: model.into(),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build().unwrap(),
        }
    }

    pub async fn chat(&self, system: &str, user: &str) -> Result<String> {
        let body = ChatBody {
            model: &self.model,
            messages: vec![
                ChatMessage { role: "system", content: system },
                ChatMessage { role: "user", content: user },
            ],
            temperature: 0.1,
            stream: false,
        };
        let resp: ChatCompletion = self
            .http
            .post(format!("{}/v1/chat/completions", self.base))
            .json(&body)
            .send().await?.error_for_status()?
            .json().await?;
        Ok(resp.choices.into_iter().next()
            .map(|c| c.message.content)
            .unwrap_or_default())
    }
}