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
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat<'a>>,
}
#[derive(Serialize)]
struct ResponseFormat<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
    json_schema: JsonSchema<'a>,
}
#[derive(Serialize)]
struct JsonSchema<'a> {
    name: &'a str,
    strict: bool,
    schema: serde_json::Value,
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
            response_format: None,
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

    /// Structured-output call: constrains the model to a JSON schema via the
    /// OpenAI-compatible `response_format` so the response cannot drift from the
    /// expected shape (the planning gate's hard guarantee). Returns the raw JSON
    /// string for the caller to parse. Temperature is pinned to 0 for determinism.
    pub async fn complete_json(
        &self,
        system: &str,
        user: &str,
        schema_name: &str,
        schema: serde_json::Value,
    ) -> Result<String> {
        let body = ChatBody {
            model: &self.model,
            messages: vec![
                ChatMessage { role: "system", content: system },
                ChatMessage { role: "user", content: user },
            ],
            temperature: 0.0,
            stream: false,
            response_format: Some(ResponseFormat {
                kind: "json_schema",
                json_schema: JsonSchema { name: schema_name, strict: true, schema },
            }),
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