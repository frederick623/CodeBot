use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone)]
pub struct LlmClient {
    base: String,
    model: String,
    /// Optional bearer token sent as `Authorization: Bearer <key>`. Required by
    /// OpenAI-compatible servers that enforce auth (vLLM `--api-key`, etc.).
    api_key: Option<String>,
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
            api_key: None,
            http: reqwest::Client::builder()
                // Local models on CPU can take minutes on a long prompt.
                .timeout(Duration::from_secs(600))
                .build().unwrap(),
        }
    }

    /// Attach a bearer token used to authenticate every request. Empty values
    /// are treated as "no key" so an unset env var doesn't send `Bearer `.
    pub fn with_api_key(mut self, key: Option<String>) -> Self {
        self.api_key = key.filter(|k| !k.trim().is_empty());
        self
    }

    /// Build the chat-completions POST, applying bearer auth when configured.
    fn endpoint(&self) -> reqwest::RequestBuilder {
        let req = self.http.post(format!("{}/v1/chat/completions", self.base));
        match &self.api_key {
            Some(k) => req.bearer_auth(k),
            None => req,
        }
    }

    pub async fn chat(&self, system: &str, user: &str) -> Result<String> {
        self.chat_with_history(system, &[], user).await
    }

    /// Chat with prior turns (`(role, content)`) placed between the system
    /// prompt and the final user message.
    pub async fn chat_with_history(
        &self,
        system: &str,
        history: &[(String, String)],
        user: &str,
    ) -> Result<String> {
        let mut messages = vec![ChatMessage { role: "system", content: system }];
        for (role, content) in history {
            messages.push(ChatMessage { role: role.as_str(), content: content.as_str() });
        }
        messages.push(ChatMessage { role: "user", content: user });
        let body = ChatBody {
            model: &self.model,
            messages,
            temperature: 0.1,
            stream: false,
            response_format: None,
        };
        let resp: ChatCompletion = self
            .endpoint()
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
            .endpoint()
            .json(&body)
            .send().await?.error_for_status()?
            .json().await?;
        Ok(resp.choices.into_iter().next()
            .map(|c| c.message.content)
            .unwrap_or_default())
    }

    /// Structured output for servers that may not implement `json_schema`
    /// (llama.cpp, Ollama, LM Studio, …). Tries the strict path first; on any
    /// failure it re-asks in plain chat mode with the schema pasted into the
    /// instructions and extracts the first JSON object from the reply. Returns a
    /// parsed value so callers never see the transport difference.
    pub async fn complete_json_lenient(
        &self,
        system: &str,
        user: &str,
        schema_name: &str,
        schema: serde_json::Value,
    ) -> Result<serde_json::Value> {
        if let Ok(raw) = self.complete_json(system, user, schema_name, schema.clone()).await {
            if let Some(v) = first_json_object(&raw) {
                return Ok(v);
            }
        }
        let system = format!(
            "{system}\n\nRespond with ONLY a single JSON object (no prose, no code fence) \
             that conforms to this JSON schema:\n{}",
            serde_json::to_string(&schema)?
        );
        let raw = self.chat(&system, user).await?;
        first_json_object(&raw)
            .ok_or_else(|| anyhow::anyhow!("model did not return a JSON object: {}", raw.chars().take(200).collect::<String>()))
    }

    /// Model ids advertised by `/v1/models`, for the settings UI.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let req = self.http.get(format!("{}/v1/models", self.base));
        let req = match &self.api_key {
            Some(k) => req.bearer_auth(k),
            None => req,
        };
        let v: serde_json::Value = req.send().await?.error_for_status()?.json().await?;
        Ok(v["data"]
            .as_array()
            .map(|a| a.iter().filter_map(|m| m["id"].as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default())
    }

    pub fn base(&self) -> &str { &self.base }
    pub fn model(&self) -> &str { &self.model }
    pub fn has_api_key(&self) -> bool { self.api_key.is_some() }
}

/// Locate and parse the first balanced `{ … }` object in `text`, tolerating
/// leading prose and markdown fences. Returns `None` when nothing parses.
pub fn first_json_object(text: &str) -> Option<serde_json::Value> {
    let bytes = text.as_bytes();
    let mut start = None;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            if escape { escape = false; }
            else if b == b'\\' { escape = true; }
            else if b == b'"' { in_str = false; }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => { if depth == 0 { start = Some(i); } depth += 1; }
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text[s..=i]) {
                            return Some(v);
                        }
                        start = None;
                    }
                }
                if depth < 0 { depth = 0; }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_json_from_prose_and_fences() {
        let v = first_json_object("Sure! ```json\n{\"mode\": \"edit\", \"n\": [1, {\"a\": \"}\"}]}\n```").unwrap();
        assert_eq!(v["mode"], "edit");
        assert_eq!(v["n"][1]["a"], "}");
        assert!(first_json_object("no json here").is_none());
    }
}