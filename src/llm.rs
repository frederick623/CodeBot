use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub const PLAN_TIMEOUT_SECS: u64 = 45;
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

/// Report progress without spawning a timer task. Dropping this future also
/// drops the operation and its heartbeat timer.
pub(crate) async fn with_progress<T>(
    stage: &str,
    emit: &(dyn Fn(String) + Send + Sync),
    future: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    with_progress_interval(stage, emit, future, PROGRESS_INTERVAL).await
}

async fn with_progress_interval<T>(
    stage: &str,
    emit: &(dyn Fn(String) + Send + Sync),
    future: impl std::future::Future<Output = Result<T>>,
    period: Duration,
) -> Result<T> {
    let started = tokio::time::Instant::now();
    emit(format!("{stage}: starting"));
    let mut ticks = tokio::time::interval_at(started + period, period);
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    tokio::pin!(future);
    loop {
        tokio::select! {
            biased;
            result = &mut future => {
                let outcome = if result.is_ok() { "completed" } else { "failed" };
                emit(format!("{stage}: {outcome} ({}s elapsed)", started.elapsed().as_secs()));
                return result;
            }
            _ = ticks.tick() => {
                emit(format!("{stage}: waiting ({}s elapsed)", started.elapsed().as_secs()));
            }
        }
    }
}

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

    /// Deadline-bounded planning. Only an explicit unsupported-option HTTP
    /// 400/422 permits one plain-chat fallback; invalid output is never retried.
    pub async fn complete_plan_json(
        &self,
        system: &str,
        user: &str,
        schema: serde_json::Value,
        emit: &(dyn Fn(String) + Send + Sync),
    ) -> Result<serde_json::Value> {
        self.complete_plan_json_with_timeout(
            system, user, schema, emit, Duration::from_secs(PLAN_TIMEOUT_SECS),
        ).await
    }

    // Injectable deadline keeps timeout tests fast without Tokio's test-util
    // feature. There is one timeout around the entire operation, not each POST.
    async fn complete_plan_json_with_timeout(
        &self,
        system: &str,
        user: &str,
        schema: serde_json::Value,
        emit: &(dyn Fn(String) + Send + Sync),
        deadline: Duration,
    ) -> Result<serde_json::Value> {
        let stage = format!(
            "Planning (model: {}, system: {} bytes, user: {} bytes)",
            self.model, system.len(), user.len(),
        );
        with_progress(&stage, emit, async {
            tokio::time::timeout(deadline, self.plan_attempts(system, user, schema, emit))
                .await
                .map_err(|_| anyhow::anyhow!(
                    "planner exceeded total deadline of {}s", deadline.as_secs_f64(),
                ))?
        }).await
    }

    async fn plan_attempts(
        &self,
        system: &str,
        user: &str,
        schema: serde_json::Value,
        emit: &(dyn Fn(String) + Send + Sync),
    ) -> Result<serde_json::Value> {
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "stream": false,
            "max_tokens": 1024,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "plan", "strict": true, "schema": schema},
            },
        });
        if self.model.to_lowercase().contains("gpt-oss") {
            body["reasoning_effort"] = serde_json::json!("low");
        }

        for attempt in 0..2 {
            let response = self.endpoint().json(&body).send().await.map_err(|err| {
                // Do not surface request URLs, credentials, prompts or bodies.
                if err.is_timeout() {
                    anyhow::anyhow!("planner request timed out")
                } else {
                    anyhow::anyhow!("planner request failed (network or transport error)")
                }
            })?;
            let status = response.status();
            let bytes = response.bytes().await
                .map_err(|_| anyhow::anyhow!("failed to read planner response"))?;
            if !status.is_success() {
                if attempt == 0
                    && matches!(status.as_u16(), 400 | 422)
                    && unsupported_plan_option(&String::from_utf8_lossy(&bytes))
                {
                    emit("Planning: unsupported structured-output/reasoning option; retrying once with plain-chat fallback (same total deadline)".to_owned());
                    let fields = body.as_object_mut().expect("planner request is an object");
                    fields.remove("response_format");
                    fields.remove("reasoning_effort");
                    body["messages"][0]["content"] = serde_json::json!(format!(
                        "{system}\n\nRespond with ONLY a single JSON object (no prose, no code fence) \
                         that conforms to this JSON schema:\n{schema}"
                    ));
                    continue;
                }
                anyhow::bail!("planner request failed with HTTP {}", status.as_u16());
            }
            return parse_plan_response(&bytes);
        }
        unreachable!("the second attempt always returns")
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

/// Match an explicit unsupported-option statement, not merely an option name
/// and an unrelated error somewhere else in the response body.
fn unsupported_plan_option(body: &str) -> bool {
    let normalized: String = body.to_lowercase().chars().map(|c| {
        if c.is_alphanumeric() || c == '_' { c } else { ' ' }
    }).collect();
    let text = format!(" {} ", normalized.split_whitespace().collect::<Vec<_>>().join(" "));
    ["response_format", "json_schema", "reasoning_effort"].iter().any(|field| {
        [
            format!("{field} is not supported"),
            format!("{field} not supported"),
            format!("{field} is unsupported"),
            format!("{field} is not implemented"),
            format!("unsupported {field}"),
            format!("unsupported parameter {field}"),
            format!("unsupported field {field}"),
            format!("unknown parameter {field}"),
            format!("unknown field {field}"),
            format!("unrecognized parameter {field}"),
            format!("unrecognized request argument supplied {field}"),
            format!("does not support {field}"),
            format!("doesn t support {field}"),
        ].iter().any(|phrase| text.contains(&format!(" {phrase} ")))
    })
}

fn parse_plan_response(bytes: &[u8]) -> Result<serde_json::Value> {
    let envelope: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|_| anyhow::anyhow!("planner returned a malformed chat completion response"))?;
    let choice = envelope.get("choices").and_then(|v| v.as_array()).and_then(|v| v.first())
        .ok_or_else(|| anyhow::anyhow!("planner response is missing a completion choice"))?;
    if choice.get("finish_reason").and_then(|v| v.as_str()) == Some("length") {
        anyhow::bail!("planner response was truncated (finish_reason=length)");
    }
    let content = choice.get("message").and_then(|v| v.get("content"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("planner response content is missing, null, or not a string"))?;
    if content.trim().is_empty() {
        anyhow::bail!("planner response content is empty");
    }
    let plan: serde_json::Value = serde_json::from_str(content)
        .map_err(|_| anyhow::anyhow!("planner response content is not valid JSON"))?;
    if !plan.is_object() {
        anyhow::bail!("planner response must be a JSON object");
    }
    Ok(plan)
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
    use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::post, Json, Router};
    use serde_json::{json, Value};
    use std::{collections::VecDeque, sync::{Arc, Mutex}};

    struct Reply {
        status: StatusCode,
        body: String,
        delay: Duration,
    }

    impl Reply {
        fn json(status: u16, body: Value) -> Self {
            Self {
                status: StatusCode::from_u16(status).unwrap(),
                body: body.to_string(),
                delay: Duration::ZERO,
            }
        }

        fn delayed(mut self, millis: u64) -> Self {
            self.delay = Duration::from_millis(millis);
            self
        }
    }

    #[derive(Clone)]
    struct MockState {
        requests: Arc<Mutex<Vec<Value>>>,
        replies: Arc<Mutex<VecDeque<Reply>>>,
    }

    struct MockServer {
        base: String,
        state: MockState,
        task: tokio::task::JoinHandle<()>,
    }

    impl Drop for MockServer {
        fn drop(&mut self) { self.task.abort(); }
    }

    impl MockServer {
        async fn start(replies: Vec<Reply>) -> Self {
            async fn handle(State(state): State<MockState>, Json(body): Json<Value>) -> impl IntoResponse {
                state.requests.lock().unwrap().push(body);
                let reply = state.replies.lock().unwrap().pop_front()
                    .unwrap_or_else(|| Reply::json(500, json!({"error": "unexpected extra request"})));
                if !reply.delay.is_zero() { tokio::time::sleep(reply.delay).await; }
                (reply.status, [("content-type", "application/json")], reply.body)
            }
            let state = MockState {
                requests: Arc::new(Mutex::new(Vec::new())),
                replies: Arc::new(Mutex::new(replies.into())),
            };
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            let app = Router::new().route("/v1/chat/completions", post(handle)).with_state(state.clone());
            let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
            Self { base, state, task }
        }

        fn client(&self, model: &str) -> LlmClient {
            let mut client = LlmClient::new(&self.base, model);
            client.http = reqwest::Client::builder().no_proxy()
                .timeout(Duration::from_secs(600)).build().unwrap();
            client
        }

        fn requests(&self) -> Vec<Value> { self.state.requests.lock().unwrap().clone() }
    }

    fn schema() -> Value {
        json!({"type": "object", "properties": {"mode": {"type": "string"}},
            "required": ["mode"], "additionalProperties": false})
    }

    fn success() -> Reply {
        Reply::json(200, json!({"choices": [{"message": {"content": "{\"mode\":\"ask\"}"}, "finish_reason": "stop"}]}))
    }

    fn unsupported(status: u16, option: &str) -> Reply {
        Reply::json(status, json!({"error": {"message": format!("Unsupported parameter: '{option}'")}}))
    }

    #[tokio::test]
    async fn planner_strict_caps_and_optional_reasoning() {
        assert_eq!(PLAN_TIMEOUT_SECS, 45);
        for (model, reasoning) in [("vendor/GPT-OSS-20b", true), ("other-model", false)] {
            let server = MockServer::start(vec![success()]).await;
            let events = Mutex::new(Vec::new());
            let emit = |msg| events.lock().unwrap().push(msg);
            let value = server.client(model).complete_plan_json("secret-system", "secret-user", schema(), &emit).await.unwrap();
            assert_eq!(value, json!({"mode": "ask"}));
            let requests = server.requests();
            assert_eq!(requests.len(), 1);
            let body = &requests[0];
            assert_eq!(body["model"], model);
            assert_eq!(body["max_tokens"], 1024);
            assert_eq!(body["stream"], false);
            assert_eq!(body["temperature"], 0.0);
            assert_eq!(body["response_format"], json!({"type": "json_schema", "json_schema": {
                "name": "plan", "strict": true, "schema": schema(),
            }}));
            assert_eq!(body.get("reasoning_effort").and_then(Value::as_str), if reasoning { Some("low") } else { None });
            assert_eq!(body["messages"], json!([
                {"role": "system", "content": "secret-system"},
                {"role": "user", "content": "secret-user"},
            ]));
            let events = events.lock().unwrap();
            assert!(events[0].contains(model));
            assert!(events[0].contains("system: 13 bytes, user: 11 bytes"));
            assert!(events[0].ends_with("starting"));
            assert!(events.last().unwrap().contains("completed (0s elapsed)"));
            assert!(!events.iter().any(|s| s.contains("secret-")));
        }
    }

    #[tokio::test]
    async fn planner_fallback_is_bounded_and_visible() {
        for status in [400, 422] {
            for option in ["response_format", "json_schema", "reasoning_effort"] {
                let server = MockServer::start(vec![unsupported(status, option), success()]).await;
                let events = Mutex::new(Vec::new());
                let emit = |msg| events.lock().unwrap().push(msg);
                server.client("gpt-oss").complete_plan_json("system", "user", schema(), &emit).await.unwrap();
                let requests = server.requests();
                assert_eq!(requests.len(), 2);
                assert_eq!(requests[0]["reasoning_effort"], "low");
                let fallback = &requests[1];
                assert!(fallback.get("response_format").is_none());
                assert!(fallback.get("reasoning_effort").is_none());
                assert_eq!(fallback["max_tokens"], 1024);
                assert_eq!(fallback["stream"], false);
                assert_eq!(fallback["temperature"], 0.0);
                assert_eq!(fallback["messages"][1]["content"], "user");
                let system = fallback["messages"][0]["content"].as_str().unwrap();
                assert!(system.starts_with("system\n"));
                assert!(system.contains(&schema().to_string()));
                assert!(events.lock().unwrap().iter().any(|s| s.contains("retrying once") && s.contains("fallback")));
            }
        }
        let server = MockServer::start(vec![unsupported(400, "response_format"), unsupported(422, "json_schema")]).await;
        assert!(server.client("gpt-oss").complete_plan_json("s", "u", schema(), &|_| {}).await.is_err());
        assert_eq!(server.requests().len(), 2);
    }

    #[tokio::test]
    async fn planner_does_not_retry_other_http_errors() {
        let cases = [
            (401, "unsupported response_format"), (403, "unsupported response_format"),
            (408, "unsupported response_format"), (429, "unsupported response_format"),
            (500, "unsupported response_format"), (503, "unsupported response_format"),
            (400, "invalid schema for response_format"),
            (422, "response_format schema validation failed"),
            (400, "unsupported model; response_format was provided"),
            (422, "response_format is supported; authentication failed"),
        ];
        for (status, message) in cases {
            let server = MockServer::start(vec![Reply::json(status, json!({"error": {"message": message}}))]).await;
            let events = Mutex::new(Vec::new());
            let emit = |msg| events.lock().unwrap().push(msg);
            let err = server.client("gpt-oss").complete_plan_json("s", "u", schema(), &emit).await.unwrap_err();
            assert!(err.to_string().contains(&format!("HTTP {status}")));
            assert_eq!(server.requests().len(), 1);
            assert!(!events.lock().unwrap().iter().any(|s| s.contains("fallback")));
        }
    }

    #[tokio::test]
    async fn planner_network_errors_do_not_fallback_or_leak_secrets() {
        let server = MockServer::start(vec![]).await;
        let mut client = server.client("model").with_api_key(Some("secret-token".to_owned()));
        // Reserve a local port, then close it to produce a connection refusal.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        client.base = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let events = Mutex::new(Vec::new());
        let emit = |msg| events.lock().unwrap().push(msg);
        let err = client.complete_plan_json("secret-system", "secret-user", schema(), &emit).await.unwrap_err();
        assert!(err.to_string().contains("network or transport"));
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 2);
        assert!(events[1].contains("failed"));
        assert!(!format!("{err:?}{events:?}").contains("secret-"));
    }

    #[tokio::test]
    async fn planner_invalid_replies_fail_without_retry() {
        let cases = [
            (json!({}), "missing a completion choice"),
            (json!({"choices": []}), "missing a completion choice"),
            (json!({"choices": [{"message": {}}]}), "missing, null"),
            (json!({"choices": [{"message": {"content": null}}]}), "missing, null"),
            (json!({"choices": [{"message": {"content": 42}}]}), "not a string"),
            (json!({"choices": [{"message": {"content": ""}}]}), "empty"),
            (json!({"choices": [{"message": {"content": " \n "}}]}), "empty"),
            (json!({"choices": [{"message": {"content": "{}"}, "finish_reason": "length"}]}), "truncated"),
            (json!({"choices": [{"message": {"content": null}, "finish_reason": "length"}]}), "truncated"),
            (json!({"choices": [{"message": {"content": "{secret-invalid"}}]}), "not valid JSON"),
            (json!({"choices": [{"message": {"content": "```json\n{}\n```"}}]}), "not valid JSON"),
            (json!({"choices": [{"message": {"content": "null"}}]}), "must be a JSON object"),
            (json!({"choices": [{"message": {"content": "[]"}}]}), "must be a JSON object"),
        ];
        for (body, expected) in cases {
            let server = MockServer::start(vec![Reply::json(200, body)]).await;
            let err = server.client("gpt-oss").complete_plan_json("s", "u", schema(), &|_| {}).await.unwrap_err();
            assert!(err.to_string().contains(expected), "{err}");
            assert!(!err.to_string().contains("secret-invalid"));
            assert_eq!(server.requests().len(), 1);
        }
        let server = MockServer::start(vec![Reply { status: StatusCode::OK, body: "not-json".into(), delay: Duration::ZERO }]).await;
        let err = server.client("model").complete_plan_json("s", "u", schema(), &|_| {}).await.unwrap_err();
        assert!(err.to_string().contains("malformed chat completion"));
        assert_eq!(server.requests().len(), 1);
        let server = MockServer::start(vec![unsupported(400, "response_format"), Reply::json(200, json!({}))]).await;
        assert!(server.client("model").complete_plan_json("s", "u", schema(), &|_| {}).await.is_err());
        assert_eq!(server.requests().len(), 2);
    }

    #[tokio::test]
    async fn planner_deadline_covers_both_attempts() {
        // Each request fits independently in 150ms, but together they do not.
        let server = MockServer::start(vec![
            unsupported(400, "response_format").delayed(60), success().delayed(120),
        ]).await;
        let events = Mutex::new(Vec::new());
        let emit = |msg| events.lock().unwrap().push(msg);
        let err = server.client("gpt-oss").complete_plan_json_with_timeout(
            "s", "u", schema(), &emit, Duration::from_millis(150),
        ).await.unwrap_err();
        assert!(err.to_string().contains("total deadline"));
        assert_eq!(server.requests().len(), 2);
        assert!(events.lock().unwrap().last().unwrap().contains("failed"));

        let server = MockServer::start(vec![unsupported(400, "response_format").delayed(100)]).await;
        let events = Mutex::new(Vec::new());
        let emit = |msg| events.lock().unwrap().push(msg);
        let err = server.client("model").complete_plan_json_with_timeout(
            "s", "u", schema(), &emit, Duration::from_millis(30),
        ).await.unwrap_err();
        assert!(err.to_string().contains("total deadline"));
        assert_eq!(server.requests().len(), 1);
        assert!(!events.lock().unwrap().iter().any(|s| s.contains("fallback")));
    }

    #[tokio::test]
    async fn progress_starts_waits_and_finishes_without_immediate_tick() {
        assert_eq!(PROGRESS_INTERVAL, Duration::from_secs(5));
        for fail in [false, true] {
            let events = Mutex::new(Vec::new());
            let done = tokio::sync::Notify::new();
            let started = tokio::time::Instant::now();
            let emit = |msg: String| {
                let mut events = events.lock().unwrap();
                events.push((started.elapsed(), msg));
                if events.iter().filter(|(_, s)| s.contains("waiting")).count() == 2 {
                    done.notify_one();
                }
            };
            let result = with_progress_interval("Test", &emit, async {
                done.notified().await;
                if fail { anyhow::bail!("test failure"); }
                Ok(42)
            }, Duration::from_millis(5)).await;
            assert_eq!(result.is_err(), fail);
            let events = events.lock().unwrap();
            assert_eq!(events.len(), 4);
            assert_eq!(events[0].1, "Test: starting");
            assert!(events[1].0 >= Duration::from_millis(5));
            assert!(events[1].1.contains("waiting (0s elapsed)"));
            assert!(events[3].1.contains(if fail { "failed" } else { "completed" }));
        }
        let events = Mutex::new(Vec::new());
        let emit = |msg| events.lock().unwrap().push(msg);
        assert_eq!(with_progress("Immediate", &emit, async { Ok(7) }).await.unwrap(), 7);
        assert_eq!(events.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn progress_cancellation_drops_operation() {
        struct DropFlag(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) { self.0.store(true, std::sync::atomic::Ordering::SeqCst); }
        }
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let guard = DropFlag(dropped.clone());
        let events = Mutex::new(Vec::new());
        let emit = |msg| events.lock().unwrap().push(msg);
        let result = tokio::time::timeout(Duration::from_millis(10), with_progress("Cancelled", &emit, async move {
            let _guard = guard;
            std::future::pending::<Result<()>>().await
        })).await;
        assert!(result.is_err());
        assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(events.lock().unwrap().as_slice(), ["Cancelled: starting"]);
    }

    #[test]
    fn extracts_json_from_prose_and_fences() {
        let v = first_json_object("Sure! ```json\n{\"mode\": \"edit\", \"n\": [1, {\"a\": \"}\"}]}\n```").unwrap();
        assert_eq!(v["mode"], "edit");
        assert_eq!(v["n"][1]["a"], "}");
        assert!(first_json_object("no json here").is_none());
    }
}