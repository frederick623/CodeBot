//! Standalone GUI for CodeBot — a prompt-driven IDE in the spirit of
//! OpenCode / Claude Code, backed by a local LLM.
//!
//! The GUI talks to the daemon purely over HTTP. On startup it probes
//! `/health`; if no server is listening it brings one up *in-process* on a
//! background Tokio runtime, then keeps using the same HTTP API. All blocking
//! work (HTTP, model loading) happens off the UI thread; results flow back to
//! egui through a channel.
//!
//! Layout: top bar (server/index status, workspace, mode, settings) · central
//! chat transcript with a prompt box (Enter sends, Ctrl+Enter inserts a
//! newline) · right "Changes" pane listing the working-tree diff vs git with
//! per-file Revert · bottom log.
mod markdown;

use codebot_daemon::agent::{AgentResult, Mode};
use codebot_daemon::api::IndexStatus;
use codebot_daemon::server::{self, RunState, Settings};
use codebot_daemon::tools::GitChange;
use eframe::egui;
use std::sync::mpsc as schan;
use std::time::{Duration, Instant};
use tokio::sync::mpsc as tchan;

/// Base URL derived from the server's bind address.
fn base() -> String {
    format!("http://{}", server::DEFAULT_ADDR)
}

/// Commands the UI thread sends to the async worker.
enum Cmd {
    Open(String),
    Run { workspace: String, prompt: String, mode: Mode },
    Status(String),
    Changes(String),
    Revert { workspace: String, file: String },
    History(String),
    ClearHistory(String),
    GetSettings,
    SetSettings(Settings),
    ListModels,
}

/// Events the worker sends back to the UI thread.
enum Ui {
    Health(bool),
    Log(String),
    Status(IndexStatus),
    Step(String),
    RunDone(Result<AgentResult, String>),
    Changes(Vec<GitChange>),
    History(Vec<(String, String)>),
    Settings(Settings),
    Models(Vec<String>),
    Error(String),
}

#[derive(Clone)]
struct Turn {
    role: String,
    content: String,
    edited: Vec<String>,
    failed: Vec<String>,
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 820.0]),
        ..Default::default()
    };
    eframe::run_native("CodeBot", options, Box::new(|cc| Ok(Box::new(GuiApp::new(cc)))))
}

struct GuiApp {
    cmd_tx: tchan::UnboundedSender<Cmd>,
    ui_rx: schan::Receiver<Ui>,
    workspace: String,
    opened: String,
    prompt: String,
    mode: Mode,
    turns: Vec<Turn>,
    steps: Vec<String>,
    changes: Vec<GitChange>,
    expanded: Option<String>,
    status: Option<IndexStatus>,
    health: Option<bool>,
    logs: Vec<String>,
    busy: bool,
    last_changes_poll: Instant,
    show_settings: bool,
    settings: Settings,
    api_key_input: String,
    models: Vec<String>,
    scroll_to_bottom: bool,
}

impl GuiApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let ctx = cc.egui_ctx.clone();
        let (cmd_tx, cmd_rx) = tchan::unbounded_channel::<Cmd>();
        let (ui_tx, ui_rx) = schan::channel::<Ui>();

        // Dedicated multi-threaded runtime so the in-process server and the
        // command loop can run concurrently, fully off the UI thread.
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("tokio runtime");
            rt.block_on(worker(cmd_rx, ui_tx, ctx));
        });

        let cwd = std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_default();
        Self {
            cmd_tx,
            ui_rx,
            workspace: cwd.clone(),
            opened: cwd,
            prompt: String::new(),
            mode: Mode::Agent,
            turns: Vec::new(),
            steps: Vec::new(),
            changes: Vec::new(),
            expanded: None,
            status: None,
            health: None,
            logs: Vec::new(),
            busy: false,
            last_changes_poll: Instant::now(),
            show_settings: false,
            settings: Settings::default(),
            api_key_input: String::new(),
            models: Vec::new(),
            scroll_to_bottom: false,
        }
    }

    fn send(&self, cmd: Cmd) {
        let _ = self.cmd_tx.send(cmd);
    }

    fn submit(&mut self) {
        let prompt = self.prompt.trim().to_string();
        if prompt.is_empty() || self.busy || self.health != Some(true) {
            return;
        }
        self.busy = true;
        self.steps.clear();
        self.turns.push(Turn { role: "user".into(), content: prompt.clone(), edited: vec![], failed: vec![] });
        self.prompt.clear();
        self.scroll_to_bottom = true;
        self.send(Cmd::Run { workspace: self.opened.clone(), prompt, mode: self.mode });
    }
}

// ----- async worker ----------------------------------------------------------

/// Async backend: ensures a server is reachable, then services UI commands.
async fn worker(mut cmd_rx: tchan::UnboundedReceiver<Cmd>, ui_tx: schan::Sender<Ui>, ctx: egui::Context) {
    let client = reqwest::Client::new();
    if ensure_server(&client, &ui_tx, &ctx).await {
        let default_ws = std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_default();
        if !default_ws.is_empty() {
            open_workspace(&client, &ui_tx, &ctx, &default_ws).await;
        }
        fetch_settings(&client, &ui_tx, &ctx).await;
    }

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            Cmd::Open(path) => open_workspace(&client, &ui_tx, &ctx, &path).await,
            Cmd::Run { workspace, prompt, mode } => {
                let r = client
                    .post(format!("{}/agent/run", base()))
                    .json(&serde_json::json!({ "workspace_path": workspace, "prompt": prompt, "mode": mode }))
                    .send()
                    .await;
                match parse::<serde_json::Value>(r).await {
                    Ok(v) => {
                        let run_id = v["run_id"].as_str().unwrap_or_default().to_string();
                        let (client, tx, ctx) = (client.clone(), ui_tx.clone(), ctx.clone());
                        tokio::spawn(async move { watch_run(&client, &tx, &ctx, &run_id, &workspace).await });
                    }
                    Err(e) => emit(&ui_tx, &ctx, Ui::RunDone(Err(format!("run failed: {e}")))),
                }
            }
            Cmd::Status(ws) => fetch_status(&client, &ui_tx, &ctx, &ws).await,
            Cmd::Changes(ws) => fetch_changes(&client, &ui_tx, &ctx, &ws).await,
            Cmd::Revert { workspace, file } => {
                let r = client
                    .post(format!("{}/git/revert", base()))
                    .json(&serde_json::json!({ "workspace_path": workspace, "file": file }))
                    .send()
                    .await;
                match parse::<serde_json::Value>(r).await {
                    Ok(v) if v["ok"].as_bool() == Some(true) => emit(&ui_tx, &ctx, Ui::Log(format!("reverted {file}"))),
                    Ok(v) => emit(&ui_tx, &ctx, Ui::Error(format!("revert {file}: {}", v["error"]))),
                    Err(e) => emit(&ui_tx, &ctx, Ui::Error(format!("revert failed: {e}"))),
                }
                fetch_changes(&client, &ui_tx, &ctx, &workspace).await;
            }
            Cmd::History(ws) => {
                let r = client
                    .get(format!("{}/history", base()))
                    .query(&[("workspace_path", ws)])
                    .send()
                    .await;
                if let Ok(v) = parse::<Vec<serde_json::Value>>(r).await {
                    let turns = v
                        .into_iter()
                        .map(|t| {
                            (
                                t["role"].as_str().unwrap_or("assistant").to_string(),
                                t["content"].as_str().unwrap_or_default().to_string(),
                            )
                        })
                        .collect();
                    emit(&ui_tx, &ctx, Ui::History(turns));
                }
            }
            Cmd::ClearHistory(ws) => {
                let _ = client
                    .delete(format!("{}/history", base()))
                    .query(&[("workspace_path", ws)])
                    .send()
                    .await;
                emit(&ui_tx, &ctx, Ui::History(vec![]));
                emit(&ui_tx, &ctx, Ui::Log("conversation cleared".into()));
            }
            Cmd::GetSettings => fetch_settings(&client, &ui_tx, &ctx).await,
            Cmd::SetSettings(s) => {
                let r = client.post(format!("{}/settings", base())).json(&s).send().await;
                match parse::<Settings>(r).await {
                    Ok(s) => {
                        emit(&ui_tx, &ctx, Ui::Log(format!("LLM: {} @ {}", s.llm_model, s.llm_base)));
                        emit(&ui_tx, &ctx, Ui::Settings(s));
                    }
                    Err(e) => emit(&ui_tx, &ctx, Ui::Error(format!("settings failed: {e}"))),
                }
            }
            Cmd::ListModels => {
                let r = client.get(format!("{}/models", base())).send().await;
                match parse::<serde_json::Value>(r).await {
                    Ok(v) => {
                        let models = v["models"]
                            .as_array()
                            .map(|a| a.iter().filter_map(|m| m.as_str().map(String::from)).collect())
                            .unwrap_or_default();
                        if let Some(e) = v["error"].as_str() {
                            emit(&ui_tx, &ctx, Ui::Log(format!("models: {e}")));
                        }
                        emit(&ui_tx, &ctx, Ui::Models(models));
                    }
                    Err(e) => emit(&ui_tx, &ctx, Ui::Error(format!("models failed: {e}"))),
                }
            }
        }
    }
}

/// Push an event to the UI thread and wake the egui repaint loop.
fn emit(tx: &schan::Sender<Ui>, ctx: &egui::Context, ev: Ui) {
    let _ = tx.send(ev);
    ctx.request_repaint();
}

/// `true` when `/health` answers successfully.
async fn health(client: &reqwest::Client) -> bool {
    matches!(client.get(format!("{}/health", base())).send().await, Ok(r) if r.status().is_success())
}

/// Connect to an existing server, or start one in-process and wait for it.
async fn ensure_server(client: &reqwest::Client, tx: &schan::Sender<Ui>, ctx: &egui::Context) -> bool {
    if health(client).await {
        emit(tx, ctx, Ui::Log("connected to running server".into()));
        emit(tx, ctx, Ui::Health(true));
        return true;
    }
    emit(tx, ctx, Ui::Log("no server found; starting in-process…".into()));
    tokio::spawn(async {
        if let Err(e) = server::run().await {
            eprintln!("in-process server exited: {e}");
        }
    });
    for _ in 0..600 {
        tokio::time::sleep(Duration::from_millis(500)).await;
        if health(client).await {
            emit(tx, ctx, Ui::Log("server ready".into()));
            emit(tx, ctx, Ui::Health(true));
            return true;
        }
    }
    emit(tx, ctx, Ui::Health(false));
    emit(tx, ctx, Ui::Error("server failed to start".into()));
    false
}

/// Open a workspace (triggers a background index) and watch its progress.
async fn open_workspace(client: &reqwest::Client, tx: &schan::Sender<Ui>, ctx: &egui::Context, path: &str) {
    let r = client
        .post(format!("{}/workspace/open", base()))
        .json(&serde_json::json!({ "workspace_path": path }))
        .send()
        .await;
    if let Err(e) = r {
        emit(tx, ctx, Ui::Error(format!("open failed: {e}")));
        return;
    }
    emit(tx, ctx, Ui::Log(format!("opened workspace: {path}")));
    fetch_changes(client, tx, ctx, path).await;

    let (client, tx, ctx, path) = (client.clone(), tx.clone(), ctx.clone(), path.to_string());
    tokio::spawn(async move { watch_indexing(&client, &tx, &ctx, &path).await });
}

async fn fetch_status(client: &reqwest::Client, tx: &schan::Sender<Ui>, ctx: &egui::Context, ws: &str) {
    let r = client
        .get(format!("{}/index/status", base()))
        .query(&[("workspace_path", ws)])
        .send()
        .await;
    match parse::<IndexStatus>(r).await {
        Ok(s) => emit(tx, ctx, Ui::Status(s)),
        Err(e) => emit(tx, ctx, Ui::Error(format!("status failed: {e}"))),
    }
}

async fn fetch_changes(client: &reqwest::Client, tx: &schan::Sender<Ui>, ctx: &egui::Context, ws: &str) {
    let r = client
        .get(format!("{}/git/changes", base()))
        .query(&[("workspace_path", ws)])
        .send()
        .await;
    if let Ok(c) = parse::<Vec<GitChange>>(r).await {
        emit(tx, ctx, Ui::Changes(c));
    }
}

async fn fetch_settings(client: &reqwest::Client, tx: &schan::Sender<Ui>, ctx: &egui::Context) {
    let r = client.get(format!("{}/settings", base())).send().await;
    if let Ok(s) = parse::<Settings>(r).await {
        emit(tx, ctx, Ui::Log(format!("LLM: {} @ {}", s.llm_model, s.llm_base)));
        emit(tx, ctx, Ui::Settings(s));
    }
}

/// Poll `/agent/status` until the run completes, streaming new steps to the UI.
async fn watch_run(client: &reqwest::Client, tx: &schan::Sender<Ui>, ctx: &egui::Context, run_id: &str, ws: &str) {
    let mut seen = 0usize;
    loop {
        tokio::time::sleep(Duration::from_millis(400)).await;
        let r = client
            .get(format!("{}/agent/status", base()))
            .query(&[("run_id", run_id)])
            .send()
            .await;
        let st = match parse::<RunState>(r).await {
            Ok(s) => s,
            Err(e) => {
                emit(tx, ctx, Ui::RunDone(Err(format!("status failed: {e}"))));
                return;
            }
        };
        for s in st.steps.iter().skip(seen) {
            emit(tx, ctx, Ui::Step(s.clone()));
        }
        seen = st.steps.len();
        if st.done {
            let out = match (st.result, st.error) {
                (Some(r), _) => Ok(r),
                (None, Some(e)) => Err(e),
                (None, None) => Err("run produced no result".into()),
            };
            emit(tx, ctx, Ui::RunDone(out));
            fetch_changes(client, tx, ctx, ws).await;
            fetch_status(client, tx, ctx, ws).await;
            return;
        }
    }
}

/// Poll `/index/status` until the counts stabilise, logging progress.
async fn watch_indexing(client: &reqwest::Client, tx: &schan::Sender<Ui>, ctx: &egui::Context, ws: &str) {
    emit(tx, ctx, Ui::Log("indexing started…".into()));
    let mut last: Option<(usize, usize)> = None;
    let mut stable = 0u32;
    for _ in 0..600 {
        tokio::time::sleep(Duration::from_millis(500)).await;
        let r = client
            .get(format!("{}/index/status", base()))
            .query(&[("workspace_path", ws)])
            .send()
            .await;
        let Ok(s) = parse::<IndexStatus>(r).await else { continue };
        let cur = (s.files, s.chunks);
        emit(tx, ctx, Ui::Status(s));
        if Some(cur) != last {
            last = Some(cur);
            stable = 0;
            continue;
        }
        stable += 1;
        if stable >= 4 {
            let (files, chunks) = cur;
            emit(tx, ctx, Ui::Log(format!("indexing complete: {files} files, {chunks} chunks")));
            return;
        }
    }
}

/// Decode a JSON response body, surfacing transport and decode errors uniformly.
async fn parse<T: serde::de::DeserializeOwned>(r: reqwest::Result<reqwest::Response>) -> Result<T, String> {
    r.map_err(|e| e.to_string())?.json::<T>().await.map_err(|e| e.to_string())
}

// ----- rendering helpers -------------------------------------------------------

/// Render a unified diff as coloured monospace lines.
fn diff_view(ui: &mut egui::Ui, diff: &str) {
    use egui::Color32;
    for line in diff.lines() {
        let color = if line.starts_with("+++") || line.starts_with("---") {
            Color32::from_rgb(120, 170, 255)
        } else if line.starts_with("@@") {
            Color32::from_rgb(190, 150, 230)
        } else if line.starts_with('+') {
            Color32::from_rgb(110, 200, 120)
        } else if line.starts_with('-') {
            Color32::from_rgb(225, 110, 110)
        } else {
            ui.visuals().text_color()
        };
        ui.add(
            egui::Label::new(egui::RichText::new(line).monospace().color(color))
                .wrap_mode(egui::TextWrapMode::Extend),
        );
    }
}

/// Badge letter and colour for a `GitChange::status` word.
fn status_badge(status: &str) -> (&'static str, egui::Color32) {
    match status {
        "added" => ("A", egui::Color32::from_rgb(110, 200, 120)),
        "deleted" => ("D", egui::Color32::from_rgb(225, 110, 110)),
        _ => ("M", egui::Color32::from_rgb(230, 180, 80)),
    }
}

impl eframe::App for GuiApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(ev) = self.ui_rx.try_recv() {
            match ev {
                Ui::Health(h) => self.health = Some(h),
                Ui::Log(m) => self.logs.push(m),
                Ui::Status(s) => self.status = Some(s),
                Ui::Error(e) => self.logs.push(format!("error: {e}")),
                Ui::Step(s) => {
                    self.steps.push(s.clone());
                    self.logs.push(s);
                }
                Ui::RunDone(res) => {
                    self.busy = false;
                    self.scroll_to_bottom = true;
                    match res {
                        Ok(r) => self.turns.push(Turn {
                            role: "assistant".into(),
                            content: r.answer,
                            edited: r.edited.iter().map(|e| e.path.clone()).collect(),
                            failed: r.failed,
                        }),
                        Err(e) => {
                            self.logs.push(format!("error: {e}"));
                            self.turns.push(Turn {
                                role: "assistant".into(),
                                content: format!("**Error:** {e}"),
                                edited: vec![],
                                failed: vec![],
                            });
                        }
                    }
                }
                Ui::Changes(c) => self.changes = c,
                Ui::History(h) => {
                    self.turns = h
                        .into_iter()
                        .map(|(role, content)| Turn { role, content, edited: vec![], failed: vec![] })
                        .collect();
                    self.scroll_to_bottom = true;
                }
                Ui::Settings(s) => self.settings = s,
                Ui::Models(m) => self.models = m,
            }
        }
        // Keep the Changes pane fresh even for edits made outside CodeBot.
        if self.health == Some(true) && self.last_changes_poll.elapsed() > Duration::from_secs(5) {
            self.last_changes_poll = Instant::now();
            self.send(Cmd::Changes(self.opened.clone()));
        }
        if self.busy {
            ctx.request_repaint_after(Duration::from_millis(250));
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.top_bar(ui);
        self.log_panel(ui);
        self.changes_panel(ui);
        self.chat_panel(ui);
        self.settings_window(ui.ctx());
    }
}

impl GuiApp {
    fn top_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("top").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                let dot = match self.health {
                    Some(true) => "🟢",
                    Some(false) => "🔴",
                    None => "🟡",
                };
                ui.label(dot);
                let counter = match &self.status {
                    Some(s) => format!("{} files · {} chunks", s.files, s.chunks),
                    None => "indexing…".to_string(),
                };
                if ui.button(counter).on_hover_text("Refresh index status").clicked() {
                    self.send(Cmd::Status(self.opened.clone()));
                }
                ui.separator();
                ui.label("Workspace:");
                let resp = ui.add(egui::TextEdit::singleline(&mut self.workspace).desired_width(360.0));
                let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                if ui.button("Open").clicked() || enter {
                    self.opened = self.workspace.trim().to_string();
                    self.status = None;
                    self.send(Cmd::Open(self.opened.clone()));
                    self.send(Cmd::History(self.opened.clone()));
                }
                ui.separator();
                ui.label("Mode:");
                ui.selectable_value(&mut self.mode, Mode::Agent, "Agent");
                ui.selectable_value(&mut self.mode, Mode::Ask, "Ask");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("⚙ Settings").clicked() {
                        self.show_settings = !self.show_settings;
                        if self.show_settings {
                            self.api_key_input.clear();
                            self.send(Cmd::GetSettings);
                            self.send(Cmd::ListModels);
                        }
                    }
                    if ui.button("🗑 Clear chat").clicked() {
                        self.send(Cmd::ClearHistory(self.opened.clone()));
                    }
                    ui.small(format!("{}", self.settings.llm_model));
                });
            });
        });
    }

    fn log_panel(&mut self, ui: &mut egui::Ui) {
        egui::Panel::bottom("logs").resizable(true).show_inside(ui, |ui| {
            ui.small("Log");
            egui::ScrollArea::vertical()
                .max_height(90.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for line in &self.logs {
                        ui.add(egui::Label::new(egui::RichText::new(line).monospace().small()).wrap());
                    }
                });
        });
    }

    fn changes_panel(&mut self, ui: &mut egui::Ui) {
        let mut revert: Option<String> = None;
        egui::Panel::right("changes")
            .resizable(true)
            .default_size(380.0)
            .min_size(220.0)
            .show_inside(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Changes");
                    ui.label(format!("({})", self.changes.len()));
                    if ui.small_button("⟳").clicked() {
                        self.send(Cmd::Changes(self.opened.clone()));
                    }
                });
                ui.separator();
                if self.changes.is_empty() {
                    ui.weak("Working tree clean.");
                    return;
                }
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for c in &self.changes {
                        let open = self.expanded.as_deref() == Some(c.path.as_str());
                        ui.horizontal(|ui| {
                            let (badge, colour) = status_badge(&c.status);
                            ui.colored_label(colour, egui::RichText::new(badge).strong().monospace())
                                .on_hover_text(&c.status);
                            let label = egui::RichText::new(&c.path).monospace();
                            if ui.selectable_label(open, label).clicked() {
                                self.expanded = if open { None } else { Some(c.path.clone()) };
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.small_button("Revert").clicked() {
                                    revert = Some(c.path.clone());
                                }
                                ui.colored_label(egui::Color32::from_rgb(225, 110, 110), format!("-{}", c.deletions));
                                ui.colored_label(egui::Color32::from_rgb(110, 200, 120), format!("+{}", c.additions));
                            });
                        });
                        if open {
                            egui::Frame::new()
                                .fill(ui.visuals().code_bg_color)
                                .inner_margin(4.0)
                                .show(ui, |ui| {
                                    egui::ScrollArea::horizontal().id_salt(&c.path).show(ui, |ui| {
                                        diff_view(ui, &c.diff);
                                    });
                                });
                        }
                    }
                });
            });
        if let Some(file) = revert {
            self.send(Cmd::Revert { workspace: self.opened.clone(), file });
        }
    }

    fn chat_panel(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            // Prompt box pinned to the bottom of the central area.
            egui::Panel::bottom("prompt").show_inside(ui, |ui| {
                ui.add_space(4.0);
                let hint = match self.mode {
                    Mode::Agent => "Ask or instruct… (Enter to send, Ctrl+Enter for newline)",
                    Mode::Ask => "Ask about the code… (Enter to send, Ctrl+Enter for newline)",
                };
                let newline = egui::KeyboardShortcut::new(egui::Modifiers::CTRL, egui::Key::Enter);
                let edit = egui::TextEdit::multiline(&mut self.prompt)
                    .desired_rows(3)
                    .desired_width(f32::INFINITY)
                    .hint_text(hint)
                    .return_key(Some(newline));
                let resp = ui.add(edit);
                // Plain Enter (no modifiers) submits; Ctrl+Enter is consumed by
                // the TextEdit as a newline via `return_key`.
                let plain_enter = resp.has_focus()
                    && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.any());
                ui.horizontal(|ui| {
                    let ready = self.health == Some(true) && !self.busy && !self.prompt.trim().is_empty();
                    if ui.add_enabled(ready, egui::Button::new("Send ⏎")).clicked() || (ready && plain_enter) {
                        self.submit();
                        resp.request_focus();
                    }
                    if self.busy {
                        ui.spinner();
                        ui.label(self.steps.last().cloned().unwrap_or_else(|| "thinking…".into()));
                    } else {
                        ui.weak(match self.mode {
                            Mode::Agent => "Agent mode: the model may edit files; review them in Changes.",
                            Mode::Ask => "Ask mode: read-only answers grounded in the indexed code.",
                        });
                    }
                });
            });

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(self.scroll_to_bottom)
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    if self.turns.is_empty() {
                        ui.add_space(24.0);
                        ui.vertical_centered(|ui| {
                            ui.heading("CodeBot");
                            ui.weak("Open a project, then ask questions or request changes.");
                            ui.weak("The program retrieves relevant code, symbols and memory for the local model.");
                        });
                    }
                    for turn in &self.turns {
                        render_turn(ui, turn);
                    }
                    if self.busy && !self.steps.is_empty() {
                        egui::Frame::group(ui.style()).show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            for s in &self.steps {
                                ui.weak(format!("· {s}"));
                            }
                        });
                    }
                });
            self.scroll_to_bottom = false;
        });
    }

    fn settings_window(&mut self, ctx: &egui::Context) {
        if !self.show_settings {
            return;
        }
        let mut open = true;
        let mut save = false;
        egui::Window::new("Settings")
            .open(&mut open)
            .collapsible(false)
            .default_size([460.0, 240.0])
            .show(ctx, |ui| {
                egui::Grid::new("settings-grid").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                    ui.label("LLM base URL");
                    ui.add(egui::TextEdit::singleline(&mut self.settings.llm_base).desired_width(300.0));
                    ui.end_row();

                    ui.label("Model");
                    ui.horizontal(|ui| {
                        ui.add(egui::TextEdit::singleline(&mut self.settings.llm_model).desired_width(200.0));
                        egui::ComboBox::from_id_salt("models")
                            .selected_text("pick…")
                            .width(90.0)
                            .show_ui(ui, |ui| {
                                if self.models.is_empty() {
                                    ui.weak("no models listed");
                                }
                                for m in &self.models {
                                    if ui.selectable_label(self.settings.llm_model == *m, m).clicked() {
                                        self.settings.llm_model = m.clone();
                                    }
                                }
                            });
                        if ui.small_button("⟳").clicked() {
                            self.send(Cmd::ListModels);
                        }
                    });
                    ui.end_row();

                    ui.label("API key");
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.api_key_input)
                                .password(true)
                                .desired_width(200.0)
                                .hint_text(if self.settings.has_api_key { "(set — leave blank to keep)" } else { "(none)" }),
                        );
                    });
                    ui.end_row();

                    ui.label("Context budget (chars)");
                    ui.add(egui::DragValue::new(&mut self.settings.context_chars).range(4_000..=200_000).speed(500));
                    ui.end_row();
                });
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        save = true;
                    }
                    ui.weak("Saved to the workspace store; applies immediately.");
                });
            });
        if save {
            let mut s = self.settings.clone();
            s.llm_api_key = if self.api_key_input.is_empty() { None } else { Some(self.api_key_input.clone()) };
            self.api_key_input.clear();
            self.send(Cmd::SetSettings(s));
        }
        self.show_settings = open;
    }
}

fn render_turn(ui: &mut egui::Ui, turn: &Turn) {
    let is_user = turn.role == "user";
    let fill = if is_user { ui.visuals().faint_bg_color } else { ui.visuals().panel_fill };
    egui::Frame::new()
        .fill(fill)
        .inner_margin(10.0)
        .corner_radius(6.0)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.small(if is_user { "You" } else { "CodeBot" });
            if is_user {
                ui.add(egui::Label::new(markdown::normalise(&turn.content)).wrap());
            } else {
                markdown::render(ui, &turn.content);
            }
            if !turn.edited.is_empty() {
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    ui.weak("Edited:");
                    for f in &turn.edited {
                        ui.label(egui::RichText::new(f).monospace().color(egui::Color32::from_rgb(110, 200, 120)));
                    }
                });
            }
            for f in &turn.failed {
                ui.colored_label(egui::Color32::from_rgb(225, 110, 110), format!("⚠ {f}"));
            }
        });
    ui.add_space(6.0);
}
