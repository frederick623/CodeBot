//! Standalone minimalist GUI for CodeBot (Path A).
//!
//! The GUI talks to the daemon purely over HTTP. On startup it probes
//! `/health`; if no server is listening it brings one up *in-process* on a
//! background Tokio runtime, then keeps using the same HTTP API. All blocking
//! work (HTTP, model loading) happens off the UI thread; results flow back to
//! egui through a channel.
use codebot_daemon::api::{ApplyResp, ChatResp, IndexStatus};
use codebot_daemon::server;
use eframe::egui;
use std::sync::mpsc as schan;
use std::time::Duration;
use tokio::sync::mpsc as tchan;

/// Base URL derived from the server's bind address.
fn base() -> String {
    format!("http://{}", server::DEFAULT_ADDR)
}

/// Commands the UI thread sends to the async worker.
enum Cmd {
    Open(String),
    Chat { workspace: String, prompt: String },
    Apply { workspace: String, file: String, diff: String, dry_run: bool },
    Status,
}

/// Events the worker sends back to the UI thread.
enum Ui {
    Health(bool),
    Log(String),
    Status(IndexStatus),
    Chat(ChatResp),
    Applied(ApplyResp),
    Error(String),
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "CodeBot",
        options,
        Box::new(|cc| Ok(Box::new(GuiApp::new(cc)))),
    )
}

struct GuiApp {
    cmd_tx: tchan::UnboundedSender<Cmd>,
    ui_rx: schan::Receiver<Ui>,
    workspace: String,
    prompt: String,
    answer: String,
    patches: Vec<codebot_daemon::api::Patch>,
    applied: Vec<ApplyResp>,
    status: Option<IndexStatus>,
    health: Option<bool>,
    logs: Vec<String>,
    busy: bool,
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

        Self {
            cmd_tx,
            ui_rx,
            workspace: std::env::current_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            prompt: String::new(),
            answer: String::new(),
            patches: Vec::new(),
            applied: Vec::new(),
            status: None,
            health: None,
            logs: Vec::new(),
            busy: false,
        }
    }
}

/// Async backend: ensures a server is reachable, then services UI commands.
async fn worker(
    mut cmd_rx: tchan::UnboundedReceiver<Cmd>,
    ui_tx: schan::Sender<Ui>,
    ctx: egui::Context,
) {
    let client = reqwest::Client::new();
    ensure_server(&client, &ui_tx, &ctx).await;

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            Cmd::Open(path) => {
                let r = client
                    .post(format!("{}/workspace/open", base()))
                    .json(&serde_json::json!({ "workspace_path": path }))
                    .send()
                    .await;
                match r {
                    Ok(_) => emit(&ui_tx, &ctx, Ui::Log(format!("opened workspace: {path}"))),
                    Err(e) => emit(&ui_tx, &ctx, Ui::Error(format!("open failed: {e}"))),
                }
                fetch_status(&client, &ui_tx, &ctx).await;
            }
            Cmd::Chat { workspace, prompt } => {
                let r = client
                    .post(format!("{}/chat", base()))
                    .json(&serde_json::json!({
                        "workspace_path": workspace,
                        "user_prompt": prompt,
                    }))
                    .send()
                    .await;
                match parse::<ChatResp>(r).await {
                    Ok(resp) => emit(&ui_tx, &ctx, Ui::Chat(resp)),
                    Err(e) => emit(&ui_tx, &ctx, Ui::Error(format!("chat failed: {e}"))),
                }
            }
            Cmd::Apply { workspace, file, diff, dry_run } => {
                let r = client
                    .post(format!("{}/apply", base()))
                    .json(&serde_json::json!({
                        "workspace_path": workspace,
                        "file": file,
                        "diff": diff,
                        "dry_run": dry_run,
                    }))
                    .send()
                    .await;
                match parse::<ApplyResp>(r).await {
                    Ok(resp) => {
                        let verb = if resp.dry_run { "previewed" } else { "applied" };
                        let msg = match &resp.error {
                            Some(e) => format!("{verb} failed ({}): {e}", resp.file),
                            None => format!("{verb} {}", resp.file),
                        };
                        emit(&ui_tx, &ctx, Ui::Log(msg));
                        emit(&ui_tx, &ctx, Ui::Applied(resp));
                    }
                    Err(e) => emit(&ui_tx, &ctx, Ui::Error(format!("apply failed: {e}"))),
                }
            }
            Cmd::Status => fetch_status(&client, &ui_tx, &ctx).await,
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
    matches!(
        client.get(format!("{}/health", base())).send().await,
        Ok(r) if r.status().is_success()
    )
}

/// Connect to an existing server, or start one in-process and wait for it.
async fn ensure_server(client: &reqwest::Client, tx: &schan::Sender<Ui>, ctx: &egui::Context) {
    if health(client).await {
        emit(tx, ctx, Ui::Log("connected to running server".into()));
        emit(tx, ctx, Ui::Health(true));
        return;
    }
    emit(tx, ctx, Ui::Log("no server found; starting in-process…".into()));
    tokio::spawn(async {
        if let Err(e) = server::run().await {
            eprintln!("in-process server exited: {e}");
        }
    });
    // First boot loads embedding models, which can take a while.
    for _ in 0..600 {
        tokio::time::sleep(Duration::from_millis(500)).await;
        if health(client).await {
            emit(tx, ctx, Ui::Log("server ready".into()));
            emit(tx, ctx, Ui::Health(true));
            return;
        }
    }
    emit(tx, ctx, Ui::Health(false));
    emit(tx, ctx, Ui::Error("server failed to start".into()));
}

/// Fetch and forward the current index status.
async fn fetch_status(client: &reqwest::Client, tx: &schan::Sender<Ui>, ctx: &egui::Context) {
    let r = client.get(format!("{}/index/status", base())).send().await;
    match parse::<IndexStatus>(r).await {
        Ok(s) => emit(tx, ctx, Ui::Status(s)),
        Err(e) => emit(tx, ctx, Ui::Error(format!("status failed: {e}"))),
    }
}

/// Decode a JSON response body, surfacing transport and decode errors uniformly.
async fn parse<T: serde::de::DeserializeOwned>(
    r: reqwest::Result<reqwest::Response>,
) -> Result<T, String> {
    r.map_err(|e| e.to_string())?
        .json::<T>()
        .await
        .map_err(|e| e.to_string())
}

/// Render a unified diff as colored, monospace lines (git-style: additions
/// green, deletions red, hunk headers highlighted).
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
            egui::Label::new(egui::RichText::new(line).monospace().color(color)).wrap_mode(egui::TextWrapMode::Extend),
        );
    }
}

impl eframe::App for GuiApp {
    fn logic(&mut self, _ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain everything the worker produced since the last frame.
        while let Ok(ev) = self.ui_rx.try_recv() {
            match ev {
                Ui::Health(h) => self.health = Some(h),
                Ui::Log(m) => self.logs.push(m),
                Ui::Status(s) => self.status = Some(s),
                Ui::Error(e) => self.logs.push(format!("error: {e}")),
                Ui::Chat(resp) => {
                    self.busy = false;
                    self.answer = resp.answer;
                    self.patches = resp.patches;
                }
                Ui::Applied(resp) => self.applied.push(resp),
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("top").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                let dot = match self.health {
                    Some(true) => "🟢 server up",
                    Some(false) => "🔴 server down",
                    None => "🟡 starting…",
                };
                ui.label(dot);
                ui.separator();
                if let Some(s) = &self.status {
                    ui.label(format!("{} files · {} chunks", s.files, s.chunks));
                }
            });
            ui.horizontal(|ui| {
                ui.label("Workspace:");
                ui.text_edit_singleline(&mut self.workspace);
                if ui.button("Open").clicked() {
                    let _ = self.cmd_tx.send(Cmd::Open(self.workspace.clone()));
                }
                if ui.button("Refresh").clicked() {
                    let _ = self.cmd_tx.send(Cmd::Status);
                }
            });
        });

        egui::Panel::bottom("logs")
            .resizable(true)
            .show_inside(ui, |ui| {
                ui.label("Log");
                egui::ScrollArea::vertical()
                    .max_height(80.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in &self.logs {
                            ui.monospace(line);
                        }
                    });
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.label("Prompt");
            ui.add(
                egui::TextEdit::multiline(&mut self.prompt)
                    .desired_rows(3)
                    .desired_width(f32::INFINITY),
            );
            ui.horizontal(|ui| {
                let ready = self.health == Some(true) && !self.busy && !self.prompt.is_empty();
                if ui.add_enabled(ready, egui::Button::new("Send")).clicked() {
                    self.busy = true;
                    self.answer.clear();
                    self.patches.clear();
                    self.applied.clear();
                    let _ = self.cmd_tx.send(Cmd::Chat {
                        workspace: self.workspace.clone(),
                        prompt: self.prompt.clone(),
                    });
                }
                if self.busy {
                    ui.spinner();
                    ui.label("thinking…");
                }
            });
            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                if !self.answer.is_empty() {
                    ui.heading("Answer");
                    ui.label(&self.answer);
                }
                if !self.patches.is_empty() {
                    ui.separator();
                    ui.heading("Proposed patches");
                    let tx = self.cmd_tx.clone();
                    let workspace = self.workspace.clone();
                    for p in &self.patches {
                        ui.collapsing(p.file.clone(), |ui| {
                            ui.horizontal(|ui| {
                                if ui.button("👁 Preview").clicked() {
                                    let _ = tx.send(Cmd::Apply {
                                        workspace: workspace.clone(),
                                        file: p.file.clone(),
                                        diff: p.diff.clone(),
                                        dry_run: true,
                                    });
                                }
                                if ui.button("✅ Apply").clicked() {
                                    let _ = tx.send(Cmd::Apply {
                                        workspace: workspace.clone(),
                                        file: p.file.clone(),
                                        diff: p.diff.clone(),
                                        dry_run: false,
                                    });
                                }
                            });
                            diff_view(ui, &p.diff);
                        });
                    }
                }
                if !self.applied.is_empty() {
                    ui.separator();
                    ui.heading("Results (before → after)");
                    for a in &self.applied {
                        let title = match (a.ok, a.dry_run) {
                            (true, true) => format!("👁 {} (preview)", a.file),
                            (true, false) => format!("✓ {}", a.file),
                            (false, _) => format!("✗ {}", a.file),
                        };
                        ui.collapsing(title, |ui| {
                            if let Some(err) = &a.error {
                                ui.colored_label(egui::Color32::from_rgb(225, 110, 110), err);
                            }
                            diff_view(ui, &a.diff);
                        });
                    }
                }
            });
        });
    }
}
