//! Headless daemon entry point.
//!
//! All wiring now lives in the shared library crate (`codebot_daemon`). This
//! binary just initialises logging and hands off to [`server::run`]. The GUI
//! binary reuses the same server and can start it in-process on demand.
use anyhow::Result;
use codebot_daemon::server;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().json().with_env_filter("info").init();
    server::run().await
}
