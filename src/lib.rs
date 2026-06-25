//! Shared library crate for the CodeBot daemon.
//!
//! The HTTP server, all core subsystems, and the request/response DTOs live here
//! so multiple binaries (the headless daemon and the standalone GUI) can reuse
//! the exact same code. The GUI can therefore bring up the server in-process
//! when one is not already listening, while still talking to it over HTTP.
pub mod api;
pub mod embed_client;
pub mod indexer;
pub mod llm;
pub mod orchestrator;
pub mod prompt;
pub mod retrieval;
pub mod server;
pub mod storage;
pub mod tools;
pub mod verifier;
pub mod watcher;
