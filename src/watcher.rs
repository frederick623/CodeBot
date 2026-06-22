use crate::indexer::Indexer;
use anyhow::Result;
use notify::{RecursiveMode, Watcher};
use notify_debouncer_full::{new_debouncer, DebouncedEvent};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

/// Watches the workspace and feeds changed paths to the indexer.
/// Each event invalidates only the affected file, never the whole index.
pub fn spawn_watcher(root: PathBuf, indexer: Arc<Indexer>) -> Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel::<Vec<DebouncedEvent>>();

    // notify runs on its own thread; bridge into tokio via the channel.
    let mut debouncer = new_debouncer(
        Duration::from_millis(400),
        None,
        move |res: notify_debouncer_full::DebounceEventResult| {
            if let Ok(events) = res {
                let _ = tx.send(events);
            }
        },
    )?;
    debouncer.watcher().watch(&root, RecursiveMode::Recursive)?;

    tokio::spawn(async move {
        // Keep the debouncer alive for the lifetime of the task.
        let _keep = debouncer;
        while let Some(batch) = rx.recv().await {
            for ev in batch {
                for path in ev.event.paths {
                    if path.is_dir() { continue; }
                    let idx = indexer.clone();
                    use notify::EventKind::*;
                    match ev.event.kind {
                        Remove(_) => { let _ = idx.remove_file(&path); }
                        Create(_) | Modify(_) => {
                            // hash check inside index_file makes this cheap if unchanged
                            let _ = idx.index_file(&path).await;
                        }
                        _ => {}
                    }
                }
            }
        }
    });
    Ok(())
}