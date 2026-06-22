use crate::api::Patch;
use anyhow::Result;
use std::path::Path;
use std::process::Command;

/// Tools the model may invoke. Apply-patch always returns a preview;
/// actual write happens only after the extension reports user approval.
pub struct ToolExecutor {
    root: String,
}

impl ToolExecutor {
    pub fn new(root: impl Into<String>) -> Self { Self { root: root.into() } }

    pub fn read_file(&self, rel: &str) -> Result<String> {
        let p = self.safe_path(rel)?;
        Ok(std::fs::read_to_string(p)?)
    }

    /// ripgrep-backed exact search.
    pub fn search(&self, pattern: &str) -> Result<Vec<String>> {
        let out = Command::new("rg")
            .args(["--line-number", "--no-heading", "--color", "never", pattern])
            .current_dir(&self.root)
            .output()?;
        Ok(String::from_utf8_lossy(&out.stdout)
            .lines().take(100).map(|s| s.to_string()).collect())
    }

    /// Build a unified-diff preview from a proposed full-file replacement.
    pub fn make_patch(&self, rel: &str, new_content: &str) -> Result<Patch> {
        let old = self.read_file(rel).unwrap_or_default();
        let diff = similar::TextDiff::from_lines(old.as_str(), new_content);
        let unified = diff
            .unified_diff()
            .context_radius(3)
            .header(rel, rel)
            .to_string();
        Ok(Patch { file: rel.to_string(), diff: unified })
    }

    /// Called by the daemon ONLY after the extension confirms user approval.
    pub fn apply_full(&self, rel: &str, new_content: &str) -> Result<()> {
        let p = self.safe_path(rel)?;
        std::fs::write(p, new_content)?;
        Ok(())
    }

    pub fn run_tests(&self, cmd: &str) -> Result<String> {
        let out = Command::new("sh").arg("-c").arg(cmd)
            .current_dir(&self.root).output()?;
        Ok(format!(
            "exit={}\nstdout:\n{}\nstderr:\n{}",
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        ))
    }

    /// Prevent path traversal outside the workspace.
    fn safe_path(&self, rel: &str) -> Result<std::path::PathBuf> {
        let root = std::fs::canonicalize(&self.root)?;
        let joined = root.join(rel);
        let canon = joined.canonicalize().unwrap_or(joined);
        if !canon.starts_with(&root) {
            anyhow::bail!("path escapes workspace: {}", rel);
        }
        Ok(canon)
    }

    #[allow(dead_code)]
    fn root_path(&self) -> &Path { Path::new(&self.root) }
}