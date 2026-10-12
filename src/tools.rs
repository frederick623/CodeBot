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
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(p, new_content)?;
        Ok(())
    }

    /// Absolute, traversal-checked path for a workspace-relative file.
    pub fn resolve(&self, rel: &str) -> Result<std::path::PathBuf> {
        self.safe_path(rel)
    }

    /// Literal (case-insensitive) text search over `files` (absolute paths),
    /// returned as `relative:line: text`. Pure Rust so there is no dependency
    /// on `rg` being installed; capped to keep prompts small.
    pub fn search_in(&self, files: &[String], needle: &str, limit: usize) -> Vec<String> {
        let needle = needle.to_lowercase();
        if needle.trim().is_empty() {
            return vec![];
        }
        let prefix = format!("{}/", self.root.trim_end_matches('/'));
        let mut out = Vec::new();
        'files: for f in files {
            let Ok(content) = std::fs::read_to_string(f) else { continue };
            let rel = f.strip_prefix(&prefix).unwrap_or(f);
            for (i, line) in content.lines().enumerate() {
                if line.to_lowercase().contains(&needle) {
                    let shown: String = line.trim().chars().take(160).collect();
                    out.push(format!("{rel}:{}: {shown}", i + 1));
                    if out.len() >= limit {
                        break 'files;
                    }
                }
            }
        }
        out
    }

    // ----- git ---------------------------------------------------------------

    /// Working-tree changes relative to HEAD (staged + unstaged + untracked),
    /// each with a unified diff. Untracked files are diffed against `/dev/null`
    /// so new files show as pure additions. Returns an empty list outside a git
    /// repository.
    pub fn git_changes(&self) -> Result<Vec<GitChange>> {
        let status = Command::new("git")
            .args(["status", "--porcelain=v1", "--untracked-files=all"])
            .current_dir(&self.root)
            .output()?;
        if !status.status.success() {
            return Ok(vec![]);
        }
        let mut out = Vec::new();
        for line in String::from_utf8_lossy(&status.stdout).lines() {
            if line.len() < 4 {
                continue;
            }
            let code = &line[..2];
            let mut path = line[3..].to_string();
            if let Some((_, to)) = path.split_once(" -> ") {
                path = to.to_string();
            }
            let path = path.trim_matches('"').to_string();
            let status = match code.trim() {
                "??" => "added",
                "A" | "AM" | "MA" => "added",
                "D" | " D" | "D " => "deleted",
                _ => "modified",
            }
            .to_string();
            let diff = if code == "??" {
                Command::new("git")
                    .args(["diff", "--no-color", "--no-index", "--", "/dev/null", &path])
                    .current_dir(&self.root)
                    .output()
                    .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                    .unwrap_or_default()
            } else {
                Command::new("git")
                    .args(["diff", "--no-color", "HEAD", "--", &path])
                    .current_dir(&self.root)
                    .output()
                    .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                    .unwrap_or_default()
            };
            let (additions, deletions) = count_diff_lines(&diff);
            out.push(GitChange { path, status, diff, additions, deletions });
        }
        Ok(out)
    }

    /// Discard the working-tree change to one file: restore tracked files from
    /// HEAD, delete untracked ones. Sandboxed to the workspace.
    pub fn git_revert(&self, rel: &str) -> Result<()> {
        let p = self.safe_path(rel)?;
        let tracked = Command::new("git")
            .args(["ls-files", "--error-unmatch", "--", rel])
            .current_dir(&self.root)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if tracked {
            let out = Command::new("git")
                .args(["checkout", "HEAD", "--", rel])
                .current_dir(&self.root)
                .output()?;
            if !out.status.success() {
                anyhow::bail!("{}", String::from_utf8_lossy(&out.stderr).trim());
            }
        } else if p.exists() {
            std::fs::remove_file(&p)?;
        }
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
        let rel_path = Path::new(rel);
        // Lexical check first: a `..` component in a not-yet-existing path would
        // survive the `starts_with` test below unresolved.
        if rel_path.is_absolute()
            || rel_path
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir | std::path::Component::Prefix(_)))
        {
            anyhow::bail!("path escapes workspace: {}", rel);
        }
        let joined = root.join(rel_path);
        let canon = joined.canonicalize().unwrap_or(joined);
        if !canon.starts_with(&root) {
            anyhow::bail!("path escapes workspace: {}", rel);
        }
        Ok(canon)
    }

    #[allow(dead_code)]
    fn root_path(&self) -> &Path { Path::new(&self.root) }
}

/// One changed file in the working tree, as shown in the GUI's Changes pane.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GitChange {
    pub path: String,
    /// "added" | "modified" | "deleted"
    pub status: String,
    pub diff: String,
    pub additions: usize,
    pub deletions: usize,
}

fn count_diff_lines(diff: &str) -> (usize, usize) {
    let mut add = 0;
    let mut del = 0;
    for l in diff.lines() {
        if l.starts_with("+++") || l.starts_with("---") {
            continue;
        }
        if l.starts_with('+') { add += 1; } else if l.starts_with('-') { del += 1; }
    }
    (add, del)
}