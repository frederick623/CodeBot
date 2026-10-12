//! SEARCH/REPLACE edit blocks.
//!
//! Small local models are unreliable at producing well-formed unified diffs
//! (line counts in `@@` headers drift, context goes stale), but they are good
//! at copying a few lines verbatim and writing their replacement. So the agent
//! asks for edits in this shape:
//!
//! ```text
//! path/to/file.rs
//! <<<<<<< SEARCH
//! lines copied exactly from the file
//! =======
//! replacement lines
//! >>>>>>> REPLACE
//! ```
//!
//! The *program* then does the hard part: locating the SEARCH text in the real
//! file (exactly, then ignoring indentation, then fuzzily), re-indenting the
//! replacement to match, and producing the final file content. An empty SEARCH
//! section creates a new file (or appends when the file already exists).

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditBlock {
    /// Workspace-relative path as written by the model.
    pub path: String,
    pub search: String,
    pub replace: String,
}

/// Parse every SEARCH/REPLACE block in `text`. The path is taken from the same
/// line as the `<<<<<<< SEARCH` marker when present, otherwise from the closest
/// preceding non-empty line (code fences and decoration are stripped).
pub fn parse_blocks(text: &str) -> Vec<EditBlock> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let t = lines[i].trim();
        let Some(rest) = strip_marker(t, '<', "SEARCH") else {
            i += 1;
            continue;
        };
        let mut path = rest.trim().to_string();
        if path.is_empty() {
            path = previous_path_line(&lines, i).unwrap_or_default();
        }
        let path = clean_path(&path);

        let mut search = Vec::new();
        let mut replace = Vec::new();
        let mut j = i + 1;
        let mut saw_divider = false;
        let mut closed = false;
        while j < lines.len() {
            let lt = lines[j].trim();
            if !saw_divider && lt.starts_with("=======") && lt.chars().all(|c| c == '=') {
                saw_divider = true;
            } else if saw_divider && strip_marker(lt, '>', "REPLACE").is_some() {
                closed = true;
                break;
            } else if !saw_divider && strip_marker(lt, '<', "SEARCH").is_some() {
                // A new block started before this one closed: abandon the broken one.
                break;
            } else if saw_divider {
                replace.push(lines[j]);
            } else {
                search.push(lines[j]);
            }
            j += 1;
        }
        if closed && !path.is_empty() {
            out.push(EditBlock {
                path,
                search: join_lines(&search),
                replace: join_lines(&replace),
            });
            i = j + 1;
        } else {
            i += 1;
        }
    }
    out
}

/// `text` with every block removed — the model's prose explanation.
pub fn strip_blocks(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut out: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let t = lines[i].trim();
        if strip_marker(t, '<', "SEARCH").is_some() {
            // Drop the path line we just emitted, if it looks like one.
            if let Some(last) = out.last() {
                if is_path_like(last.trim()) {
                    out.pop();
                }
            }
            while i < lines.len() && strip_marker(lines[i].trim(), '>', "REPLACE").is_none() {
                i += 1;
            }
            i += 1;
            continue;
        }
        if t.starts_with("```") {
            // Fences that only wrapped blocks become noise once blocks are gone.
            i += 1;
            continue;
        }
        out.push(lines[i]);
        i += 1;
    }
    let s = out.join("\n");
    collapse_blank_lines(s.trim())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApplyError {
    /// The SEARCH text could not be located; carries the path and the search text.
    NotFound { path: String, search: String },
}

impl std::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApplyError::NotFound { path, search } => {
                let preview: String = search.lines().take(3).collect::<Vec<_>>().join(" | ");
                write!(f, "{path}: SEARCH text not found: {preview}")
            }
        }
    }
}

/// Outcome of applying a set of blocks to in-memory file contents.
#[derive(Debug, Default)]
pub struct ApplyOutcome {
    /// Final content for every file touched by at least one applied block.
    pub files: BTreeMap<String, String>,
    pub applied: usize,
    pub errors: Vec<ApplyError>,
}

/// Apply `blocks` to `contents` (path → current text, `None` = file does not
/// exist). Blocks are applied in order; later blocks see earlier replacements.
pub fn apply_blocks(blocks: &[EditBlock], contents: &BTreeMap<String, Option<String>>) -> ApplyOutcome {
    let mut working: BTreeMap<String, Option<String>> = contents.clone();
    let mut out = ApplyOutcome::default();
    for b in blocks {
        let cur = working.get(&b.path).cloned().flatten();
        match apply_one(cur.as_deref(), &b.search, &b.replace) {
            Some(new) => {
                working.insert(b.path.clone(), Some(new.clone()));
                out.files.insert(b.path.clone(), new);
                out.applied += 1;
            }
            None => out.errors.push(ApplyError::NotFound {
                path: b.path.clone(),
                search: b.search.clone(),
            }),
        }
    }
    out
}

/// Apply a single block to `current`. Returns the new content or `None` when
/// the SEARCH text cannot be located.
pub fn apply_one(current: Option<&str>, search: &str, replace: &str) -> Option<String> {
    let crlf = current.map_or(false, |c| c.contains("\r\n"));
    let cur = current.map(|c| c.replace("\r\n", "\n"));
    let search = search.replace("\r\n", "\n");
    let replace = replace.replace("\r\n", "\n");

    let result = match cur {
        None => {
            if search.trim().is_empty() { Some(replace.clone()) } else { None }
        }
        Some(cur) => {
            if search.trim().is_empty() {
                // Append to an existing file.
                let mut s = cur.clone();
                if !s.is_empty() && !s.ends_with('\n') { s.push('\n'); }
                s.push_str(&replace);
                Some(s)
            } else if search == replace {
                Some(cur)
            } else {
                replace_exact(&cur, &search, &replace)
                    .or_else(|| replace_by_lines(&cur, &search, &replace, 1.0))
                    .or_else(|| replace_by_lines(&cur, &search, &replace, 0.88))
            }
        }
    };
    result.map(|r| if crlf { r.replace('\n', "\r\n") } else { r })
}

fn replace_exact(cur: &str, search: &str, replace: &str) -> Option<String> {
    let idx = cur.find(search)?;
    let mut s = String::with_capacity(cur.len() + replace.len());
    s.push_str(&cur[..idx]);
    s.push_str(replace);
    s.push_str(&cur[idx + search.len()..]);
    Some(s)
}

/// Line-window matching that ignores indentation and trailing whitespace.
/// `min_ratio == 1.0` requires every trimmed line to match; lower values accept
/// the best window whose per-line similarity averages at least `min_ratio`.
fn replace_by_lines(cur: &str, search: &str, replace: &str, min_ratio: f32) -> Option<String> {
    let cur_lines: Vec<&str> = split_keep(cur);
    let s_lines: Vec<&str> = trim_block_lines(search);
    if s_lines.is_empty() || cur_lines.len() < s_lines.len() {
        return None;
    }
    let n = s_lines.len();
    let mut best: Option<(usize, f32)> = None;
    for start in 0..=cur_lines.len() - n {
        let mut total = 0f32;
        let mut ok = true;
        for k in 0..n {
            let a = cur_lines[start + k].trim();
            let b = s_lines[k].trim();
            let r = if a == b { 1.0 } else if min_ratio >= 1.0 { ok = false; break } else { line_similarity(a, b) };
            total += r;
        }
        if !ok { continue; }
        let avg = total / n as f32;
        if avg >= min_ratio && best.map_or(true, |(_, b)| avg > b) {
            best = Some((start, avg));
            if avg >= 1.0 { break; }
        }
    }
    let (start, _) = best?;

    // Re-indent the replacement by the difference between the file's actual
    // indentation and the indentation the model wrote in its SEARCH text.
    let actual_indent = leading_ws(cur_lines[start]);
    let written_indent = leading_ws(first_nonblank(&s_lines).unwrap_or(""));
    let r_lines: Vec<&str> = trim_block_lines(replace);
    let reindented: Vec<String> = r_lines
        .iter()
        .map(|l| reindent(l, written_indent, actual_indent))
        .collect();

    let mut out: Vec<String> = Vec::with_capacity(cur_lines.len());
    out.extend(cur_lines[..start].iter().map(|s| s.to_string()));
    out.extend(reindented);
    out.extend(cur_lines[start + n..].iter().map(|s| s.to_string()));
    let mut s = out.join("\n");
    if cur.ends_with('\n') && !s.ends_with('\n') { s.push('\n'); }
    Some(s)
}

/// Ratio of shared characters in order (normalised Levenshtein-like via
/// `similar`), used only for the fuzzy pass.
fn line_similarity(a: &str, b: &str) -> f32 {
    if a.is_empty() && b.is_empty() { return 1.0; }
    similar::TextDiff::from_chars(a, b).ratio()
}

fn reindent(line: &str, written: &str, actual: &str) -> String {
    if line.trim().is_empty() {
        return String::new();
    }
    match line.strip_prefix(written) {
        Some(rest) => format!("{actual}{rest}"),
        None => {
            // Model used less indentation than it wrote in SEARCH; keep the
            // line's own indentation but shift it by the net difference.
            let own = leading_ws(line);
            let extra = actual.len().saturating_sub(written.len());
            format!("{}{}{}", own, " ".repeat(extra), line.trim_start())
        }
    }
}

fn leading_ws(s: &str) -> &str {
    let end = s.len() - s.trim_start().len();
    &s[..end]
}

fn first_nonblank<'a>(lines: &[&'a str]) -> Option<&'a str> {
    lines.iter().copied().find(|l| !l.trim().is_empty())
}

/// Split into lines without a trailing phantom line for a final newline.
fn split_keep(s: &str) -> Vec<&str> {
    let mut v: Vec<&str> = s.split('\n').collect();
    if s.ends_with('\n') { v.pop(); }
    v
}

/// Block lines with leading/trailing blank lines dropped (models often pad).
fn trim_block_lines(s: &str) -> Vec<&str> {
    let mut v = split_keep(s);
    while v.first().map_or(false, |l| l.trim().is_empty()) { v.remove(0); }
    while v.last().map_or(false, |l| l.trim().is_empty()) { v.pop(); }
    v
}

fn join_lines(lines: &[&str]) -> String {
    let mut s = lines.join("\n");
    if !lines.is_empty() { s.push('\n'); }
    s
}

/// `<<<<<<< SEARCH[ path]` / `>>>>>>> REPLACE` with 3+ marker characters.
fn strip_marker<'a>(line: &'a str, ch: char, word: &str) -> Option<&'a str> {
    let n = line.chars().take_while(|c| *c == ch).count();
    if n < 3 { return None; }
    let rest = line[n..].trim_start();
    let rest = rest.strip_prefix(word)?;
    if rest.chars().next().map_or(false, |c| c.is_alphanumeric()) { return None; }
    Some(rest)
}

fn previous_path_line(lines: &[&str], idx: usize) -> Option<String> {
    (0..idx).rev().map(|k| lines[k].trim()).find(|l| {
        !l.is_empty() && !l.starts_with("```") && !l.starts_with("=======") && is_path_like(l)
    }).map(|s| s.to_string())
}

fn is_path_like(s: &str) -> bool {
    let c = clean_path(s);
    !c.is_empty() && !c.contains(' ') && (c.contains('/') || c.contains('.'))
}

/// Strip decoration models add around paths: backticks, bold, "File:", a
/// trailing colon, leading `./`.
fn clean_path(s: &str) -> String {
    let mut p = s.trim().trim_matches('`').trim_matches('*').trim();
    for prefix in ["File:", "file:", "Path:", "path:", "###", "##", "#"] {
        if let Some(r) = p.strip_prefix(prefix) { p = r.trim(); }
    }
    let p = p.trim_end_matches(':').trim().trim_matches('`').trim();
    let p = p.strip_prefix("./").unwrap_or(p);
    p.to_string()
}

fn collapse_blank_lines(s: &str) -> String {
    let mut out = String::new();
    let mut blank = 0;
    for l in s.lines() {
        if l.trim().is_empty() {
            blank += 1;
            if blank > 1 { continue; }
        } else {
            blank = 0;
        }
        out.push_str(l);
        out.push('\n');
    }
    out.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "Here is the change.\n\nsrc/lib.rs\n```rust\n<<<<<<< SEARCH\nfn a() {}\n=======\nfn a() { 1 }\n>>>>>>> REPLACE\n```\n\n`src/new.rs`\n<<<<<<< SEARCH\n=======\npub fn b() {}\n>>>>>>> REPLACE\n";

    #[test]
    fn parses_blocks_and_paths() {
        let b = parse_blocks(SAMPLE);
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].path, "src/lib.rs");
        assert_eq!(b[0].search, "fn a() {}\n");
        assert_eq!(b[0].replace, "fn a() { 1 }\n");
        assert_eq!(b[1].path, "src/new.rs");
        assert_eq!(b[1].search, "");
        assert_eq!(strip_blocks(SAMPLE), "Here is the change.");
    }

    #[test]
    fn path_on_marker_line() {
        let b = parse_blocks("<<<<<<< SEARCH a/b.py\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n");
        assert_eq!(b[0].path, "a/b.py");
    }

    #[test]
    fn exact_and_indent_tolerant_apply() {
        let cur = "fn main() {\n    let x = 1;\n    println!(\"{x}\");\n}\n";
        let exact = apply_one(Some(cur), "    let x = 1;\n", "    let x = 2;\n").unwrap();
        assert!(exact.contains("let x = 2;"));
        // Model dropped the indentation in both sections: re-indented on apply.
        let loose = apply_one(Some(cur), "let x = 1;\nprintln!(\"{x}\");\n", "let x = 3;\nprintln!(\"{x}!\");\n").unwrap();
        assert_eq!(loose, "fn main() {\n    let x = 3;\n    println!(\"{x}!\");\n}\n");
    }

    #[test]
    fn fuzzy_apply_tolerates_small_drift() {
        let cur = "a\nlet total = compute_total(items);\nb\n";
        let got = apply_one(Some(cur), "let total = compute_total(item);\n", "let total = 0;\n").unwrap();
        assert_eq!(got, "a\nlet total = 0;\nb\n");
        assert!(apply_one(Some(cur), "completely different text\n", "x\n").is_none());
    }

    #[test]
    fn new_file_and_append() {
        assert_eq!(apply_one(None, "", "hello\n").unwrap(), "hello\n");
        assert!(apply_one(None, "something\n", "x\n").is_none());
        assert_eq!(apply_one(Some("a"), "", "b\n").unwrap(), "a\nb\n");
    }

    #[test]
    fn apply_blocks_chains_edits_and_reports_errors() {
        let blocks = parse_blocks(
            "f.txt\n<<<<<<< SEARCH\none\n=======\nuno\n>>>>>>> REPLACE\nf.txt\n<<<<<<< SEARCH\nuno\ntwo\n=======\nuno\ndos\n>>>>>>> REPLACE\nf.txt\n<<<<<<< SEARCH\nmissing\n=======\nx\n>>>>>>> REPLACE\n",
        );
        let mut contents = BTreeMap::new();
        contents.insert("f.txt".to_string(), Some("one\ntwo\n".to_string()));
        let out = apply_blocks(&blocks, &contents);
        assert_eq!(out.applied, 2);
        assert_eq!(out.errors.len(), 1);
        assert_eq!(out.files["f.txt"], "uno\ndos\n");
    }
}
