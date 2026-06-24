use crate::embed_client::EmbedClient;
use crate::storage::Store;
use anyhow::Result;
use ignore::WalkBuilder;
use std::collections::HashSet;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tree_sitter::{Parser, Query, QueryCursor};

const MAX_FILE_BYTES: u64 = 1_000_000; // skip huge/generated files

pub struct Indexer {
    store: Store,
    embed: EmbedClient,
}

impl Indexer {
    pub fn new(store: Store, embed: EmbedClient) -> Self {
        Self { store, embed }
    }

    /// Full walk, but each file is only re-embedded if its content hash changed.
    pub async fn index_workspace(&self, root: &Path) -> Result<()> {
        let walker = WalkBuilder::new(root)
            .standard_filters(true) // respects .gitignore, hidden files
            .build();
        for entry in walker.flatten() {
            let path = entry.path();
            if !path.is_file() { continue; }
            if let Ok(meta) = path.metadata() {
                if meta.len() > MAX_FILE_BYTES { continue; }
            }
            let _ = self.index_file(path).await;
        }
        Ok(())
    }

    /// Incremental: hash content, skip if unchanged, otherwise invalidate + rebuild.
    pub async fn index_file(&self, path: &Path) -> Result<()> {
        let rel = path.to_string_lossy().to_string();
        if is_probably_secret(&rel) { return Ok(()); }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return Ok(()), // binary or unreadable
        };
        let hash = blake3::hash(content.as_bytes()).to_hex().to_string();

        if let Some(existing) = self.store.file_hash(&rel)? {
            if existing == hash {
                return Ok(()); // unchanged → no work, the heart of incremental indexing
            }
        }
        self.store.invalidate_file(&rel)?;

        let lang = detect_lang(path);
        let chunks = chunk_file(&content, lang);

        // Persist symbols for the symbol graph / exact lookup, plus the
        // import/call edges that make up the code graph.
        if let Some(l) = lang {
            let symbols = extract_symbols(&content, l);
            for sym in &symbols {
                self.store.insert_symbol(&rel, &sym.name, &sym.kind, sym.start, sym.end)?;
            }

            // File-level import edges: <file>::<file> -> imported module.
            let file_node = format!("{}::<file>", rel);
            let mut seen: HashSet<(String, String)> = HashSet::new();
            for module in extract_imports(&content, l) {
                if seen.insert((file_node.clone(), module.clone())) {
                    self.store.insert_edge(&file_node, &module, "import")?;
                }
            }

            // Call edges attributed to the enclosing definition (or the file
            // node when a call sits at module scope).
            for (callee, line) in extract_calls(&content, l) {
                let src = enclosing_symbol(&symbols, line)
                    .map(|s| format!("{}::{}", rel, s))
                    .unwrap_or_else(|| file_node.clone());
                if seen.insert((src.clone(), callee.clone())) {
                    self.store.insert_edge(&src, &callee, "call")?;
                }
            }
        }

        // Insert chunks, then batch-embed them in-process.
        let mut chunk_ids = Vec::new();
        let mut payloads = Vec::new();
        for ch in &chunks {
            let id = self.store.insert_chunk(
                &rel, ch.start, ch.end, ch.symbol.as_deref(), &ch.text,
            )?;
            chunk_ids.push(id);
            // The text sent for embedding is a DATA payload, never an instruction.
            payloads.push(format!("file: {}\n{}", rel, ch.text));
        }
        if !payloads.is_empty() {
            let vecs = self.embed.embed(&payloads).await?;
            for (id, v) in chunk_ids.iter().zip(vecs) {
                self.store.insert_embedding(*id, &v)?;
            }
        }

        let now = now_ms();
        let size = content.len() as i64;
        self.store.upsert_file(&rel, &hash, size, now, now)?;
        Ok(())
    }

    pub fn remove_file(&self, path: &Path) -> Result<()> {
        self.store.invalidate_file(&path.to_string_lossy())
    }
}

#[derive(Clone, Copy)]
pub enum Lang { Ts, Rust, Python, C, Cpp, Java, CSharp, Go }

pub(crate) fn detect_lang(p: &Path) -> Option<Lang> {
    match p.extension().and_then(|e| e.to_str()) {
        Some("ts") | Some("tsx") | Some("js") | Some("jsx") => Some(Lang::Ts),
        Some("rs") => Some(Lang::Rust),
        Some("py") => Some(Lang::Python),
        Some("c") => Some(Lang::C),
        // Headers are routed to the C++ grammar, which is a superset of C.
        Some("cc") | Some("cpp") | Some("cxx") | Some("c++")
        | Some("h") | Some("hh") | Some("hpp") | Some("hxx") => Some(Lang::Cpp),
        Some("java") => Some(Lang::Java),
        Some("cs") => Some(Lang::CSharp),
        Some("go") => Some(Lang::Go),
        _ => None,
    }
}

pub struct Chunk {
    pub start: usize,
    pub end: usize,
    pub symbol: Option<String>,
    pub text: String,
}

/// AST-aware chunking for known languages; fall back to a sliding window.
fn chunk_file(content: &str, lang: Option<Lang>) -> Vec<Chunk> {
    if let Some(l) = lang {
        let syms = extract_symbols(content, l);
        if !syms.is_empty() {
            let lines: Vec<&str> = content.lines().collect();
            return syms
                .into_iter()
                .map(|s| {
                    let body = lines
                        .get(s.start.saturating_sub(1)..s.end.min(lines.len()))
                        .map(|sl| sl.join("\n"))
                        .unwrap_or_default();
                    Chunk { start: s.start, end: s.end, symbol: Some(s.name), text: body }
                })
                .collect();
        }
    }
    sliding_window(content, 60, 10)
}

fn sliding_window(content: &str, window: usize, overlap: usize) -> Vec<Chunk> {
    let lines: Vec<&str> = content.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let end = (i + window).min(lines.len());
        out.push(Chunk {
            start: i + 1,
            end,
            symbol: None,
            text: lines[i..end].join("\n"),
        });
        if end == lines.len() { break; }
        i += window - overlap;
    }
    out
}

pub struct Symbol { pub name: String, pub kind: String, pub start: usize, pub end: usize }

/// Extract top-level definitions using tree-sitter queries.
fn extract_symbols(content: &str, lang: Lang) -> Vec<Symbol> {
    let language = language_for(lang);
    let query_src = def_query_src(lang);

    let mut parser = Parser::new();
    if parser.set_language(&language).is_err() { return vec![]; }
    let tree = match parser.parse(content, None) { Some(t) => t, None => return vec![] };
    let query = match Query::new(&language, query_src) { Ok(q) => q, Err(_) => return vec![] };

    let name_idx = query.capture_index_for_name("name");
    let mut cursor = QueryCursor::new();
    let bytes = content.as_bytes();
    let mut out = Vec::new();

    let mut matches = cursor.matches(&query, tree.root_node(), bytes);
    while let Some(m) = matches.next() {
        let mut name = String::from("<anon>");
        let mut node_for_span = None;
        for cap in m.captures {
            if Some(cap.index) == name_idx {
                name = cap.node.utf8_text(bytes).unwrap_or("<anon>").to_string();
            } else {
                node_for_span = Some(cap.node);
            }
        }
        if let Some(n) = node_for_span {
            out.push(Symbol {
                name,
                kind: n.kind().to_string(),
                start: n.start_position().row + 1,
                end: n.end_position().row + 1,
            });
        }
    }
    out
}

/// Per-language definition queries. Each pattern binds the definition node to
/// `@def` and (where one exists) its name identifier to `@name`. Shared by
/// `extract_symbols` and `defined_names` so both see the same definition set.
fn def_query_src(lang: Lang) -> &'static str {
    match lang {
        Lang::Ts => r#"
            (function_declaration name: (identifier) @name) @def
            (class_declaration name: (type_identifier) @name) @def
            (method_definition name: (property_identifier) @name) @def
            "#,
        Lang::Rust => r#"
            (function_item name: (identifier) @name) @def
            (struct_item name: (type_identifier) @name) @def
            (impl_item) @def
            "#,
        Lang::Python => r#"
            (function_definition name: (identifier) @name) @def
            (class_definition name: (identifier) @name) @def
            "#,
        Lang::C => r#"
            (function_definition declarator: (function_declarator declarator: (identifier) @name)) @def
            (function_definition declarator: (pointer_declarator declarator: (function_declarator declarator: (identifier) @name))) @def
            (struct_specifier name: (type_identifier) @name) @def
            (union_specifier name: (type_identifier) @name) @def
            (enum_specifier name: (type_identifier) @name) @def
            (type_definition declarator: (type_identifier) @name) @def
            "#,
        Lang::Cpp => r#"
            (function_definition declarator: (function_declarator declarator: (identifier) @name)) @def
            (function_definition declarator: (function_declarator declarator: (field_identifier) @name)) @def
            (function_definition declarator: (function_declarator declarator: (qualified_identifier) @name)) @def
            (class_specifier name: (type_identifier) @name) @def
            (struct_specifier name: (type_identifier) @name) @def
            (enum_specifier name: (type_identifier) @name) @def
            (namespace_definition name: (namespace_identifier) @name) @def
            "#,
        Lang::Java => r#"
            (class_declaration name: (identifier) @name) @def
            (interface_declaration name: (identifier) @name) @def
            (enum_declaration name: (identifier) @name) @def
            (method_declaration name: (identifier) @name) @def
            (constructor_declaration name: (identifier) @name) @def
            "#,
        Lang::CSharp => r#"
            (class_declaration name: (identifier) @name) @def
            (interface_declaration name: (identifier) @name) @def
            (struct_declaration name: (identifier) @name) @def
            (enum_declaration name: (identifier) @name) @def
            (method_declaration name: (identifier) @name) @def
            "#,
        Lang::Go => r#"
            (function_declaration name: (identifier) @name) @def
            (method_declaration name: (field_identifier) @name) @def
            (type_declaration (type_spec name: (type_identifier) @name)) @def
            "#,
    }
}

/// A definition name identifier with its byte span and enclosing definition
/// kind, so a verifier can rewrite an offending name in place.
#[derive(Debug, Clone)]
pub struct NameSpan {
    pub text: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub line: usize,
    /// Kind of the `@def` node (e.g. "function_item", "method_declaration").
    pub kind: String,
}

/// Byte-accurate spans of every definition *name*, reusing the same per-language
/// definition queries as `extract_symbols`.
pub fn defined_names(content: &str, lang: Lang) -> Vec<NameSpan> {
    let language = language_for(lang);
    let mut parser = Parser::new();
    if parser.set_language(&language).is_err() { return vec![]; }
    let tree = match parser.parse(content, None) { Some(t) => t, None => return vec![] };
    let query = match Query::new(&language, def_query_src(lang)) { Ok(q) => q, Err(_) => return vec![] };

    let name_idx = query.capture_index_for_name("name");
    let bytes = content.as_bytes();
    let mut cursor = QueryCursor::new();
    let mut out = Vec::new();

    let mut matches = cursor.matches(&query, tree.root_node(), bytes);
    while let Some(m) = matches.next() {
        let mut name_node = None;
        let mut def_kind = "";
        for cap in m.captures {
            if Some(cap.index) == name_idx {
                name_node = Some(cap.node);
            } else {
                def_kind = cap.node.kind();
            }
        }
        if let Some(n) = name_node {
            out.push(NameSpan {
                text: n.utf8_text(bytes).unwrap_or("").to_string(),
                start_byte: n.start_byte(),
                end_byte: n.end_byte(),
                line: n.start_position().row + 1,
                kind: def_kind.to_string(),
            });
        }
    }
    out
}

/// Lines (1-based) carrying an ERROR or MISSING node — the cheapest cascade
/// gate. An empty result means the candidate parses clean under the grammar.
/// A failed parse (no tree at all) is reported as a single error at line 1.
pub fn syntax_errors(content: &str, lang: Lang) -> Vec<usize> {
    let language = language_for(lang);
    let mut parser = Parser::new();
    if parser.set_language(&language).is_err() {
        return vec![];
    }
    let tree = match parser.parse(content, None) {
        Some(t) => t,
        None => return vec![1],
    };
    let root = tree.root_node();
    if !root.has_error() {
        return vec![];
    }
    // DFS, descending only into subtrees that actually contain an error so we
    // surface the shallowest offending positions without walking the whole AST.
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.is_error() || node.is_missing() {
            out.push(node.start_position().row + 1);
            continue;
        }
        let mut i = 0;
        while i < node.child_count() {
            if let Some(c) = node.child(i) {
                if c.has_error() {
                    stack.push(c);
                }
            }
            i += 1;
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Byte-accurate spans of every call-site callee identifier, reusing the same
/// per-language call queries as the code-graph extractor. Lets the reference
/// gate rewrite a drifted callee in place.
pub fn called_names(content: &str, lang: Lang) -> Vec<NameSpan> {
    capture_spans(content, lang, call_queries(lang), "callee")
}

/// Like `capture_query` but returns byte-accurate `NameSpan`s (with the capture
/// name recorded as `kind`) so callers can splice in-place corrections.
fn capture_spans(content: &str, lang: Lang, queries: &[&str], cap: &str) -> Vec<NameSpan> {
    let language = language_for(lang);
    let mut parser = Parser::new();
    if parser.set_language(&language).is_err() {
        return vec![];
    }
    let tree = match parser.parse(content, None) {
        Some(t) => t,
        None => return vec![],
    };
    let bytes = content.as_bytes();
    let mut out = Vec::new();
    for qsrc in queries {
        let query = match Query::new(&language, qsrc) {
            Ok(q) => q,
            Err(_) => continue,
        };
        let cap_idx = match query.capture_index_for_name(cap) {
            Some(i) => i,
            None => continue,
        };
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&query, tree.root_node(), bytes);
        while let Some(m) = matches.next() {
            for c in m.captures {
                if c.index == cap_idx {
                    out.push(NameSpan {
                        text: c.node.utf8_text(bytes).unwrap_or("").to_string(),
                        start_byte: c.node.start_byte(),
                        end_byte: c.node.end_byte(),
                        line: c.node.start_position().row + 1,
                        kind: cap.to_string(),
                    });
                }
            }
        }
    }
    out
}

/// Map a language to its tree-sitter grammar. These grammar crate versions all
/// expose a `language()`-style fn returning `Language`.
fn language_for(lang: Lang) -> tree_sitter::Language {
    match lang {
        Lang::Ts => tree_sitter_typescript::language_typescript(),
        Lang::Rust => tree_sitter_rust::language(),
        Lang::Python => tree_sitter_python::language(),
        Lang::C => tree_sitter_c::language(),
        Lang::Cpp => tree_sitter_cpp::language(),
        Lang::Java => tree_sitter_java::language(),
        Lang::CSharp => tree_sitter_c_sharp::language(),
        Lang::Go => tree_sitter_go::language(),
    }
}

/// Parse once, then run each query in `queries`, returning (text, start, end)
/// for every node bound to `cap`. Queries that fail to compile are skipped so a
/// single grammar mismatch never suppresses the rest.
fn capture_query(
    content: &str, lang: Lang, queries: &[&str], cap: &str,
) -> Vec<(String, usize, usize)> {
    let language = language_for(lang);
    let mut parser = Parser::new();
    if parser.set_language(&language).is_err() { return vec![]; }
    let tree = match parser.parse(content, None) { Some(t) => t, None => return vec![] };
    let bytes = content.as_bytes();
    let mut out = Vec::new();
    for qsrc in queries {
        let query = match Query::new(&language, qsrc) { Ok(q) => q, Err(_) => continue };
        let cap_idx = match query.capture_index_for_name(cap) { Some(i) => i, None => continue };
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&query, tree.root_node(), bytes);
        while let Some(m) = matches.next() {
            for c in m.captures {
                if c.index == cap_idx {
                    let text = c.node.utf8_text(bytes).unwrap_or("").to_string();
                    out.push((
                        text,
                        c.node.start_position().row + 1,
                        c.node.end_position().row + 1,
                    ));
                }
            }
        }
    }
    out
}

/// Imported module/path tokens (one per import site), used for "import" edges.
fn extract_imports(content: &str, lang: Lang) -> Vec<String> {
    capture_query(content, lang, import_queries(lang), "path")
        .into_iter()
        .map(|(t, _, _)| normalize_import(&t))
        .filter(|s| !s.is_empty())
        .collect()
}

/// Call sites as (callee, line), used to attribute "call" edges to a definition.
fn extract_calls(content: &str, lang: Lang) -> Vec<(String, usize)> {
    capture_query(content, lang, call_queries(lang), "callee")
        .into_iter()
        .map(|(t, start, _)| (t, start))
        .filter(|(t, _)| !t.is_empty())
        .collect()
}

/// The innermost symbol whose line range contains `line`.
fn enclosing_symbol(symbols: &[Symbol], line: usize) -> Option<&str> {
    symbols
        .iter()
        .filter(|s| s.start <= line && line <= s.end)
        .min_by_key(|s| s.end.saturating_sub(s.start))
        .map(|s| s.name.as_str())
}

/// Strip surrounding quotes / angle brackets from a captured import token.
fn normalize_import(raw: &str) -> String {
    raw.trim()
        .trim_matches(|c| c == '"' || c == '\'' || c == '<' || c == '>')
        .to_string()
}

/// Per-language import queries. Each pattern binds the module path to `@path`.
fn import_queries(lang: Lang) -> &'static [&'static str] {
    match lang {
        Lang::Ts => &[
            "(import_statement source: (string) @path)",
            "(export_statement source: (string) @path)",
        ],
        Lang::Rust => &["(use_declaration argument: (_) @path)"],
        Lang::Python => &[
            "(import_statement name: (dotted_name) @path)",
            "(import_statement name: (aliased_import name: (dotted_name) @path))",
            "(import_from_statement module_name: (dotted_name) @path)",
            "(import_from_statement module_name: (relative_import) @path)",
        ],
        Lang::C | Lang::Cpp => &[
            "(preproc_include path: (string_literal) @path)",
            "(preproc_include path: (system_lib_string) @path)",
        ],
        Lang::Java => &[
            "(import_declaration (scoped_identifier) @path)",
            "(import_declaration (identifier) @path)",
        ],
        Lang::CSharp => &[
            "(using_directive (qualified_name) @path)",
            "(using_directive (identifier) @path)",
        ],
        Lang::Go => &["(import_spec path: (interpreted_string_literal) @path)"],
    }
}

/// Per-language call queries. Each pattern binds the callee name to `@callee`.
fn call_queries(lang: Lang) -> &'static [&'static str] {
    match lang {
        Lang::Ts => &[
            "(call_expression function: (identifier) @callee)",
            "(call_expression function: (member_expression property: (property_identifier) @callee))",
        ],
        Lang::Rust => &[
            "(call_expression function: (identifier) @callee)",
            "(call_expression function: (scoped_identifier name: (identifier) @callee))",
            "(call_expression function: (field_expression field: (field_identifier) @callee))",
            "(macro_invocation macro: (identifier) @callee)",
        ],
        Lang::Python => &[
            "(call function: (identifier) @callee)",
            "(call function: (attribute attribute: (identifier) @callee))",
        ],
        Lang::C => &["(call_expression function: (identifier) @callee)"],
        Lang::Cpp => &[
            "(call_expression function: (identifier) @callee)",
            "(call_expression function: (field_expression field: (field_identifier) @callee))",
            "(call_expression function: (qualified_identifier name: (identifier) @callee))",
        ],
        Lang::Java => &[
            "(method_invocation name: (identifier) @callee)",
            "(object_creation_expression type: (type_identifier) @callee)",
        ],
        Lang::CSharp => &[
            "(invocation_expression function: (identifier) @callee)",
            "(invocation_expression function: (member_access_expression name: (identifier) @callee))",
        ],
        Lang::Go => &[
            "(call_expression function: (identifier) @callee)",
            "(call_expression function: (selector_expression field: (field_identifier) @callee))",
        ],
    }
}

fn is_probably_secret(path: &str) -> bool {
    const BANNED: &[&str] = &[".env", "id_rsa", ".pem", ".key", "secrets", ".pfx"];
    let lower = path.to_lowercase();
    BANNED.iter().any(|b| lower.contains(b))
}

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64
}