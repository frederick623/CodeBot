use crate::indexer::{defined_names, Lang};
use crate::orchestrator::to_snake_case;
use std::collections::HashSet;

/// A defined identifier that does not match the frozen registry.
#[derive(Debug, Clone)]
pub struct NameViolation {
    pub found: String,
    pub line: usize,
    /// Canonical name to rename to, when normalization maps `found` onto a
    /// registry entry. `None` means the symbol is not in the plan at all.
    pub suggestion: Option<String>,
}

/// Outcome of the deterministic name check.
#[derive(Debug, Clone)]
pub struct VerifyReport {
    pub ok: bool,
    pub violations: Vec<NameViolation>,
    /// Source with all auto-correctable renames applied; `None` if nothing was
    /// rewritten.
    pub corrected: Option<String>,
}

/// Function-like definition node kinds across the supported grammars. Only these
/// are enforced, since the planning registry holds functions.
fn is_function_kind(kind: &str) -> bool {
    matches!(
        kind,
        "function_declaration"
            | "function_definition"
            | "function_item"
            | "method_definition"
            | "method_declaration"
            | "constructor_declaration"
    )
}

/// Deterministic name-registry check. Every function definition must carry a
/// canonical name from `registry`. Names that differ only by casing/style are
/// auto-corrected via the same normalizer that produced the registry; genuinely
/// unknown names are reported without a fix. The model never decides a name —
/// this gate does.
pub fn verify_names(code: &str, lang: Lang, registry: &[String]) -> VerifyReport {
    let canon: HashSet<&str> = registry.iter().map(|s| s.as_str()).collect();
    let mut violations = Vec::new();
    // (start_byte, end_byte, replacement); applied right-to-left so earlier byte
    // offsets stay valid as we splice.
    let mut edits: Vec<(usize, usize, String)> = Vec::new();

    for ns in defined_names(code, lang) {
        if !is_function_kind(&ns.kind) {
            continue; // registry is function-only; ignore type/class defs
        }
        if canon.contains(ns.text.as_str()) {
            continue; // exact match — compliant
        }
        let normalized = to_snake_case(&ns.text);
        if normalized != ns.text && canon.contains(normalized.as_str()) {
            edits.push((ns.start_byte, ns.end_byte, normalized.clone()));
            violations.push(NameViolation {
                found: ns.text,
                line: ns.line,
                suggestion: Some(normalized),
            });
        } else {
            violations.push(NameViolation { found: ns.text, line: ns.line, suggestion: None });
        }
    }

    let corrected = if edits.is_empty() { None } else { Some(apply_edits(code, &mut edits)) };
    VerifyReport { ok: violations.is_empty(), violations, corrected }
}

/// Splice replacements into `code` by descending start offset so each edit's
/// byte range stays valid as earlier ones are applied.
fn apply_edits(code: &str, edits: &mut [(usize, usize, String)]) -> String {
    edits.sort_by(|a, b| b.0.cmp(&a.0));
    let mut out = code.to_string();
    for (start, end, repl) in edits.iter() {
        out.replace_range(*start..*end, repl);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_passes() {
        let code = "fn verify_token(token: String) -> bool { true }";
        let r = verify_names(code, Lang::Rust, &["verify_token".to_string()]);
        assert!(r.ok);
        assert!(r.violations.is_empty());
        assert!(r.corrected.is_none());
    }

    #[test]
    fn casing_drift_is_auto_corrected() {
        let code = "fn verifyToken() -> bool { true }";
        let r = verify_names(code, Lang::Rust, &["verify_token".to_string()]);
        assert!(!r.ok);
        assert_eq!(r.violations.len(), 1);
        assert_eq!(r.violations[0].suggestion.as_deref(), Some("verify_token"));
        assert_eq!(
            r.corrected.as_deref(),
            Some("fn verify_token() -> bool { true }")
        );
    }

    #[test]
    fn unknown_symbol_is_flagged_without_fix() {
        let code = "fn helper() {}";
        let r = verify_names(code, Lang::Rust, &["verify_token".to_string()]);
        assert!(!r.ok);
        assert_eq!(r.violations.len(), 1);
        assert!(r.violations[0].suggestion.is_none());
        assert!(r.corrected.is_none());
    }
}
