use crate::indexer::{called_names, defined_names, syntax_errors, Lang};
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

/// One unresolved issue from the cascade, tagged with the stage that produced
/// it so the message fed back to the model names the failing checker.
#[derive(Debug, Clone)]
pub struct CascadeViolation {
    pub checker: String,
    pub message: String,
    pub line: usize,
    /// Canonical target when the fault is deterministically fixable.
    pub suggestion: Option<String>,
}

/// Verdict of the composed cascade. Either it is clean (possibly after applying
/// deterministic corrections, in which case `corrected`/`fixes` are populated),
/// or it failed at `stage` with `violations` the model must address.
#[derive(Debug, Clone)]
pub struct CascadeReport {
    pub ok: bool,
    /// The failing stage when `!ok`; `None` when the candidate passed.
    pub stage: Option<String>,
    pub violations: Vec<CascadeViolation>,
    /// Human-readable description of each deterministic correction applied.
    pub fixes: Vec<String>,
    /// Final source after all deterministic corrections; `None` if unchanged.
    pub corrected: Option<String>,
}

/// Compose the deterministic checkers cheapest-first: **parse → names →
/// references**. The first stage with an unfixable violation short-circuits;
/// nothing more expensive runs once a cheaper gate has already rejected the
/// candidate. Auto-correctable drift (casing/style that lands on a registry
/// name) is fixed in place and the cascade re-runs from the top, so the final
/// verdict reflects fully-canonicalized source. Because every correction is an
/// in-place identifier rename, byte offsets across stages and reported line
/// numbers stay valid throughout.
///
/// (A `tests` stage — compile + run — is the natural next link, but it operates
/// on materialized, assembled code rather than an isolated function body, so it
/// belongs to a later build phase, not this per-symbol gate.)
pub fn run_cascade(code: &str, lang: Lang, registry: &[String]) -> CascadeReport {
    let mut current = code.to_string();
    let mut fixes: Vec<String> = Vec::new();

    // Bounded fixpoint: corrections only canonicalize identifiers (monotonic),
    // so a handful of passes is far more than enough to stabilize.
    for _ in 0..8 {
        // Stage 1 — parse (cheapest). No deterministic fix for bad syntax.
        let errs = syntax_errors(&current, lang);
        if !errs.is_empty() {
            let violations = errs
                .into_iter()
                .map(|line| CascadeViolation {
                    checker: "parse".to_string(),
                    message: format!("syntax error at line {line}"),
                    line,
                    suggestion: None,
                })
                .collect();
            return fail("parse", violations, fixes);
        }

        // Stage 2 — names. Auto-correct only when every offending definition is
        // fixable; a mixed/invented set is bounced back to the model.
        let names = verify_names(&current, lang, registry);
        if !names.ok {
            let all_fixable = names.violations.iter().all(|v| v.suggestion.is_some());
            if all_fixable {
                if let Some(fixed) = names.corrected {
                    for v in &names.violations {
                        if let Some(s) = &v.suggestion {
                            fixes.push(format!("renamed definition `{}` -> `{}`", v.found, s));
                        }
                    }
                    current = fixed;
                    continue; // re-run from the top on canonicalized source
                }
            }
            let violations = names
                .violations
                .into_iter()
                .map(|v| CascadeViolation {
                    message: match &v.suggestion {
                        Some(s) => format!("definition `{}` must be named `{}`", v.found, s),
                        None => format!("`{}` is not a planned symbol", v.found),
                    },
                    checker: "names".to_string(),
                    line: v.line,
                    suggestion: v.suggestion,
                })
                .collect();
            return fail("names", violations, fixes);
        }

        // Stage 3 — references. Calls whose callee normalizes onto a planned
        // name but drifted in casing are canonicalized; unknown calls (stdlib /
        // external) never match the registry and are left untouched, so this
        // stage only ever auto-corrects — it does not invent failures.
        let (rfixes, rcorrected) = check_calls(&current, lang, registry);
        if let Some(fixed) = rcorrected {
            fixes.extend(rfixes);
            current = fixed;
            continue;
        }

        // All stages clean.
        let corrected = if fixes.is_empty() { None } else { Some(current) };
        return CascadeReport { ok: true, stage: None, violations: vec![], fixes, corrected };
    }

    // Fixpoint not reached (not expected): accept what we have.
    let corrected = if fixes.is_empty() { None } else { Some(current) };
    CascadeReport { ok: true, stage: None, violations: vec![], fixes, corrected }
}

/// Assemble a failing report. Deterministic corrections that were applied before
/// the failing stage are discarded from `corrected` (the model regenerates from
/// its own output on a retry); `fixes` is retained purely as a record.
fn fail(stage: &str, violations: Vec<CascadeViolation>, fixes: Vec<String>) -> CascadeReport {
    CascadeReport {
        ok: false,
        stage: Some(stage.to_string()),
        violations,
        fixes,
        corrected: None,
    }
}

/// Reference gate. Returns (descriptions, corrected_source) when at least one
/// callee was canonicalized, else (empty, None).
fn check_calls(code: &str, lang: Lang, registry: &[String]) -> (Vec<String>, Option<String>) {
    let canon: HashSet<&str> = registry.iter().map(|s| s.as_str()).collect();
    let mut fixes = Vec::new();
    let mut edits: Vec<(usize, usize, String)> = Vec::new();
    for ns in called_names(code, lang) {
        if canon.contains(ns.text.as_str()) {
            continue; // already canonical
        }
        let normalized = to_snake_case(&ns.text);
        if normalized != ns.text && canon.contains(normalized.as_str()) {
            fixes.push(format!("canonicalized call `{}` -> `{}`", ns.text, normalized));
            edits.push((ns.start_byte, ns.end_byte, normalized));
        }
        // else: external/unknown callee — deliberately ignored.
    }
    if edits.is_empty() {
        (fixes, None)
    } else {
        (fixes, Some(apply_edits(code, &mut edits)))
    }
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

    // --- Cascade composition ---

    #[test]
    fn cascade_clean_passes_with_no_corrections() {
        let code = "fn verify_token(token: String) -> bool { true }";
        let r = run_cascade(code, Lang::Rust, &["verify_token".to_string()]);
        assert!(r.ok);
        assert!(r.fixes.is_empty());
        assert!(r.corrected.is_none());
    }

    #[test]
    fn cascade_parse_stage_short_circuits() {
        // Missing closing brace: the parse gate fires before names is consulted.
        let code = "fn verify_token() -> bool { true ";
        let r = run_cascade(code, Lang::Rust, &["verify_token".to_string()]);
        assert!(!r.ok);
        assert_eq!(r.stage.as_deref(), Some("parse"));
    }

    #[test]
    fn cascade_autocorrects_name_then_canonicalizes_call() {
        // Definition name drifts (verifyToken) and it calls a planned sibling by
        // a drifted name (checkScope). Both are fixed deterministically across
        // stages, so the cascade reaches a clean verdict with corrected source.
        let code = "fn verifyToken() -> bool { checkScope() }";
        let registry = vec!["verify_token".to_string(), "check_scope".to_string()];
        let r = run_cascade(code, Lang::Rust, &registry);
        assert!(r.ok, "expected clean after auto-correction, got {:?}", r);
        let corrected = r.corrected.as_deref().unwrap();
        assert!(corrected.contains("fn verify_token()"));
        assert!(corrected.contains("check_scope()"));
        assert!(!corrected.contains("verifyToken"));
        assert!(!corrected.contains("checkScope"));
    }

    #[test]
    fn cascade_names_stage_flags_invented_symbol() {
        let code = "fn helper() -> bool { true }";
        let r = run_cascade(code, Lang::Rust, &["verify_token".to_string()]);
        assert!(!r.ok);
        assert_eq!(r.stage.as_deref(), Some("names"));
        assert!(r.violations[0].suggestion.is_none());
    }

    #[test]
    fn cascade_ignores_unknown_external_calls() {
        // `println` (macro) and `format` are not planned; they must not be
        // flagged or rewritten — only drift toward a registry name is corrected.
        let code = "fn verify_token() -> bool { println(); true }";
        let r = run_cascade(code, Lang::Rust, &["verify_token".to_string()]);
        assert!(r.ok);
        assert!(r.corrected.is_none());
    }
}
