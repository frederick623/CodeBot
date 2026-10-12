//! Minimal Markdown renderer for egui.
//!
//! Local models emit plain CommonMark-ish text: headings, fenced code, pipe
//! tables, bullet/numbered lists and inline `code` / **bold** / *italic*. This
//! module parses that subset into blocks and draws them with native egui
//! widgets so answers no longer appear as raw `|---|` pipes and backticks.
//!
//! It also normalises typographic characters that egui's bundled fonts lack
//! (non‑breaking hyphens, thin spaces, variation selectors…) so they render as
//! their ASCII equivalents instead of "□" tofu boxes.
use eframe::egui::{self, text::LayoutJob, Color32, FontId, TextFormat};

/// Replace characters that egui's default fonts cannot draw with visually
/// equivalent ASCII so the UI never shows tofu boxes.
pub fn normalise(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            // hyphens / dashes
            '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2212}' => out.push('-'),
            '\u{2013}' => out.push_str("--"),
            '\u{2014}' | '\u{2015}' => out.push_str("---"),
            // spaces
            '\u{00A0}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => out.push(' '),
            // zero-width / joiners / variation selectors / BOM
            '\u{200B}'..='\u{200D}' | '\u{2060}' | '\u{FEFF}' | '\u{FE0E}' | '\u{FE0F}' => {}
            // quotes
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{2032}' => out.push('\''),
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{2033}' => out.push('"'),
            '\u{2026}' => out.push_str("..."),
            '\u{2022}' | '\u{25E6}' | '\u{2043}' => out.push('-'),
            '\u{2192}' => out.push_str("->"),
            '\u{2190}' => out.push_str("<-"),
            '\u{21D2}' => out.push_str("=>"),
            '\u{2713}' | '\u{2714}' => out.push_str("[x]"),
            '\u{2717}' | '\u{2718}' => out.push_str("[ ]"),
            _ => out.push(c),
        }
    }
    out
}

/// A parsed block-level element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Block {
    Heading(u8, String),
    Paragraph(String),
    Code { lang: String, body: String },
    Bullets(Vec<String>),
    Numbered(Vec<String>),
    Table { header: Vec<String>, rows: Vec<Vec<String>> },
    Rule,
}

fn is_table_sep(line: &str) -> bool {
    let t = line.trim();
    t.starts_with('|')
        && t.len() >= 3
        && t.chars().all(|c| matches!(c, '|' | '-' | ':' | ' '))
        && t.contains('-')
}

fn split_row(line: &str) -> Vec<String> {
    let t = line.trim();
    let t = t.strip_prefix('|').unwrap_or(t);
    let t = t.strip_suffix('|').unwrap_or(t);
    t.split('|').map(|c| c.trim().to_string()).collect()
}

fn list_item(line: &str) -> Option<(bool, String)> {
    let t = line.trim_start();
    if let Some(rest) = t.strip_prefix("- ").or_else(|| t.strip_prefix("* ")).or_else(|| t.strip_prefix("+ ")) {
        return Some((false, rest.to_string()));
    }
    let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
    if !digits.is_empty() {
        let rest = &t[digits.len()..];
        if let Some(rest) = rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") ")) {
            return Some((true, rest.to_string()));
        }
    }
    None
}

/// Parse markdown text into blocks.
pub fn parse(text: &str) -> Vec<Block> {
    let text = normalise(text);
    let lines: Vec<&str> = text.lines().collect();
    let mut blocks = Vec::new();
    let mut para: Vec<String> = Vec::new();
    let mut i = 0;

    fn flush(para: &mut Vec<String>, blocks: &mut Vec<Block>) {
        if !para.is_empty() {
            blocks.push(Block::Paragraph(para.join(" ")));
            para.clear();
        }
    }

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        if let Some(rest) = trimmed.strip_prefix("```") {
            flush(&mut para, &mut blocks);
            let lang = rest.trim().to_string();
            let mut body = Vec::new();
            i += 1;
            while i < lines.len() && !lines[i].trim().starts_with("```") {
                body.push(lines[i]);
                i += 1;
            }
            blocks.push(Block::Code { lang, body: body.join("\n") });
            i += 1;
            continue;
        }

        if trimmed.is_empty() {
            flush(&mut para, &mut blocks);
            i += 1;
            continue;
        }

        if trimmed.starts_with('#') {
            let level = trimmed.chars().take_while(|&c| c == '#').count();
            if level <= 6 && trimmed[level..].starts_with(' ') {
                flush(&mut para, &mut blocks);
                blocks.push(Block::Heading(level as u8, trimmed[level..].trim().to_string()));
                i += 1;
                continue;
            }
        }

        if trimmed == "---" || trimmed == "***" || trimmed == "___" {
            flush(&mut para, &mut blocks);
            blocks.push(Block::Rule);
            i += 1;
            continue;
        }

        if trimmed.starts_with('|') && i + 1 < lines.len() && is_table_sep(lines[i + 1]) {
            flush(&mut para, &mut blocks);
            let header = split_row(line);
            i += 2;
            let mut rows = Vec::new();
            while i < lines.len() && lines[i].trim().starts_with('|') {
                let mut row = split_row(lines[i]);
                row.resize(header.len(), String::new());
                rows.push(row);
                i += 1;
            }
            blocks.push(Block::Table { header, rows });
            continue;
        }

        if let Some((numbered, first)) = list_item(line) {
            flush(&mut para, &mut blocks);
            let mut items = vec![first];
            i += 1;
            while i < lines.len() {
                match list_item(lines[i]) {
                    Some((n, item)) if n == numbered => {
                        items.push(item);
                        i += 1;
                    }
                    // Continuation line (indented, non-empty) belongs to the previous item.
                    None if lines[i].starts_with("  ") && !lines[i].trim().is_empty() => {
                        if let Some(last) = items.last_mut() {
                            last.push(' ');
                            last.push_str(lines[i].trim());
                        }
                        i += 1;
                    }
                    _ => break,
                }
            }
            blocks.push(if numbered { Block::Numbered(items) } else { Block::Bullets(items) });
            continue;
        }

        para.push(trimmed.to_string());
        i += 1;
    }
    flush(&mut para, &mut blocks);
    blocks
}

/// Inline styling: `code`, **bold**, *italic* / _italic_.
fn inline_job(text: &str, base: FontId, color: Color32, strong: Color32, code_bg: Color32) -> LayoutJob {
    let mut job = LayoutJob::default();
    job.wrap.max_width = f32::INFINITY;
    let mono = FontId::monospace(base.size * 0.92);
    let plain = TextFormat { font_id: base.clone(), color, ..Default::default() };
    // egui ships no bold face, so "bold" is the stronger text colour plus a
    // slightly larger size.
    let bold = TextFormat {
        font_id: FontId::new(base.size * 1.04, base.family.clone()),
        color: strong,
        ..Default::default()
    };
    let italic = TextFormat { font_id: base.clone(), color, italics: true, ..Default::default() };
    let code = TextFormat { font_id: mono, color, background: code_bg, ..Default::default() };

    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    let mut buf = String::new();
    let mut in_bold = false;
    let mut in_italic = false;
    let flush = |buf: &mut String, job: &mut LayoutJob, b: bool, it: bool| {
        if buf.is_empty() {
            return;
        }
        let fmt = if b { bold.clone() } else if it { italic.clone() } else { plain.clone() };
        job.append(buf, 0.0, fmt);
        buf.clear();
    };
    while i < chars.len() {
        let c = chars[i];
        if c == '`' {
            if let Some(end) = chars[i + 1..].iter().position(|&x| x == '`') {
                flush(&mut buf, &mut job, in_bold, in_italic);
                let s: String = chars[i + 1..i + 1 + end].iter().collect();
                job.append(&s, 0.0, code.clone());
                i += end + 2;
                continue;
            }
        }
        if c == '*' && i + 1 < chars.len() && chars[i + 1] == '*' {
            flush(&mut buf, &mut job, in_bold, in_italic);
            in_bold = !in_bold;
            i += 2;
            continue;
        }
        let word_boundary = i == 0 || !chars[i - 1].is_alphanumeric();
        if (c == '*' || (c == '_' && (word_boundary || in_italic)))
            && (in_italic || (i + 1 < chars.len() && !chars[i + 1].is_whitespace()))
        {
            flush(&mut buf, &mut job, in_bold, in_italic);
            in_italic = !in_italic;
            i += 1;
            continue;
        }
        buf.push(c);
        i += 1;
    }
    flush(&mut buf, &mut job, in_bold, in_italic);
    job
}

fn inline_label(ui: &mut egui::Ui, text: &str, size: f32) {
    let color = ui.visuals().text_color();
    let strong = ui.visuals().strong_text_color();
    let bg = ui.visuals().code_bg_color;
    let mut job = inline_job(text, FontId::proportional(size), color, strong, bg);
    job.wrap.max_width = ui.available_width().max(50.0);
    ui.add(egui::Label::new(job).wrap());
}

/// Render markdown `text` into `ui`.
pub fn render(ui: &mut egui::Ui, text: &str) {
    let base = egui::TextStyle::Body.resolve(ui.style()).size;
    for block in parse(text) {
        match block {
            Block::Heading(level, t) => {
                let size = match level {
                    1 => base * 1.6,
                    2 => base * 1.35,
                    3 => base * 1.15,
                    _ => base * 1.05,
                };
                ui.add_space(4.0);
                ui.label(egui::RichText::new(t).strong().size(size));
                ui.add_space(2.0);
            }
            Block::Paragraph(t) => {
                inline_label(ui, &t, base);
                ui.add_space(4.0);
            }
            Block::Rule => {
                ui.separator();
            }
            Block::Code { lang, body } => {
                let fill = ui.visuals().code_bg_color;
                egui::Frame::new()
                    .fill(fill)
                    .inner_margin(6.0)
                    .corner_radius(4.0)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        if !lang.is_empty() {
                            ui.small(lang);
                        }
                        egui::ScrollArea::horizontal()
                            .id_salt(ui.next_auto_id())
                            .show(ui, |ui| {
                                ui.add(
                                    egui::Label::new(egui::RichText::new(body).monospace())
                                        .wrap_mode(egui::TextWrapMode::Extend),
                                );
                            });
                    });
                ui.add_space(4.0);
            }
            Block::Bullets(items) => {
                for item in items {
                    ui.horizontal_top(|ui| {
                        ui.add_space(8.0);
                        ui.label("•");
                        inline_label(ui, &item, base);
                    });
                }
                ui.add_space(4.0);
            }
            Block::Numbered(items) => {
                for (n, item) in items.iter().enumerate() {
                    ui.horizontal_top(|ui| {
                        ui.add_space(8.0);
                        ui.label(format!("{}.", n + 1));
                        inline_label(ui, item, base);
                    });
                }
                ui.add_space(4.0);
            }
            Block::Table { header, rows } => {
                let cols = header.len().max(1);
                egui::Frame::group(ui.style()).inner_margin(4.0).show(ui, |ui| {
                    egui::Grid::new(ui.next_auto_id())
                        .num_columns(cols)
                        .striped(true)
                        .spacing([14.0, 4.0])
                        .show(ui, |ui| {
                            for h in &header {
                                ui.label(egui::RichText::new(h).strong());
                            }
                            ui.end_row();
                            for row in &rows {
                                for cell in row {
                                    let color = ui.visuals().text_color();
                                    let strong = ui.visuals().strong_text_color();
                                    let bg = ui.visuals().code_bg_color;
                                    let mut job = inline_job(cell, FontId::proportional(base), color, strong, bg);
                                    job.wrap.max_width = 360.0;
                                    ui.add(egui::Label::new(job));
                                }
                                ui.end_row();
                            }
                        });
                });
                ui.add_space(4.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalises_unsupported_glyphs() {
        assert_eq!(normalise("context\u{2011}window"), "context-window");
        assert_eq!(normalise("a\u{2009}b\u{FE0F}"), "a b");
        assert_eq!(normalise("x \u{2192} y"), "x -> y");
    }

    #[test]
    fn parses_table_and_lists() {
        let md = "# Title\n\nSome *text* here.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n| 3 |\n\n- one\n- two\n  more\n\n1. first\n2. second\n\n```rust\nfn x() {}\n```";
        let blocks = parse(md);
        assert_eq!(blocks[0], Block::Heading(1, "Title".into()));
        assert_eq!(blocks[1], Block::Paragraph("Some *text* here.".into()));
        assert_eq!(
            blocks[2],
            Block::Table {
                header: vec!["A".into(), "B".into()],
                rows: vec![vec!["1".into(), "2".into()], vec!["3".into(), String::new()]],
            }
        );
        assert_eq!(blocks[3], Block::Bullets(vec!["one".into(), "two more".into()]));
        assert_eq!(blocks[4], Block::Numbered(vec!["first".into(), "second".into()]));
        assert_eq!(blocks[5], Block::Code { lang: "rust".into(), body: "fn x() {}".into() });
    }

    #[test]
    fn unterminated_code_fence_is_tolerated() {
        let blocks = parse("```\nabc");
        assert_eq!(blocks, vec![Block::Code { lang: String::new(), body: "abc".into() }]);
    }

    #[test]
    fn inline_job_splits_code_and_bold() {
        let job = inline_job("use `foo` and **bar** now", FontId::proportional(14.0), Color32::WHITE, Color32::WHITE, Color32::BLACK);
        assert_eq!(job.text, "use foo and bar now");
        assert_eq!(job.sections.len(), 5);
        assert_eq!(job.sections[1].format.background, Color32::BLACK);
    }
}
