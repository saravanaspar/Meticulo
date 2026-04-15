use crate::app::{App, Panel, SettingsField, Status};
use crate::hardware::format_bytes;
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect, Alignment},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap},
    Frame, Terminal,
};
use std::io::{self, Stdout};

// ── Color Palette ──────────────────────────────────────────
const BG: Color = Color::Rgb(18, 18, 24);
const SURFACE: Color = Color::Rgb(28, 28, 38);
const SURFACE_BRIGHT: Color = Color::Rgb(38, 38, 52);
const BORDER: Color = Color::Rgb(58, 58, 78);
const BORDER_ACTIVE: Color = Color::Rgb(120, 90, 255);
const TEXT: Color = Color::Rgb(220, 220, 235);
const TEXT_DIM: Color = Color::Rgb(120, 120, 150);
const TEXT_MUTED: Color = Color::Rgb(80, 80, 110);
const ACCENT: Color = Color::Rgb(120, 90, 255);
const ACCENT_DIM: Color = Color::Rgb(80, 60, 180);
const USER_MSG: Color = Color::Rgb(90, 160, 255);
const ASSISTANT_MSG: Color = Color::Rgb(180, 140, 255);
const ERROR_COLOR: Color = Color::Rgb(255, 90, 90);
const SUCCESS: Color = Color::Rgb(90, 220, 140);
const WARNING: Color = Color::Rgb(255, 180, 50);
const SEARCH_BG: Color = Color::Rgb(35, 35, 50);

pub type Tui = Terminal<CrosstermBackend<Stdout>>;

pub fn setup_terminal() -> Result<Tui, Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;
    Ok(terminal)
}

pub fn restore_terminal(mut terminal: Tui) -> Result<(), Box<dyn std::error::Error>> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

pub fn draw(f: &mut Frame, app: &App) {
    let size = f.area();

    // Background fill
    let bg_block = Block::default().style(Style::default().bg(BG));
    f.render_widget(bg_block, size);

    // Main layout: header, body, input, status
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(6),    // Chat area
            Constraint::Length(3), // Input
            Constraint::Length(1), // Status bar
        ])
        .split(size);

    draw_header(f, app, main_layout[0]);
    draw_chat(f, app, main_layout[1]);
    draw_input(f, app, main_layout[2]);
    draw_status_bar(f, app, main_layout[3]);

    // Overlays
    match app.active_panel {
        Panel::ModelSelector => draw_model_selector(f, app, size),
        Panel::Recommendations => draw_recommendations(f, app, size),
        Panel::PullModel => draw_pull_model(f, app, size),
        Panel::Settings => draw_settings(f, app, size),
        Panel::Chat => {}
    }
}

fn draw_header(f: &mut Frame, app: &App, area: Rect) {
    let header_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(22), // Logo
            Constraint::Min(10),   // Model info
            Constraint::Length(16), // Status indicator
        ])
        .split(area);

    // Logo
    let logo = Paragraph::new(Line::from(vec![
        Span::styled(" ◆ ", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled("METICULO", Style::default().fg(TEXT).add_modifier(Modifier::BOLD)),
        Span::styled(" edge", Style::default().fg(ACCENT_DIM)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(BORDER))
            .style(Style::default().bg(SURFACE)),
    );
    f.render_widget(logo, header_layout[0]);

    // Model info
    let model_text = match &app.current_model {
        Some(m) => Line::from(vec![
            Span::styled(" Model: ", Style::default().fg(TEXT_DIM)),
            Span::styled(m, Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("  T={:.1} K={} P={:.2}", app.temperature, app.top_k, app.top_p),
                Style::default().fg(TEXT_MUTED),
            ),
        ]),
        None => Line::from(vec![
            Span::styled(" No model loaded ", Style::default().fg(WARNING)),
            Span::styled(" [Ctrl+K] select  [Ctrl+P] pull", Style::default().fg(TEXT_MUTED)),
        ]),
    };
    let model_para = Paragraph::new(model_text).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(BORDER))
            .style(Style::default().bg(SURFACE)),
    );
    f.render_widget(model_para, header_layout[1]);

    // Connection status dot
    let (dot, dot_color) = if app.server_online {
        ("● Online", SUCCESS)
    } else {
        ("○ Offline", ERROR_COLOR)
    };
    let status_dot = Paragraph::new(Line::from(vec![
        Span::styled(format!(" {dot}"), Style::default().fg(dot_color)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(BORDER))
            .style(Style::default().bg(SURFACE)),
    );
    f.render_widget(status_dot, header_layout[2]);
}

fn draw_chat(f: &mut Frame, app: &App, area: Rect) {
    let chat_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if app.active_panel == Panel::Chat {
            BORDER_ACTIVE
        } else {
            BORDER
        }))
        .style(Style::default().bg(BG));
    let inner = chat_block.inner(area);
    f.render_widget(chat_block, area);

    if app.messages.is_empty() && app.streaming_text.is_empty() {
        draw_welcome(f, app, inner);
        return;
    }

    // Build chat lines
    let mut lines: Vec<Line> = Vec::new();
    let width = inner.width.saturating_sub(4) as usize;

    for msg in &app.messages {
        let (prefix, prefix_style, text_style) = match msg.role.as_str() {
            "user" => (
                " ▸ You",
                Style::default().fg(USER_MSG).add_modifier(Modifier::BOLD),
                Style::default().fg(TEXT),
            ),
            "assistant" => (
                " ◆ Meticulo",
                Style::default().fg(ASSISTANT_MSG).add_modifier(Modifier::BOLD),
                Style::default().fg(TEXT),
            ),
            _ => (
                " ◇ System",
                Style::default().fg(TEXT_DIM).add_modifier(Modifier::BOLD),
                Style::default().fg(TEXT_DIM),
            ),
        };

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(prefix, prefix_style)));

        // Word-wrap message content
        for wrapped in word_wrap(&msg.content, width) {
            lines.push(Line::from(Span::styled(
                format!("   {wrapped}"),
                text_style,
            )));
        }
    }

    // Currently streaming text
    if !app.streaming_text.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            " ◆ Meticulo",
            Style::default()
                .fg(ASSISTANT_MSG)
                .add_modifier(Modifier::BOLD),
        )));
        for wrapped in word_wrap(&app.streaming_text, width) {
            lines.push(Line::from(Span::styled(
                format!("   {wrapped}"),
                Style::default().fg(TEXT),
            )));
        }
        lines.push(Line::from(Span::styled(
            "   ▍",
            Style::default().fg(ACCENT).add_modifier(Modifier::SLOW_BLINK),
        )));
    }

    let total_lines = lines.len() as u16;
    let visible = inner.height;

    // Auto-scroll to bottom
    let scroll = if matches!(app.status, Status::Generating) || app.scroll_offset == 0 {
        total_lines.saturating_sub(visible)
    } else {
        total_lines
            .saturating_sub(visible)
            .saturating_sub(app.scroll_offset)
    };

    let chat_text = Paragraph::new(Text::from(lines))
        .scroll((scroll, 0))
        .wrap(Wrap { trim: false });
    f.render_widget(chat_text, inner);
}

fn draw_welcome(f: &mut Frame, app: &App, area: Rect) {
    let hw_line = app.hw.summary_line();

    let lines = vec![
        Line::from(""),
        Line::from(""),
        Line::from(Span::styled(
            "◆  M E T I C U L O",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Raw LLM performance. Zero overhead.",
            Style::default().fg(TEXT_DIM),
        )),
        Line::from(""),
        Line::from(""),
        Line::from(vec![
            Span::styled("  ⚙  ", Style::default().fg(TEXT_MUTED)),
            Span::styled(&hw_line, Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "─────────────────────────────────────",
            Style::default().fg(SURFACE_BRIGHT),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Ctrl+K  ", Style::default().fg(ACCENT)),
            Span::styled("Select model      ", Style::default().fg(TEXT_DIM)),
            Span::styled("  Ctrl+P  ", Style::default().fg(ACCENT)),
            Span::styled("Pull new model", Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(vec![
            Span::styled("  Ctrl+R  ", Style::default().fg(ACCENT)),
            Span::styled("Recommended models", Style::default().fg(TEXT_DIM)),
            Span::styled("  Ctrl+S  ", Style::default().fg(ACCENT)),
            Span::styled("Settings", Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(vec![
            Span::styled("  Ctrl+N  ", Style::default().fg(ACCENT)),
            Span::styled("New conversation  ", Style::default().fg(TEXT_DIM)),
            Span::styled("  Ctrl+L  ", Style::default().fg(ACCENT)),
            Span::styled("Clear chat", Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(vec![
            Span::styled("  Enter   ", Style::default().fg(ACCENT)),
            Span::styled("Send message      ", Style::default().fg(TEXT_DIM)),
            Span::styled("  PgUp/Dn ", Style::default().fg(ACCENT)),
            Span::styled("Scroll chat", Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            format!(
                "  {} models available  │  port 22434",
                app.pull_catalog.len()
            ),
            Style::default().fg(TEXT_MUTED),
        )),
    ];

    let welcome = Paragraph::new(Text::from(lines)).alignment(Alignment::Center);
    f.render_widget(welcome, area);
}

fn draw_input(f: &mut Frame, app: &App, area: Rect) {
    let is_active = app.active_panel == Panel::Chat;
    let border_color = if is_active { BORDER_ACTIVE } else { BORDER };

    let (input_text, input_style) = if app.input.is_empty() && is_active {
        (
            "Type a message...".to_string(),
            Style::default().fg(TEXT_MUTED),
        )
    } else {
        (app.input.clone(), Style::default().fg(TEXT))
    };

    let prompt = if matches!(app.status, Status::Generating) {
        Span::styled(" ⟳ ", Style::default().fg(ACCENT).add_modifier(Modifier::SLOW_BLINK))
    } else {
        Span::styled(" ▸ ", Style::default().fg(ACCENT))
    };

    let input_line = Line::from(vec![
        prompt,
        Span::styled(input_text, input_style),
    ]);

    let input_box = Paragraph::new(input_line).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color))
            .style(Style::default().bg(SURFACE)),
    );
    f.render_widget(input_box, area);

    // Show cursor
    if is_active {
        f.set_cursor_position((
            area.x + 4 + app.cursor_pos as u16,
            area.y + 1,
        ));
    }
}

fn draw_status_bar(f: &mut Frame, app: &App, area: Rect) {
    let (status_text, status_color) = match &app.status {
        Status::Ready => ("Ready", SUCCESS),
        Status::Generating => ("Generating...", ACCENT),
        Status::Loading(m) => return draw_status_custom(f, area, &format!("⟳ {m}..."), WARNING),
        Status::Error(e) => return draw_status_custom(f, area, &format!("✗ {e}"), ERROR_COLOR),
        Status::Offline => ("○ Server offline — starting...", ERROR_COLOR),
    };

    let bar = Paragraph::new(Line::from(vec![
        Span::styled(" ", Style::default()),
        Span::styled(status_text, Style::default().fg(status_color)),
        Span::styled(
            format!(
                "  │  {} models  │  {}  │  VRAM={}  │  [Ctrl+C] quit ",
                app.available_models.len(),
                app.current_model.as_deref().unwrap_or("none"),
                app.running_model_size.map(format_bytes).unwrap_or_else(|| "n/a".to_string())
            ),
            Style::default().fg(TEXT_MUTED),
        ),
    ]))
    .style(Style::default().bg(SURFACE));
    f.render_widget(bar, area);
}

fn draw_status_custom(f: &mut Frame, area: Rect, text: &str, color: Color) {
    let bar = Paragraph::new(Line::from(Span::styled(
        format!(" {text}"),
        Style::default().fg(color),
    )))
    .style(Style::default().bg(SURFACE));
    f.render_widget(bar, area);
}

// ── Model Selector with Search ─────────────────────────────

fn draw_model_selector(f: &mut Frame, app: &App, area: Rect) {
    let popup = centered_rect(55, 65, area);
    f.render_widget(Clear, popup);

    let block = Block::default()
        .title(Line::from(vec![
            Span::styled(" ◆ ", Style::default().fg(ACCENT)),
            Span::styled("Select Model ", Style::default().fg(TEXT).add_modifier(Modifier::BOLD)),
            Span::styled("(type to search) ", Style::default().fg(TEXT_MUTED)),
        ]))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER_ACTIVE))
        .style(Style::default().bg(SURFACE));

    let inner = block.inner(popup);
    f.render_widget(block, popup);

    // Search bar
    let search_area = Rect { height: 1, ..inner };
    let list_area = Rect {
        y: inner.y + 2,
        height: inner.height.saturating_sub(2),
        ..inner
    };

    let search_text = if app.model_filter.is_empty() {
        Span::styled("  🔍 Search models...", Style::default().fg(TEXT_MUTED))
    } else {
        Span::styled(format!("  🔍 {}", app.model_filter), Style::default().fg(TEXT))
    };
    let search_bar = Paragraph::new(Line::from(search_text))
        .style(Style::default().bg(SEARCH_BG));
    f.render_widget(search_bar, search_area);

    // Separator
    let sep = Paragraph::new(Line::from(Span::styled(
        "─".repeat(inner.width as usize),
        Style::default().fg(BORDER),
    )));
    f.render_widget(sep, Rect { y: inner.y + 1, height: 1, ..inner });

    if app.model_filtered.is_empty() {
        let empty_text = if app.available_models.is_empty() {
            " No models downloaded. [Ctrl+P] to pull a model"
        } else {
            " No matches"
        };
        let empty = Paragraph::new(Line::from(Span::styled(
            empty_text,
            Style::default().fg(TEXT_DIM),
        )));
        f.render_widget(empty, list_area);
        return;
    }

    // Calculate scroll offset for visible window
    let visible_height = list_area.height as usize;
    let scroll_start = if app.model_selector_idx >= visible_height {
        app.model_selector_idx - visible_height + 1
    } else {
        0
    };

    let items: Vec<ListItem> = app
        .model_filtered
        .iter()
        .enumerate()
        .skip(scroll_start)
        .take(visible_height)
        .map(|(i, &real_idx)| {
            let name = &app.available_models[real_idx];
            let model_meta = app.model_catalog.iter().find(|m| m.name == *name);
            let size_txt = model_meta
                .map(|m| format_bytes(m.size))
                .unwrap_or_else(|| "n/a".to_string());
            let modified_txt = model_meta
                .and_then(|m| {
                    if m.modified_at.is_empty() { None }
                    else { Some(m.modified_at.chars().take(10).collect::<String>()) }
                })
                .unwrap_or_default();

            let is_current = app.current_model.as_deref() == Some(name);
            let is_selected = i == app.model_selector_idx;

            let marker = if is_current { "●" } else { "○" };
            let style = if is_selected {
                Style::default()
                    .fg(ACCENT)
                    .bg(SURFACE_BRIGHT)
                    .add_modifier(Modifier::BOLD)
            } else if is_current {
                Style::default().fg(SUCCESS)
            } else {
                Style::default().fg(TEXT)
            };

            ListItem::new(Line::from(Span::styled(
                format!("  {marker} {name:<28} {size_txt:>8}  {modified_txt}"),
                style,
            )))
        })
        .collect();

    let list = List::new(items);
    f.render_widget(list, list_area);

    // Footer hints
    if list_area.height > 2 {
        let footer_y = list_area.y + list_area.height.saturating_sub(1);
        let footer = Paragraph::new(Line::from(vec![
            Span::styled(" Enter", Style::default().fg(ACCENT)),
            Span::styled("=load  ", Style::default().fg(TEXT_MUTED)),
            Span::styled("Ctrl+D", Style::default().fg(ACCENT)),
            Span::styled("=delete  ", Style::default().fg(TEXT_MUTED)),
            Span::styled("Esc", Style::default().fg(ACCENT)),
            Span::styled("=close", Style::default().fg(TEXT_MUTED)),
        ]))
        .style(Style::default().bg(SURFACE));
        f.render_widget(footer, Rect {
            y: footer_y,
            height: 1,
            ..list_area
        });
    }
}

// ── Pull Model with Catalog Search ─────────────────────────

fn draw_pull_model(f: &mut Frame, app: &App, area: Rect) {
    let popup = centered_rect(60, 70, area);
    f.render_widget(Clear, popup);

    let block = Block::default()
        .title(Line::from(vec![
            Span::styled(" ⬇ ", Style::default().fg(WARNING)),
            Span::styled("Pull Model ", Style::default().fg(TEXT).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("({} available) ", app.pull_catalog.len()),
                Style::default().fg(TEXT_MUTED),
            ),
        ]))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER_ACTIVE))
        .style(Style::default().bg(SURFACE));

    let inner = block.inner(popup);
    f.render_widget(block, popup);

    // Search bar
    let search_area = Rect { height: 1, ..inner };
    let list_area = Rect {
        y: inner.y + 2,
        height: inner.height.saturating_sub(3),
        ..inner
    };

    let search_text = if app.pull_filter.is_empty() {
        Span::styled(
            "  🔍 Type to search (e.g. qwen, llama, gemma, deepseek...)",
            Style::default().fg(TEXT_MUTED),
        )
    } else {
        Span::styled(
            format!("  🔍 {} ({} matches)", app.pull_filter, app.pull_filtered.len()),
            Style::default().fg(TEXT),
        )
    };
    let search_bar = Paragraph::new(Line::from(search_text))
        .style(Style::default().bg(SEARCH_BG));
    f.render_widget(search_bar, search_area);

    // Separator
    let sep = Paragraph::new(Line::from(Span::styled(
        "─".repeat(inner.width as usize),
        Style::default().fg(BORDER),
    )));
    f.render_widget(sep, Rect { y: inner.y + 1, height: 1, ..inner });

    if app.pull_filtered.is_empty() {
        let empty = Paragraph::new(vec![
            Line::from(""),
            Line::from(Span::styled(" No matching models", Style::default().fg(TEXT_DIM))),
        ]);
        f.render_widget(empty, list_area);
        return;
    }

    // Calculate scroll offset for visible window
    let visible_height = list_area.height as usize;
    let scroll_start = if app.pull_selector_idx >= visible_height {
        app.pull_selector_idx - visible_height + 1
    } else {
        0
    };

    let items: Vec<ListItem> = app
        .pull_filtered
        .iter()
        .enumerate()
        .skip(scroll_start)
        .take(visible_height)
        .map(|(i, name)| {
            let is_selected = i == app.pull_selector_idx;
            let is_downloaded = app.available_models.iter().any(|m| m == name);

            let marker = if is_downloaded { "✓" } else { " " };
            let style = if is_selected {
                Style::default()
                    .fg(ACCENT)
                    .bg(SURFACE_BRIGHT)
                    .add_modifier(Modifier::BOLD)
            } else if is_downloaded {
                Style::default().fg(SUCCESS)
            } else {
                Style::default().fg(TEXT)
            };

            ListItem::new(Line::from(Span::styled(
                format!("  {marker} {name}"),
                style,
            )))
        })
        .collect();

    let list = List::new(items);
    f.render_widget(list, list_area);

    // Footer
    let footer_y = inner.y + inner.height.saturating_sub(1);
    let footer = Paragraph::new(Line::from(vec![
        Span::styled(" Enter", Style::default().fg(ACCENT)),
        Span::styled("=pull & load  ", Style::default().fg(TEXT_MUTED)),
        Span::styled("↑↓", Style::default().fg(ACCENT)),
        Span::styled("=navigate  ", Style::default().fg(TEXT_MUTED)),
        Span::styled("Esc", Style::default().fg(ACCENT)),
        Span::styled("=close", Style::default().fg(TEXT_MUTED)),
    ]))
    .style(Style::default().bg(SURFACE));
    f.render_widget(footer, Rect {
        y: footer_y,
        height: 1,
        ..inner
    });
}

// ── Settings Panel ─────────────────────────────────────────

fn draw_settings(f: &mut Frame, app: &App, area: Rect) {
    let popup = centered_rect(50, 50, area);
    f.render_widget(Clear, popup);

    let block = Block::default()
        .title(Line::from(vec![
            Span::styled(" ⚙ ", Style::default().fg(ACCENT)),
            Span::styled("Inference Settings ", Style::default().fg(TEXT).add_modifier(Modifier::BOLD)),
        ]))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER_ACTIVE))
        .style(Style::default().bg(SURFACE));

    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));

    // Temperature
    let temp_selected = app.settings_field == SettingsField::Temperature;
    let temp_marker = if temp_selected { "▸" } else { " " };
    let temp_style = if temp_selected {
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(TEXT)
    };
    let temp_bar = make_slider_bar(app.temperature, 0.0, 2.0, 20);
    lines.push(Line::from(vec![
        Span::styled(format!("  {temp_marker} Temperature:   "), temp_style),
        Span::styled(format!("{:.1}", app.temperature), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
    ]));
    lines.push(Line::from(vec![
        Span::styled("      ", Style::default()),
        Span::styled(temp_bar, Style::default().fg(if temp_selected { ACCENT } else { TEXT_DIM })),
        Span::styled("  ◂ ▸ adjust", Style::default().fg(TEXT_MUTED)),
    ]));
    lines.push(Line::from(""));

    // Top-K
    let topk_selected = app.settings_field == SettingsField::TopK;
    let topk_marker = if topk_selected { "▸" } else { " " };
    let topk_style = if topk_selected {
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(TEXT)
    };
    let topk_bar = make_slider_bar(app.top_k as f32, 0.0, 200.0, 20);
    lines.push(Line::from(vec![
        Span::styled(format!("  {topk_marker} Top-K:         "), topk_style),
        Span::styled(format!("{}", app.top_k), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
    ]));
    lines.push(Line::from(vec![
        Span::styled("      ", Style::default()),
        Span::styled(topk_bar, Style::default().fg(if topk_selected { ACCENT } else { TEXT_DIM })),
        Span::styled("  ◂ ▸ adjust", Style::default().fg(TEXT_MUTED)),
    ]));
    lines.push(Line::from(""));

    // Top-P
    let topp_selected = app.settings_field == SettingsField::TopP;
    let topp_marker = if topp_selected { "▸" } else { " " };
    let topp_style = if topp_selected {
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(TEXT)
    };
    let topp_bar = make_slider_bar(app.top_p, 0.0, 1.0, 20);
    lines.push(Line::from(vec![
        Span::styled(format!("  {topp_marker} Top-P:         "), topp_style),
        Span::styled(format!("{:.2}", app.top_p), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
    ]));
    lines.push(Line::from(vec![
        Span::styled("      ", Style::default()),
        Span::styled(topp_bar, Style::default().fg(if topp_selected { ACCENT } else { TEXT_DIM })),
        Span::styled("  ◂ ▸ adjust", Style::default().fg(TEXT_MUTED)),
    ]));
    lines.push(Line::from(""));

    // System Prompt
    let sys_selected = app.settings_field == SettingsField::SystemPrompt;
    let sys_marker = if sys_selected { "▸" } else { " " };
    let sys_style = if sys_selected {
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(TEXT)
    };
    lines.push(Line::from(vec![
        Span::styled(format!("  {sys_marker} System Prompt: "), sys_style),
        if app.settings_editing_sysprompt {
            Span::styled("[editing]", Style::default().fg(WARNING))
        } else {
            Span::styled("[Enter to edit]", Style::default().fg(TEXT_MUTED))
        },
    ]));

    let prompt_display = if app.system_prompt.is_empty() {
        "(none)".to_string()
    } else {
        let max = 50;
        if app.system_prompt.len() > max {
            format!("{}...", &app.system_prompt[..max])
        } else {
            app.system_prompt.clone()
        }
    };
    let prompt_style = if app.settings_editing_sysprompt {
        Style::default().fg(TEXT).bg(SEARCH_BG)
    } else {
        Style::default().fg(TEXT_DIM)
    };
    lines.push(Line::from(vec![
        Span::styled("      ", Style::default()),
        Span::styled(prompt_display, prompt_style),
        if app.settings_editing_sysprompt {
            Span::styled("▍", Style::default().fg(ACCENT).add_modifier(Modifier::SLOW_BLINK))
        } else {
            Span::styled("", Style::default())
        },
    ]));

    lines.push(Line::from(""));
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("  ↑↓", Style::default().fg(ACCENT)),
        Span::styled("=select  ", Style::default().fg(TEXT_MUTED)),
        Span::styled("◂▸", Style::default().fg(ACCENT)),
        Span::styled("=adjust  ", Style::default().fg(TEXT_MUTED)),
        Span::styled("Esc", Style::default().fg(ACCENT)),
        Span::styled("=close", Style::default().fg(TEXT_MUTED)),
    ]));

    let para = Paragraph::new(Text::from(lines));
    f.render_widget(para, inner);
}

fn make_slider_bar(value: f32, min: f32, max: f32, width: usize) -> String {
    let ratio = ((value - min) / (max - min)).clamp(0.0, 1.0);
    let filled = (ratio * width as f32).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

// ── Recommendations ────────────────────────────────────────

fn draw_recommendations(f: &mut Frame, app: &App, area: Rect) {
    let popup = centered_rect(65, 70, area);
    f.render_widget(Clear, popup);

    let block = Block::default()
        .title(Line::from(vec![
            Span::styled(" ⚡ ", Style::default().fg(WARNING)),
            Span::styled(
                "Recommended for your hardware ",
                Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
            ),
        ]))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER_ACTIVE))
        .style(Style::default().bg(SURFACE));

    if app.recommendations.is_empty() {
        let empty = Paragraph::new(Span::styled(
            " No recommendations available",
            Style::default().fg(TEXT_DIM),
        ))
        .block(block);
        f.render_widget(empty, popup);
        return;
    }

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        format!("  {}", app.hw.summary_line()),
        Style::default().fg(TEXT_DIM),
    )));
    lines.push(Line::from(Span::styled(
        format!("  ARCH={}  Free RAM={}MB", app.hw.cpu_arch, app.hw.available_ram_mb),
        Style::default().fg(TEXT_MUTED),
    )));
    lines.push(Line::from(""));

    for (i, rec) in app.recommendations.iter().enumerate() {
        let is_selected = i == app.rec_selector_idx;
        let is_downloaded = app.available_models.iter().any(|m| m == &rec.name);
        let marker = if is_selected { "▸" } else { " " };
        let dl_marker = if is_downloaded { " ✓" } else { "" };
        let name_style = if is_selected {
            Style::default()
                .fg(ACCENT)
                .add_modifier(Modifier::BOLD)
        } else if is_downloaded {
            Style::default().fg(SUCCESS)
        } else {
            Style::default().fg(TEXT)
        };

        lines.push(Line::from(vec![
            Span::styled(format!("  {marker} "), Style::default().fg(ACCENT)),
            Span::styled(&rec.name, name_style),
            Span::styled(
                format!("  {}  {}  {:.1}GB{dl_marker}", rec.params, rec.quant, rec.size_gb),
                Style::default().fg(TEXT_DIM),
            ),
        ]));
        lines.push(Line::from(Span::styled(
            format!("      {}", rec.reason),
            Style::default().fg(TEXT_MUTED),
        )));
        lines.push(Line::from(""));
    }

    lines.push(Line::from(vec![
        Span::styled("  Enter", Style::default().fg(ACCENT)),
        Span::styled("=pull & load  ", Style::default().fg(TEXT_MUTED)),
        Span::styled("Esc", Style::default().fg(ACCENT)),
        Span::styled("=close", Style::default().fg(TEXT_MUTED)),
    ]));

    let para = Paragraph::new(Text::from(lines)).block(block);
    f.render_widget(para, popup);
}

// ── Helpers ────────────────────────────────────────────────

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

fn word_wrap(text: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 {
        return vec![text.to_string()];
    }

    let mut lines = Vec::new();
    for line in text.split('\n') {
        if line.is_empty() {
            lines.push(String::new());
            continue;
        }

        let words: Vec<&str> = line.split_whitespace().collect();
        if words.is_empty() {
            lines.push(String::new());
            continue;
        }

        let mut current = String::new();
        for word in words {
            if current.is_empty() {
                current = word.to_string();
            } else if current.len() + 1 + word.len() <= max_width {
                current.push(' ');
                current.push_str(word);
            } else {
                lines.push(current);
                current = word.to_string();
            }
        }
        if !current.is_empty() {
            lines.push(current);
        }
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}
