use crate::api::{ChatMessage, MeticuloApi, ModelInfo};
use crate::hardware::HardwareInfo;
use crate::ui;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use futures::StreamExt;
use std::time::Duration;
use tokio::sync::mpsc;

#[derive(Debug, Clone, PartialEq)]
pub enum Panel {
    Chat,
    ModelSelector,
    Recommendations,
    PullModel,
    Settings,
}

#[derive(Debug, Clone)]
pub enum Status {
    Ready,
    Generating,
    Loading(String),
    Error(String),
    Offline,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SettingsField {
    Temperature,
    TopK,
    TopP,
    SystemPrompt,
}

pub struct App {
    pub api: MeticuloApi,
    pub hw: HardwareInfo,
    pub messages: Vec<ChatMessage>,
    pub input: String,
    pub cursor_pos: usize,
    pub scroll_offset: u16,
    pub active_panel: Panel,
    pub status: Status,
    pub current_model: Option<String>,
    pub available_models: Vec<String>,
    pub model_catalog: Vec<ModelInfo>,
    pub model_selector_idx: usize,
    pub recommendations: Vec<crate::hardware::ModelRecommendation>,
    pub rec_selector_idx: usize,
    pub streaming_text: String,
    pub should_quit: bool,
    pub server_online: bool,
    pub running_model_size: Option<u64>,

    // Model selector search
    pub model_filter: String,
    pub model_filtered: Vec<usize>,

    // Pull model panel
    pub pull_catalog: Vec<String>,
    pub pull_filter: String,
    pub pull_filtered: Vec<String>,
    pub pull_selector_idx: usize,

    // Settings / inference params
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub system_prompt: String,
    pub settings_field: SettingsField,
    pub settings_editing_sysprompt: bool,
}

impl App {
    pub async fn new() -> Self {
        let api = MeticuloApi::new();
        let hw = HardwareInfo::detect();
        let recommendations = hw.recommend_models();
        let mut server_online = api.health().await.is_ok();

        // Auto-serve: start meticulo server if offline
        if !server_online {
            Self::start_server_background();
            // Poll for server to come up (up to 8 seconds)
            for _ in 0..16 {
                tokio::time::sleep(Duration::from_millis(500)).await;
                if api.health().await.is_ok() {
                    server_online = true;
                    break;
                }
            }
        }

        let model_catalog = if server_online {
            api.list_models().await.unwrap_or_default()
        } else {
            Vec::new()
        };
        let available_models = model_catalog.iter().map(|mi| mi.name.clone()).collect::<Vec<_>>();
        let model_filtered = (0..available_models.len()).collect();

        let current_model = available_models.first().cloned();

        // Fetch full catalog for pull search
        let pull_catalog = if server_online {
            api.fetch_catalog().await.unwrap_or_default()
        } else {
            Vec::new()
        };

        App {
            api,
            hw,
            messages: Vec::new(),
            input: String::new(),
            cursor_pos: 0,
            scroll_offset: 0,
            active_panel: Panel::Chat,
            status: if server_online {
                Status::Ready
            } else {
                Status::Offline
            },
            current_model,
            available_models,
            model_catalog,
            model_selector_idx: 0,
            recommendations,
            rec_selector_idx: 0,
            streaming_text: String::new(),
            should_quit: false,
            server_online,
            running_model_size: None,
            model_filter: String::new(),
            model_filtered,
            pull_catalog: pull_catalog.clone(),
            pull_filter: String::new(),
            pull_filtered: pull_catalog,
            pull_selector_idx: 0,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            system_prompt: String::new(),
            settings_field: SettingsField::Temperature,
            settings_editing_sysprompt: false,
        }
    }

    fn start_server_background() {
        use std::process::Command;
        // Try 'meticulo serve' first
        let result = Command::new("meticulo")
            .args(["serve", "--port", "22434"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();

        if result.is_err() {
            // Fallback: python3 -m meticulo serve
            let _ = Command::new("python3")
                .args(["-m", "meticulo", "serve", "--port", "22434"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn();
        }
    }

    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut terminal = ui::setup_terminal()?;

        // Channel for streaming tokens
        let (token_tx, mut token_rx) = mpsc::channel::<TokenEvent>(256);

        // Periodic health check
        let health_api = self.api.clone();
        let (health_tx, mut health_rx) = mpsc::channel::<HealthUpdate>(4);
        tokio::spawn(async move {
            loop {
                let update = match health_api.health().await {
                    Ok(health) => {
                        let online = !matches!(health.status.as_deref(), Some("error"));
                        let running = health_api
                            .list_running()
                            .await
                            .ok()
                            .and_then(|models| {
                                models
                                    .first()
                                    .map(|m| (m.name.clone(), m.size))
                            });
                        HealthUpdate {
                            online,
                            running_model: running
                                .as_ref()
                                .map(|(name, _)| name.clone())
                                .or(health.model),
                            running_model_size: running.map(|(_, size)| size),
                        }
                    }
                    Err(_) => HealthUpdate {
                        online: false,
                        running_model: None,
                        running_model_size: None,
                    },
                };

                if health_tx.send(update).await.is_err() {
                    break;
                }
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        });

        loop {
            // Draw
            terminal.draw(|f| ui::draw(f, self))?;

            // Process incoming tokens
            while let Ok(evt) = token_rx.try_recv() {
                match evt {
                    TokenEvent::Token(t) => {
                        self.streaming_text.push_str(&t);
                    }
                    TokenEvent::Done => {
                        if !self.streaming_text.is_empty() {
                            self.messages.push(ChatMessage {
                                role: "assistant".into(),
                                content: std::mem::take(&mut self.streaming_text),
                            });
                        }
                        self.status = Status::Ready;
                    }
                    TokenEvent::Error(e) => {
                        self.status = Status::Error(e);
                        self.streaming_text.clear();
                    }
                }
            }

            // Health updates
            while let Ok(update) = health_rx.try_recv() {
                self.server_online = update.online;
                if !update.online {
                    if !matches!(self.status, Status::Generating) {
                        self.status = Status::Offline;
                    }
                } else {
                    if matches!(self.status, Status::Offline) {
                        self.status = Status::Ready;
                        // Refresh catalog when server comes online
                        if self.pull_catalog.is_empty() {
                            if let Ok(cat) = self.api.fetch_catalog().await {
                                self.pull_filtered = cat.clone();
                                self.pull_catalog = cat;
                            }
                        }
                    }
                    let _ = self.refresh_models().await;
                    self.running_model_size = update.running_model_size;
                    if let Some(running) = update.running_model {
                        self.current_model = Some(running);
                    }
                }
            }

            // Handle input (with short poll timeout for responsiveness)
            if event::poll(Duration::from_millis(33))? {
                if let Event::Key(key) = event::read()? {
                    self.handle_key(key, &token_tx).await;
                }
            }

            if self.should_quit {
                break;
            }
        }

        ui::restore_terminal(terminal)?;
        Ok(())
    }

    async fn handle_key(&mut self, key: KeyEvent, token_tx: &mpsc::Sender<TokenEvent>) {
        // Global shortcuts
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c') => {
                    self.should_quit = true;
                    return;
                }
                KeyCode::Char('n') => {
                    self.messages.clear();
                    self.streaming_text.clear();
                    self.status = Status::Ready;
                    self.scroll_offset = 0;
                    return;
                }
                KeyCode::Char('k') => {
                    if self.active_panel == Panel::ModelSelector {
                        self.active_panel = Panel::Chat;
                        self.model_filter.clear();
                        self.update_model_filter();
                    } else {
                        self.active_panel = Panel::ModelSelector;
                        self.model_filter.clear();
                        self.update_model_filter();
                        self.model_selector_idx = 0;
                    }
                    return;
                }
                KeyCode::Char('r') => {
                    self.active_panel = if self.active_panel == Panel::Recommendations {
                        Panel::Chat
                    } else {
                        Panel::Recommendations
                    };
                    return;
                }
                KeyCode::Char('p') => {
                    if self.active_panel == Panel::PullModel {
                        self.active_panel = Panel::Chat;
                        self.pull_filter.clear();
                        self.update_pull_filter();
                    } else {
                        self.active_panel = Panel::PullModel;
                        self.pull_filter.clear();
                        self.update_pull_filter();
                        self.pull_selector_idx = 0;
                    }
                    return;
                }
                KeyCode::Char('s') => {
                    if self.active_panel == Panel::Settings {
                        self.active_panel = Panel::Chat;
                        self.settings_editing_sysprompt = false;
                    } else {
                        self.active_panel = Panel::Settings;
                        self.settings_field = SettingsField::Temperature;
                        self.settings_editing_sysprompt = false;
                    }
                    return;
                }
                KeyCode::Char('l') => {
                    // Clear chat
                    self.messages.clear();
                    self.streaming_text.clear();
                    self.scroll_offset = 0;
                    return;
                }
                _ => {}
            }
        }

        match self.active_panel {
            Panel::Chat => self.handle_chat_key(key, token_tx).await,
            Panel::ModelSelector => self.handle_model_key(key).await,
            Panel::Recommendations => self.handle_rec_key(key).await,
            Panel::PullModel => self.handle_pull_key(key).await,
            Panel::Settings => self.handle_settings_key(key),
        }
    }

    async fn handle_chat_key(&mut self, key: KeyEvent, token_tx: &mpsc::Sender<TokenEvent>) {
        match key.code {
            KeyCode::Enter => {
                if !self.input.is_empty() && matches!(self.status, Status::Ready) {
                    self.send_message(token_tx).await;
                }
            }
            KeyCode::Char(c) => {
                self.input.insert(self.cursor_pos, c);
                self.cursor_pos += 1;
            }
            KeyCode::Backspace => {
                if self.cursor_pos > 0 {
                    self.cursor_pos -= 1;
                    self.input.remove(self.cursor_pos);
                }
            }
            KeyCode::Delete => {
                if self.cursor_pos < self.input.len() {
                    self.input.remove(self.cursor_pos);
                }
            }
            KeyCode::Left => {
                if self.cursor_pos > 0 {
                    self.cursor_pos -= 1;
                }
            }
            KeyCode::Right => {
                if self.cursor_pos < self.input.len() {
                    self.cursor_pos += 1;
                }
            }
            KeyCode::Home => {
                self.cursor_pos = 0;
            }
            KeyCode::End => {
                self.cursor_pos = self.input.len();
            }
            KeyCode::Up => {
                if self.scroll_offset > 0 {
                    self.scroll_offset -= 1;
                }
            }
            KeyCode::Down => {
                self.scroll_offset += 1;
            }
            KeyCode::PageUp => {
                self.scroll_offset = self.scroll_offset.saturating_add(10);
            }
            KeyCode::PageDown => {
                self.scroll_offset = self.scroll_offset.saturating_sub(10);
            }
            KeyCode::Tab => {
                self.active_panel = Panel::ModelSelector;
                self.model_filter.clear();
                self.update_model_filter();
            }
            KeyCode::Esc => {
                self.active_panel = Panel::Chat;
            }
            _ => {}
        }
    }

    async fn handle_model_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up => {
                if self.model_selector_idx > 0 {
                    self.model_selector_idx -= 1;
                }
            }
            KeyCode::Down => {
                if self.model_selector_idx + 1 < self.model_filtered.len() {
                    self.model_selector_idx += 1;
                }
            }
            KeyCode::Enter => {
                if let Some(&real_idx) = self.model_filtered.get(self.model_selector_idx) {
                    if let Some(model) = self.available_models.get(real_idx) {
                        let model = model.clone();
                        self.status = Status::Loading(format!("{model} (loading)"));
                        match self.api.load_model(&model).await {
                            Ok(_) => {
                                let _ = self.api.warmup_model(&model).await;
                                self.current_model = Some(model);
                                self.status = Status::Ready;
                            }
                            Err(e) => {
                                self.status = Status::Error(e);
                            }
                        }
                        self.active_panel = Panel::Chat;
                        self.model_filter.clear();
                        self.update_model_filter();
                    }
                }
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some(&real_idx) = self.model_filtered.get(self.model_selector_idx) {
                    if let Some(model) = self.available_models.get(real_idx).cloned() {
                        self.status = Status::Loading(format!("{model} (deleting)"));
                        match self.api.delete_model(&model).await {
                            Ok(_) => {
                                let _ = self.refresh_models().await;
                                if self.current_model.as_ref() == Some(&model) {
                                    self.current_model = self.available_models.first().cloned();
                                }
                                self.update_model_filter();
                                self.status = Status::Ready;
                            }
                            Err(e) => {
                                self.status = Status::Error(e);
                            }
                        }
                    }
                }
            }
            KeyCode::Char(c) => {
                self.model_filter.push(c);
                self.update_model_filter();
                self.model_selector_idx = 0;
            }
            KeyCode::Backspace => {
                self.model_filter.pop();
                self.update_model_filter();
                self.model_selector_idx = 0;
            }
            KeyCode::Tab | KeyCode::Esc => {
                self.active_panel = Panel::Chat;
                self.model_filter.clear();
                self.update_model_filter();
            }
            _ => {}
        }
    }

    fn update_model_filter(&mut self) {
        let query = self.model_filter.to_lowercase();
        if query.is_empty() {
            self.model_filtered = (0..self.available_models.len()).collect();
        } else {
            self.model_filtered = self
                .available_models
                .iter()
                .enumerate()
                .filter(|(_, name)| name.to_lowercase().contains(&query))
                .map(|(i, _)| i)
                .collect();
        }
        if self.model_selector_idx >= self.model_filtered.len() {
            self.model_selector_idx = self.model_filtered.len().saturating_sub(1);
        }
    }

    async fn handle_pull_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up => {
                if self.pull_selector_idx > 0 {
                    self.pull_selector_idx -= 1;
                }
            }
            KeyCode::Down => {
                if self.pull_selector_idx + 1 < self.pull_filtered.len() {
                    self.pull_selector_idx += 1;
                }
            }
            KeyCode::Enter => {
                if let Some(model) = self.pull_filtered.get(self.pull_selector_idx).cloned() {
                    self.status = Status::Loading(format!("{model} (pulling)"));
                    self.active_panel = Panel::Chat;
                    self.pull_filter.clear();
                    self.update_pull_filter();

                    match self.api.pull_model(&model).await {
                        Ok(_) => {
                            self.status = Status::Loading(format!("{model} (loading)"));
                            match self.api.load_model(&model).await {
                                Ok(_) => {
                                    let _ = self.api.warmup_model(&model).await;
                                    let _ = self.refresh_models().await;
                                    self.current_model = Some(model);
                                    self.status = Status::Ready;
                                }
                                Err(e) => {
                                    let _ = self.refresh_models().await;
                                    self.status = Status::Error(e);
                                }
                            }
                        }
                        Err(e) => {
                            self.status = Status::Error(e);
                        }
                    }
                }
            }
            KeyCode::Char(c) => {
                self.pull_filter.push(c);
                self.update_pull_filter();
                self.pull_selector_idx = 0;
            }
            KeyCode::Backspace => {
                self.pull_filter.pop();
                self.update_pull_filter();
                self.pull_selector_idx = 0;
            }
            KeyCode::Esc | KeyCode::Tab => {
                self.active_panel = Panel::Chat;
                self.pull_filter.clear();
                self.update_pull_filter();
            }
            _ => {}
        }
    }

    fn update_pull_filter(&mut self) {
        let query = self.pull_filter.to_lowercase();
        if query.is_empty() {
            self.pull_filtered = self.pull_catalog.clone();
        } else {
            self.pull_filtered = self
                .pull_catalog
                .iter()
                .filter(|name| name.to_lowercase().contains(&query))
                .cloned()
                .collect();
        }
        if self.pull_selector_idx >= self.pull_filtered.len() {
            self.pull_selector_idx = self.pull_filtered.len().saturating_sub(1);
        }
    }

    fn handle_settings_key(&mut self, key: KeyEvent) {
        // If editing system prompt, handle text input
        if self.settings_editing_sysprompt {
            match key.code {
                KeyCode::Esc | KeyCode::Tab => {
                    self.settings_editing_sysprompt = false;
                }
                KeyCode::Char(c) => {
                    self.system_prompt.push(c);
                }
                KeyCode::Backspace => {
                    self.system_prompt.pop();
                }
                KeyCode::Enter => {
                    self.settings_editing_sysprompt = false;
                }
                _ => {}
            }
            return;
        }

        match key.code {
            KeyCode::Up => {
                self.settings_field = match self.settings_field {
                    SettingsField::Temperature => SettingsField::SystemPrompt,
                    SettingsField::TopK => SettingsField::Temperature,
                    SettingsField::TopP => SettingsField::TopK,
                    SettingsField::SystemPrompt => SettingsField::TopP,
                };
            }
            KeyCode::Down => {
                self.settings_field = match self.settings_field {
                    SettingsField::Temperature => SettingsField::TopK,
                    SettingsField::TopK => SettingsField::TopP,
                    SettingsField::TopP => SettingsField::SystemPrompt,
                    SettingsField::SystemPrompt => SettingsField::Temperature,
                };
            }
            KeyCode::Left => {
                match self.settings_field {
                    SettingsField::Temperature => {
                        self.temperature = (self.temperature - 0.1).max(0.0);
                        self.temperature = (self.temperature * 10.0).round() / 10.0;
                    }
                    SettingsField::TopK => {
                        self.top_k = self.top_k.saturating_sub(5);
                    }
                    SettingsField::TopP => {
                        self.top_p = (self.top_p - 0.05).max(0.0);
                        self.top_p = (self.top_p * 100.0).round() / 100.0;
                    }
                    SettingsField::SystemPrompt => {}
                }
            }
            KeyCode::Right => {
                match self.settings_field {
                    SettingsField::Temperature => {
                        self.temperature = (self.temperature + 0.1).min(2.0);
                        self.temperature = (self.temperature * 10.0).round() / 10.0;
                    }
                    SettingsField::TopK => {
                        self.top_k = (self.top_k + 5).min(200);
                    }
                    SettingsField::TopP => {
                        self.top_p = (self.top_p + 0.05).min(1.0);
                        self.top_p = (self.top_p * 100.0).round() / 100.0;
                    }
                    SettingsField::SystemPrompt => {}
                }
            }
            KeyCode::Enter => {
                if self.settings_field == SettingsField::SystemPrompt {
                    self.settings_editing_sysprompt = true;
                }
            }
            KeyCode::Esc | KeyCode::Tab => {
                self.active_panel = Panel::Chat;
                self.settings_editing_sysprompt = false;
            }
            _ => {}
        }
    }

    async fn handle_rec_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up => {
                if self.rec_selector_idx > 0 {
                    self.rec_selector_idx -= 1;
                }
            }
            KeyCode::Down => {
                if self.rec_selector_idx + 1 < self.recommendations.len() {
                    self.rec_selector_idx += 1;
                }
            }
            KeyCode::Enter => {
                if let Some(rec) = self.recommendations.get(self.rec_selector_idx) {
                    let model = rec.name.clone();
                    self.status = Status::Loading(format!("{model} (pulling)"));

                    if !self.available_models.iter().any(|m| m == &model) {
                        if let Err(e) = self.api.pull_model(&model).await {
                            self.status = Status::Error(e);
                            return;
                        }
                    }

                    self.status = Status::Loading(format!("{model} (loading)"));
                    match self.api.load_model(&model).await {
                        Ok(_) => {
                            let _ = self.api.warmup_model(&model).await;
                            let _ = self.refresh_models().await;
                            self.current_model = Some(model);
                            self.status = Status::Ready;
                            self.active_panel = Panel::Chat;
                        }
                        Err(e) => {
                            self.status = Status::Error(e);
                        }
                    }
                }
            }
            KeyCode::Esc | KeyCode::Tab => {
                self.active_panel = Panel::Chat;
            }
            _ => {}
        }
    }

    async fn send_message(&mut self, token_tx: &mpsc::Sender<TokenEvent>) {
        let model = match &self.current_model {
            Some(m) => m.clone(),
            None => {
                self.status = Status::Error("No model selected".into());
                return;
            }
        };

        let user_msg = std::mem::take(&mut self.input);
        self.cursor_pos = 0;

        self.messages.push(ChatMessage {
            role: "user".into(),
            content: user_msg,
        });

        self.status = Status::Generating;
        self.streaming_text.clear();

        let api = self.api.clone();

        // Build messages with optional system prompt
        let mut messages = Vec::new();
        if !self.system_prompt.is_empty() {
            messages.push(ChatMessage {
                role: "system".into(),
                content: self.system_prompt.clone(),
            });
        }
        messages.extend(self.messages.clone());

        let tx = token_tx.clone();
        let temp = self.temperature;
        let top_p = self.top_p;
        let top_k = self.top_k;

        tokio::spawn(async move {
            let mut stream = api.chat_stream(
                &model,
                &messages,
                Some(temp),
                Some(top_p),
                Some(top_k),
            );
            while let Some(result) = stream.next().await {
                match result {
                    Ok(token) => {
                        if tx.send(TokenEvent::Token(token)).await.is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(TokenEvent::Error(e)).await;
                        return;
                    }
                }
            }
            let _ = tx.send(TokenEvent::Done).await;
        });
    }

    async fn refresh_models(&mut self) -> Result<(), String> {
        let models = self.api.list_models().await?;
        self.available_models = models.iter().map(|m| m.name.clone()).collect();
        self.model_catalog = models;

        self.update_model_filter();

        if self.current_model.is_none() {
            self.current_model = self.available_models.first().cloned();
        }
        Ok(())
    }
}

struct HealthUpdate {
    online: bool,
    running_model: Option<String>,
    running_model_size: Option<u64>,
}

enum TokenEvent {
    Token(String),
    Done,
    Error(String),
}
