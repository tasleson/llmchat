mod system_metrics;

use anyhow::Result;
use chrono::Utc;
use clap::Parser;
use colored::Colorize;
use comfy_table::{presets::UTF8_FULL, Table};
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures_util::StreamExt;

#[cfg(target_os = "linux")]
use gio::prelude::SettingsExt;

use indicatif::{ProgressBar, ProgressStyle};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::Style as RatatuiStyle,
    widgets::{Block, Borders},
    Terminal,
};
use rustyline::{history::DefaultHistory, Editor};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::{self, Read, Write},
    path::PathBuf,
    process::Command,
    time::{Duration, Instant},
};
use syntect::{
    easy::HighlightLines,
    highlighting::{Style, ThemeSet},
    parsing::SyntaxSet,
    util::{as_24_bit_terminal_escaped, LinesWithEndings},
};
use system_metrics::{MetricsMonitor, SystemMetricsStats};
use tempfile::NamedTempFile;
use tui_textarea::TextArea;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    session: Option<PathBuf>,

    #[arg(short, long)]
    model: Option<String>,

    #[arg(long)]
    endpoint: Option<String>,

    /// API key for authentication (Bearer token sent as Authorization header)
    #[arg(long)]
    api_key: Option<String>,

    /// Optional one-shot prompt
    prompt: Option<String>,

    /// Use internal multiline editor instead of external $EDITOR
    #[arg(long, default_value = "false")]
    internal_editor: bool,

    /// Temperature for response randomness (0.0-2.0, lower = more deterministic)
    #[arg(long)]
    temperature: Option<f32>,

    /// Seed for reproducible outputs
    #[arg(long)]
    seed: Option<i64>,

    /// Run benchmark from YAML file
    #[arg(long)]
    benchmark: Option<PathBuf>,

    /// Output file for benchmark results (JSON)
    #[arg(long)]
    benchmark_output: Option<PathBuf>,

    /// Maximum context window size in tokens (auto-detected from model if available)
    #[arg(long)]
    max_tokens: Option<usize>,

    /// Display version information
    #[arg(long)]
    version: bool,
}

struct SessionConfig {
    endpoint: String,
    model: String,
    api_key: Option<String>,
    system_prompt: Option<String>,
    temperature: f32,
    seed: Option<i64>,
    last_actual_total_tokens: Option<usize>,
}

impl SessionConfig {
    fn new(endpoint: String, model: String, api_key: Option<String>, temperature: f32, seed: Option<i64>) -> Self {
        Self {
            endpoint,
            model,
            api_key,
            system_prompt: None,
            temperature,
            seed,
            last_actual_total_tokens: None,
        }
    }
}

#[derive(Serialize, Deserialize, Default)]
struct AppConfig {
    // Optional server address, e.g. http://localhost:1234/v1
    #[serde(skip_serializing_if = "Option::is_none")]
    endpoint: Option<String>,
    // Optional model identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    // Optional API key (Authorization: Bearer header)
    #[serde(skip_serializing_if = "Option::is_none")]
    api_key: Option<String>,
    // Optional temperature (0.0-2.0, lower = more deterministic)
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    // Optional seed for reproducible outputs
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    // Optional maximum context window size in tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
}

// Config and history live in XDG config dir: $XDG_CONFIG_HOME/llmchat/,
// falling back to ~/.config/llmchat/
fn config_dir() -> PathBuf {
    if let Some(xdg) = env::var_os("XDG_CONFIG_HOME") {
        if !xdg.is_empty() {
            return PathBuf::from(xdg).join("llmchat");
        }
    }
    match env::var_os("HOME") {
        Some(home) if !home.is_empty() => PathBuf::from(home).join(".config/llmchat"),
        _ => PathBuf::from(".config/llmchat"),
    }
}

fn config_path() -> PathBuf {
    config_dir().join("config.yaml")
}

fn history_path() -> PathBuf {
    config_dir().join("history")
}

fn load_config() -> AppConfig {
    let path = config_path();
    match fs::read_to_string(&path) {
        Ok(data) => match serde_yaml::from_str(&data) {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!(
                    "{}",
                    format!("Warning: Failed to parse config {}: {}", path.display(), e).yellow()
                );
                AppConfig::default()
            }
        },
        Err(_) => AppConfig::default(),
    }
}

fn save_config(cfg: &AppConfig) -> Result<()> {
    let path = config_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let data = serde_yaml::to_string(cfg)?;
    fs::write(&path, data)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // Config may contain an API key; keep it private to the owner.
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

// Show only the last 4 chars of a secret, e.g. "*…1234"
fn mask_key(key: &str) -> String {
    let chars: Vec<char> = key.chars().collect();
    if chars.len() <= 4 {
        return "****".to_string();
    }
    let (head, tail) = chars.split_at(chars.len() - 4);
    format!("{}…{}", "*".repeat(head.len()), tail.iter().collect::<String>())
}

#[derive(Serialize, Deserialize)]
struct BenchmarkConfig {
    name: String,
    temperature: f32,
    seed: Option<i64>,
    prompts: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone)]
struct PromptMetrics {
    prompt: String,
    ttft: Duration,
    total_time: Duration,
    tokens: usize,
    tokens_actual: bool,
    speed: f64,
    response_length: usize,
    response_hash: String,
}

#[derive(Serialize, Deserialize)]
struct BenchmarkResults {
    name: String,
    model: String,
    endpoint: String,
    temperature: f32,
    seed: Option<i64>,
    timestamp: String,
    prompts: Vec<PromptMetrics>,
    summary: BenchmarkSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_metrics: Option<SystemMetricsStats>,
}

#[derive(Serialize, Deserialize)]
struct BenchmarkSummary {
    total_prompts: usize,
    total_time: f64,
    total_tokens: usize,
    total_tokens_all_actual: bool,
    responses_hash: String,
    ttft_avg: f64,
    ttft_median: f64,
    ttft_min: f64,
    ttft_max: f64,
    ttft_stddev: f64,
    speed_avg: f64,
    speed_median: f64,
    speed_min: f64,
    speed_max: f64,
    speed_stddev: f64,
}

#[derive(Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}

enum ColorScheme {
    Light,
    Dark,
}

// Try to get terminal color scheme. Some people may have dark terminal theme
// even though their system theme is light.
fn terminal_scheme() -> Result<ColorScheme> {
    termbg::theme(Duration::from_millis(100))
        .map(|theme| match theme {
            termbg::Theme::Dark => ColorScheme::Dark,
            termbg::Theme::Light => ColorScheme::Light,
        })
        .map_err(|e| e.into())
}

#[cfg(target_os = "macos")]
fn color_scheme() -> ColorScheme {
    if let Ok(scheme) = terminal_scheme() {
        return scheme;
    }

    if let Ok(output) = Command::new("defaults")
        .args(["read", "-g", "AppleInterfaceStyle"])
        .output()
    {
        if output.status.success() {
            let scheme = String::from_utf8_lossy(&output.stdout)
                .trim()
                .to_lowercase();
            if scheme == "dark" {
                return ColorScheme::Dark;
            }
        }
    }

    ColorScheme::Light
}

#[cfg(target_os = "linux")]
fn color_scheme() -> ColorScheme {
    if let Ok(scheme) = terminal_scheme() {
        return scheme;
    }

    // Try to detect GNOME color scheme as a fallback (works for many Linux
    // distros)
    let settings = gio::Settings::new("org.gnome.desktop.interface");
    let scheme: String = settings.string("color-scheme").to_string();
    if scheme.contains("dark") {
        ColorScheme::Dark
    } else {
        ColorScheme::Light
    }
}

/// State machine for streaming markdown rendering
struct MarkdownStreamer {
    buffer: String,
    in_code_block: bool,
    code_language: String,
    code_buffer: String,
    syntax_set: SyntaxSet,
    theme_set: ThemeSet,
    color_scheme: ColorScheme,
}

impl MarkdownStreamer {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            in_code_block: false,
            code_language: String::new(),
            code_buffer: String::new(),
            syntax_set: SyntaxSet::load_defaults_newlines(),
            theme_set: ThemeSet::load_defaults(),
            color_scheme: color_scheme(),
        }
    }

    /// Process incoming token and print formatted output
    fn process_token(&mut self, token: &str) -> io::Result<()> {
        self.buffer.push_str(token);

        // Check for complete lines to process
        while let Some(newline_pos) = self.buffer.find('\n') {
            let line = self.buffer[..=newline_pos].to_string();
            self.buffer.drain(..=newline_pos);

            self.process_line(&line)?;
        }

        Ok(())
    }

    fn process_line(&mut self, line: &str) -> io::Result<()> {
        // Detect code block boundaries
        let trimmed = line.trim_start();
        if let Some(after_backticks) = trimmed.strip_prefix("```") {
            if self.in_code_block {
                // Closing code block - highlight and print
                self.print_code_block()?;
                print!("{}", "```".truecolor(100, 100, 100));
                if !after_backticks.is_empty() {
                    print!("{}", after_backticks.truecolor(100, 100, 100));
                }
                self.in_code_block = false;
                self.code_language.clear();
                self.code_buffer.clear();
            } else {
                // Opening code block
                self.in_code_block = true;
                self.code_language = after_backticks.trim().to_string();
                print!("{}", "```".truecolor(100, 100, 100));
                if !self.code_language.is_empty() {
                    print!("{}", self.code_language.truecolor(100, 100, 100));
                }
                println!();
            }
        } else if self.in_code_block {
            // Accumulate code block content
            self.code_buffer.push_str(line);
        } else {
            // Regular text - apply basic markdown formatting
            self.print_formatted_line(line)?;
        }

        Ok(())
    }

    fn print_code_block(&mut self) -> io::Result<()> {
        if self.code_buffer.is_empty() {
            return Ok(());
        }

        // Use a light theme suitable for white backgrounds
        let theme = &self.theme_set.themes["InspiredGitHub"];
        let syntax = self
            .syntax_set
            .find_syntax_by_token(&self.code_language)
            .unwrap_or_else(|| self.syntax_set.find_syntax_plain_text());

        let mut highlighter = HighlightLines::new(syntax, theme);

        for line in LinesWithEndings::from(&self.code_buffer) {
            let ranges: Vec<(Style, &str)> = highlighter
                .highlight_line(line, &self.syntax_set)
                .unwrap_or_default();
            let escaped = as_24_bit_terminal_escaped(&ranges[..], false);
            print!("{}", escaped);
        }

        io::stdout().flush()
    }

    fn print_formatted_line(&self, line: &str) -> io::Result<()> {
        // Simple formatting for common markdown patterns
        let result;

        // Headers (use dark colors for light backgrounds)
        if let Some(stripped) = line.strip_prefix("### ") {
            result = stripped.trim().blue().bold().to_string() + "\n";
        } else if let Some(stripped) = line.strip_prefix("## ") {
            result = stripped.trim().blue().bold().to_string() + "\n";
        } else if let Some(stripped) = line.strip_prefix("# ") {
            result = stripped.trim().blue().bold().to_string() + "\n";
        } else {
            // Inline code with `backticks`
            result = self.highlight_inline_code(line);
        }

        match self.color_scheme {
            ColorScheme::Dark => print!("{}", result.white()),
            ColorScheme::Light => print!("{}", result.black()),
        }

        io::stdout().flush()
    }

    fn highlight_inline_code(&self, text: &str) -> String {
        let mut result = String::new();
        let mut in_backtick = false;
        let mut current = String::new();

        for ch in text.chars() {
            if ch == '`' {
                if in_backtick {
                    // Closing backtick - format as code (dark text on light gray background)
                    result.push_str(&format!("{}", current.black().on_truecolor(230, 230, 230)));
                    current.clear();
                } else {
                    // Opening backtick - flush any normal text
                    result.push_str(&current);
                    current.clear();
                }
                in_backtick = !in_backtick;
            } else {
                current.push(ch);
            }
        }

        result.push_str(&current);
        result
    }

    /// Flush any remaining buffered content
    fn flush(&mut self) -> io::Result<()> {
        if !self.buffer.is_empty() {
            let remaining = self.buffer.clone();
            self.buffer.clear();

            if self.in_code_block {
                print!("{}", remaining.black());
            } else {
                self.print_formatted_line(&remaining)?;
            }
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = Args::parse();

    // Handle --version flag
    if args.version {
        const VERSION: &str = env!("CARGO_PKG_VERSION");
        const GIT_SHA: &str = env!("GIT_SHA");
        const GIT_BRANCH: &str = env!("GIT_BRANCH");
        const GIT_DIRTY: &str = env!("GIT_DIRTY");

        println!("llmchat     {}", VERSION);
        println!("Git SHA:    {}", GIT_SHA);
        println!("Git Branch: {}", GIT_BRANCH);
        println!("Git Dirty:  {}", GIT_DIRTY);
        return Ok(());
    }

    // Load config file (if any) and apply precedence:
    // CLI flag > config file > built-in default.
    let app_config = load_config();
    args.endpoint = args
        .endpoint
        .or(app_config.endpoint)
        .or_else(|| Some("http://localhost:1234/v1".to_string()));
    args.model = args
        .model
        .or(app_config.model)
        .or_else(|| Some("local-model".to_string()));
    args.temperature = args.temperature.or(app_config.temperature).or(Some(1.0));
    args.seed = args.seed.or(app_config.seed);
    args.max_tokens = args.max_tokens.or(app_config.max_tokens).or(Some(8192));
    args.api_key = args
        .api_key
        .or(app_config.api_key)
        .or_else(|| env::var("LLMCHAT_API_KEY").ok());

    // Try to auto-detect context window from model info
    if let (Some(endpoint), Some(model)) = (&args.endpoint, &args.model) {
        if let Some(detected_context) =
            fetch_model_context_window(endpoint, model, args.api_key.as_deref()).await
        {
            if args.max_tokens == Some(8192) {
                // Only override if using default value
                args.max_tokens = Some(detected_context);
                eprintln!(
                    "{}",
                    format!("Auto-detected context window: {} tokens", detected_context).dimmed()
                );
            }
        }
    }

    // Best-effort: warn (but don't block) if the configured model isn't listed
    // by the endpoint.
    if let (Some(endpoint), Some(model)) = (&args.endpoint, &args.model) {
        if validate_model(endpoint, model, args.api_key.as_deref())
            .await
            == Some(false)
        {
            eprintln!(
                "{}",
                format!(
                    "Warning: Model '{}' is not in the endpoint's model list. Use /models to list available models.",
                    model
                )
                .yellow()
            );
        }
    }

    // --- Benchmark mode ---
    if let Some(ref benchmark_file) = args.benchmark {
        run_benchmark(benchmark_file, &args).await?;
        return Ok(());
    }

    let mut messages = load_session(&args.session)?;
    let mut config = SessionConfig::new(
        args.endpoint.clone().unwrap_or_default(),
        args.model.clone().unwrap_or_default(),
        args.api_key.clone(),
        args.temperature.unwrap_or(1.0),
        args.seed,
    );

    // --- One-shot from CLI argument ---
    if let Some(ref p) = args.prompt {
        let (_metrics, _total) = handle_prompt(p.clone(), &mut messages, &config).await?;
        save_session(&args.session, &messages)?;
        return Ok(());
    }

    // --- One-shot from pipe ---
    if !atty::is(atty::Stream::Stdin) {
        let mut input = String::new();
        io::stdin().read_to_string(&mut input)?;
        let (_metrics, _total) = handle_prompt(input, &mut messages, &config).await?;
        save_session(&args.session, &messages)?;
        return Ok(());
    }

    // --- Interactive REPL ---
    let mut rl = Editor::<(), DefaultHistory>::new()?;
    let history = history_path();
    if history.exists() {
        rl.load_history(&history)?;
    }
    loop {
        let line = rl.readline(&format!("{}", "lms> ".green().bold()));

        match line {
            Ok(input) => {
                let input: &str = input.trim();

                if input.is_empty() {
                    continue;
                }

                rl.add_history_entry(input)?;

                if input.starts_with('/') {
                    if handle_command(input, &mut messages, &mut config, &args).await? {
                        break;
                    }
                    save_session(&args.session, &messages)?;
                    continue;
                }
                // Lets add a newline to create a clearer boundary between request and response
                println!();
                let (_metrics, actual_total) =
                    handle_prompt(input.to_string(), &mut messages, &config).await?;

                // Update last known actual token count
                if actual_total.is_some() {
                    config.last_actual_total_tokens = actual_total;
                }

                check_context_usage(
                    &messages,
                    args.max_tokens.unwrap_or(8192),
                    actual_total,
                    &config.system_prompt,
                );
                save_session(&args.session, &messages)?;
            }
            Err(_) => break,
        }
    }

    if let Some(parent) = history.parent() {
        fs::create_dir_all(parent)?;
    }
    rl.save_history(&history)?;

    Ok(())
}

fn load_session(path: &Option<PathBuf>) -> Result<Vec<Message>> {
    if let Some(p) = path {
        if p.exists() {
            let data = fs::read_to_string(p)?;
            return Ok(serde_json::from_str(&data)?);
        }
    }
    Ok(vec![])
}

fn save_session(path: &Option<PathBuf>, messages: &[Message]) -> Result<()> {
    if let Some(p) = path {
        fs::write(p, serde_json::to_string_pretty(messages)?)?;
    }
    Ok(())
}

fn estimate_tokens(messages: &[Message], system_prompt: &Option<String>) -> usize {
    // Rough estimation: ~4 characters per token
    let mut total_chars: usize = messages.iter().map(|m| m.content.len()).sum();
    if let Some(sp) = system_prompt {
        total_chars += sp.len();
    }
    total_chars / 4
}

fn check_context_usage(
    messages: &[Message],
    max_tokens: usize,
    actual_total: Option<usize>,
    system_prompt: &Option<String>,
) {
    let (token_count, is_actual) = if let Some(actual) = actual_total {
        (actual, true)
    } else {
        (estimate_tokens(messages, system_prompt), false)
    };

    let percentage = (token_count as f64 / max_tokens as f64) * 100.0;

    if percentage >= 90.0 {
        eprintln!(
            "{}",
            format!(
                "⚠️  WARNING: Context window usage at {:.1}% ({} / {}) {}",
                percentage,
                if is_actual { "ACTUAL:" } else { "ESTIMATED:" },
                token_count,
                max_tokens
            )
            .yellow()
        );
    } else if percentage >= 75.0 {
        eprintln!(
            "{}",
            format!(
                "Notice: Context window usage at {:.1}% ({} / {}) {}",
                percentage,
                if is_actual { "ACTUAL:" } else { "ESTIMATED:" },
                token_count,
                max_tokens
            )
            .yellow()
        );
    }
}

async fn fetch_model_context_window(
    endpoint: &str,
    model: &str,
    api_key: Option<&str>,
) -> Option<usize> {
    let client = reqwest::Client::new();
    let models_url = format!("{}/models", endpoint);

    let mut request = client.get(&models_url);
    if let Some(key) = api_key {
        request = request.header("Authorization", format!("Bearer {}", key));
    }

    let response = request.send().await.ok()?;
    let models_data: serde_json::Value = response.json().await.ok()?;

    let models_array = models_data.get("data")?.as_array()?;

    for model_info in models_array {
        if model_info.get("id")?.as_str()? == model {
            // Try to find context window in various possible fields
            let possible_fields = [
                "context_window",
                "max_tokens",
                "context_length",
                "max_model_len",
                "max_position_embeddings",
            ];

            for field in &possible_fields {
                if let Some(value) = model_info.get(field) {
                    if let Some(num) = value.as_u64() {
                        return Some(num as usize);
                    }
                }
            }
        }
    }

    None
}

async fn list_models(endpoint: &str, api_key: Option<&str>) -> Result<Vec<String>> {
    let client = reqwest::Client::new();
    let models_url = format!("{}/models", endpoint);

    let mut request = client.get(&models_url);
    if let Some(key) = api_key {
        request = request.header("Authorization", format!("Bearer {}", key));
    }

    let response = request.send().await?;
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unable to read error".to_string());
        return Err(anyhow::anyhow!(
            "HTTP {} from endpoint: {}",
            status,
            error_text
        ));
    }

    let models_data: serde_json::Value = response.json().await?;
    let models_array = models_data.get("data").and_then(|d| d.as_array());

    let mut models = Vec::new();
    if let Some(models_array) = models_array {
        for model_info in models_array {
            if let Some(id) = model_info.get("id").and_then(|id| id.as_str()) {
                models.push(id.to_string());
            }
        }
    }

    models.sort();
    Ok(models)
}

// Best-effort: returns Some(true) if the model is listed, Some(false) if not,
// and None if the model list cannot be fetched or is empty (skip validation).
async fn validate_model(endpoint: &str, model: &str, api_key: Option<&str>) -> Option<bool> {
    match list_models(endpoint, api_key).await {
        Ok(models) if !models.is_empty() => Some(models.iter().any(|m| m == model)),
        _ => None,
    }
}

fn open_internal_editor() -> Result<String> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create text area widget
    let mut textarea = TextArea::default();
    textarea.set_block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Multi-line Editor - Press Esc or Ctrl+D to finish "),
    );
    // Remove the cursor line underline styling
    textarea.set_cursor_line_style(RatatuiStyle::default());

    // Event loop
    loop {
        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1)])
                .split(f.area());
            f.render_widget(&textarea, chunks[0]);
        })?;

        if let Event::Key(key) = event::read()? {
            match (key.code, key.modifiers) {
                (KeyCode::Esc, _) => break,
                (KeyCode::Char('d'), KeyModifiers::CONTROL) => break,
                _ => {
                    textarea.input(key);
                }
            }
        }
    }

    // Cleanup terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    // Get the content
    let content = textarea.lines().join("\n");
    Ok(content)
}

fn open_external_editor() -> Result<String> {
    let tmp = NamedTempFile::new()?;
    let editor = env::var("EDITOR").unwrap_or_else(|_| "vim".into());
    Command::new(editor).arg(tmp.path()).status()?;
    Ok(fs::read_to_string(tmp.path())?)
}

fn open_editor(use_internal: bool) -> Result<String> {
    if use_internal {
        open_internal_editor()
    } else {
        open_external_editor()
    }
}

fn create_spinner(message: &str) -> ProgressBar {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );
    spinner.set_message(message.to_string());
    spinner.enable_steady_tick(std::time::Duration::from_millis(80));
    spinner
}

async fn handle_command(
    cmd: &str,
    messages: &mut Vec<Message>,
    config: &mut SessionConfig,
    args: &Args,
) -> Result<bool> {
    const HELP_WIDTH: usize = 20;
    let parts: Vec<&str> = cmd.splitn(2, ' ').collect();

    match parts[0] {
        "/help" => {
            println!("{}", "Available Commands:".green().bold());
            println!();
            println!(
                "  {:<width$}  Exit the program",
                "/exit".cyan(),
                width = HELP_WIDTH
            );
            println!(
                "  {:<width$}  Clear conversation history (removes all previous messages)",
                "/clear".cyan(),
                width = HELP_WIDTH
            );
            println!(
                "  {:<width$}  Open your default editor (${}) for multi-line input",
                "/edit".cyan(),
                "EDITOR".yellow(),
                width = HELP_WIDTH
            );
            println!(
                "  {:<width$}  Set a system prompt for the conversation",
                "/system <prompt>".cyan(),
                width = HELP_WIDTH
            );
            println!(
                "  {:<width$}  Example: /system You are a helpful coding assistant",
                "".dimmed(),
                width = HELP_WIDTH
            );
            println!(
                "  {:<width$}  Set temperature (0.0-2.0, lower = more deterministic)",
                "/temperature <value>".cyan(),
                width = HELP_WIDTH
            );
            println!(
                "  {:<width$}  Example: /temperature 0.7",
                "".dimmed(),
                width = HELP_WIDTH
            );
            println!(
                "  {:<width$}  Set seed for reproducible outputs",
                "/seed <value>".cyan(),
                width = HELP_WIDTH
            );
            println!(
                "  {:<width$}  Example: /seed 42",
                "".dimmed(),
                width = HELP_WIDTH
            );
            println!(
                "  {:<width$}  Clear the seed (use random generation)",
                "/seed clear".cyan(),
                width = HELP_WIDTH
            );
            println!(
                "  {:<width$}  Show current configuration",
                "/config".cyan(),
                width = HELP_WIDTH
            );
            println!(
                "  {:<width$}  Set endpoint for this session (no config change)",
                "/endpoint <url>".cyan(),
                width = HELP_WIDTH
            );
            println!(
                "  {:<width$}  Set endpoint and persist it to config file",
                "/save-endpoint <url>".cyan(),
                width = HELP_WIDTH
            );
            println!(
                "  {:<width$}  Set API key for this session (no config change)",
                "/api-key <key>".cyan(),
                width = HELP_WIDTH
            );
            println!(
                "  {:<width$}  Set API key and persist it to config file",
                "/save-api-key <key>".cyan(),
                width = HELP_WIDTH
            );
            println!(
                "  {:<width$}  List models available on the endpoint",
                "/models".cyan(),
                width = HELP_WIDTH
            );
            println!(
                "  {:<width$}  Set model for this session (no config change)",
                "/model <name>".cyan(),
                width = HELP_WIDTH
            );
            println!(
                "  {:<width$}  Set model and persist it to config file",
                "/save-model <name>".cyan(),
                width = HELP_WIDTH
            );
            println!(
                "  {:<width$}  Show session info and token usage",
                "/info".cyan(),
                width = HELP_WIDTH
            );
            println!(
                "  {:<width$}  Show this help message",
                "/help".cyan(),
                width = HELP_WIDTH
            );
            println!();
            println!(
                "{}",
                "Tip: Use Ctrl+D or Ctrl+C to exit at any time".dimmed()
            );
        }
        "/exit" => return Ok(true),
        "/clear" => {
            messages.clear();
            config.last_actual_total_tokens = None;
            println!("Session cleared.");
        }
        "/edit" => {
            let content = open_editor(args.internal_editor)?;
            // Lets output the prompt, so that the user can see exactly what they sent to the
            // model
            println!("{content}");
            let (_metrics, _total) = handle_prompt(content, messages, config).await?;
        }
        "/system" => {
            if parts.len() > 1 {
                config.system_prompt = Some(parts[1].to_string());
                println!("System prompt set.");
            } else {
                println!("Usage: /system <prompt>");
            }
        }
        "/temperature" => {
            if parts.len() > 1 {
                match parts[1].parse::<f32>() {
                    Ok(temp) if (0.0..=2.0).contains(&temp) => {
                        config.temperature = temp;
                        println!("Temperature set to: {}", temp);
                    }
                    Ok(_) => println!("Temperature must be between 0.0 and 2.0"),
                    Err(_) => {
                        println!("Invalid temperature value. Use a number between 0.0 and 2.0")
                    }
                }
            } else {
                println!("Current temperature: {}", config.temperature);
                println!("Usage: /temperature <value>");
            }
        }
        "/seed" => {
            if parts.len() > 1 {
                if parts[1] == "clear" {
                    config.seed = None;
                    println!("Seed cleared (using random generation).");
                } else {
                    match parts[1].parse::<i64>() {
                        Ok(seed_val) => {
                            config.seed = Some(seed_val);
                            println!("Seed set to: {}", seed_val);
                        }
                        Err(_) => println!("Invalid seed value. Use an integer or 'clear'."),
                    }
                }
            } else {
                match config.seed {
                    Some(s) => println!("Current seed: {}", s),
                    None => println!("Current seed: None (random generation)"),
                }
                println!("Usage: /seed <value> or /seed clear");
            }
        }
        "/endpoint" | "/save-endpoint" => {
            let persist = parts[0] == "/save-endpoint";
            if parts.len() > 1 {
                config.endpoint = parts[1].trim().to_string();
                println!("Endpoint set to: {}", config.endpoint.cyan());
            } else {
                println!("Current endpoint: {}", config.endpoint.cyan());
            }
            if persist {
                let mut cfg = load_config();
                cfg.endpoint = Some(config.endpoint.clone());
                save_config(&cfg)?;
                println!(
                    "{}",
                    format!("Saved endpoint to config: {}", config_path().display()).green()
                );
            } else if parts.len() <= 1 {
                println!("Usage: /endpoint <url>");
            }
        }
        "/api-key" | "/save-api-key" => {
            let persist = parts[0] == "/save-api-key";
            let clear = parts.len() > 1 && parts[1].trim() == "clear";
            if clear {
                config.api_key = None;
                println!("API key cleared.");
            } else if parts.len() > 1 {
                config.api_key = Some(parts[1].trim().to_string());
                println!("API key set (stored in memory).");
            } else {
                match &config.api_key {
                    Some(key) => println!("Current API key: {}", mask_key(key).cyan()),
                    None => println!("Current API key: {} (not set)", "None".dimmed()),
                }
            }
            if persist {
                let mut cfg = load_config();
                if clear {
                    cfg.api_key = None;
                } else if let Some(ref key) = config.api_key {
                    cfg.api_key = Some(key.clone());
                }
                save_config(&cfg)?;
                println!(
                    "{}",
                    format!("Saved API key setting to config: {}", config_path().display()).green()
                );
            }
            if parts.len() <= 1 {
                println!("Usage: /api-key <key> | /api-key clear");
            }
        }
        "/models" => {
            match list_models(&config.endpoint, config.api_key.as_deref()).await {
                Ok(models) if models.is_empty() => {
                    println!("No models found at endpoint.");
                }
                Ok(models) => {
                    println!("{}", "Available Models:".green().bold());
                    for model in &models {
                        if *model == config.model {
                            println!("  {} {}", "*".green().bold(), model.cyan());
                        } else {
                            println!("    {}", model);
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "{}",
                        format!("Error fetching models: {}", e).red()
                    );
                }
            }
        }
        "/model" | "/save-model" => {
            let persist = parts[0] == "/save-model";
            if parts.len() > 1 {
                config.model = parts[1].trim().to_string();
                println!("Model set to: {}", config.model.cyan());
                if validate_model(&config.endpoint, &config.model, config.api_key.as_deref())
                    .await
                    == Some(false)
                {
                    println!(
                        "{}",
                        format!(
                            "Warning: Model '{}' is not in the endpoint's model list. Use /models to list available models.",
                            config.model
                        )
                        .yellow()
                    );
                }
            } else {
                println!("Current model: {}", config.model.cyan());
            }
            if persist {
                let mut cfg = load_config();
                cfg.model = Some(config.model.clone());
                save_config(&cfg)?;
                println!(
                    "{}",
                    format!("Saved model to config: {}", config_path().display()).green()
                );
            } else if parts.len() <= 1 {
                println!("Usage: /model <name> | /save-model <name>");
            }
        }
        "/config" => {
            println!("{}", "Current Configuration:".green().bold());
            println!("  Model:       {}", config.model.cyan());
            println!("  Endpoint:    {}", config.endpoint.cyan());
            match &config.api_key {
                Some(key) => println!("  API key:     {}", mask_key(key).cyan()),
                None => println!("  API key:     {}", "None".dimmed()),
            }
            println!("  Max tokens:  {}", args.max_tokens.unwrap_or(8192).to_string().cyan());
            println!("  Temperature: {}", config.temperature.to_string().cyan());
            match config.seed {
                Some(s) => println!("  Seed:        {}", s.to_string().cyan()),
                None => println!("  Seed:        {}", "None (random)".dimmed()),
            }
            match &config.system_prompt {
                Some(sp) => println!("  System:      {}", sp.cyan()),
                None => println!("  System:      {}", "None".dimmed()),
            }
        }
        "/info" => {
            let max_tokens = args.max_tokens.unwrap_or(8192);

            println!("{}", "Session Information:".green().bold());
            println!("  Messages:    {}", messages.len());

            if let Some(actual) = config.last_actual_total_tokens {
                let percentage = (actual as f64 / max_tokens as f64) * 100.0;
                println!(
                    "  {}",
                    format!(
                        "Tokens (ACTUAL): {} / {} ({:.1}%)",
                        actual, max_tokens, percentage
                    )
                    .cyan()
                );
                println!("  {}", "└─ From last API response".dimmed());
            } else {
                let estimated = estimate_tokens(messages, &config.system_prompt);
                let percentage = (estimated as f64 / max_tokens as f64) * 100.0;
                println!(
                    "  {}",
                    format!(
                        "Tokens (ESTIMATE): ~{} / {} ({:.1}%)",
                        estimated, max_tokens, percentage
                    )
                    .yellow()
                );
                println!("  {}", "└─ Based on ~4 chars/token".dimmed());
            }
        }
        _ => println!("Unknown command. Type /help for available commands."),
    }

    Ok(false)
}

async fn handle_prompt(
    input: String,
    messages: &mut Vec<Message>,
    config: &SessionConfig,
) -> Result<(Option<PromptMetrics>, Option<usize>)> {
    let trimmed = input.trim().to_string();

    if trimmed.is_empty() {
        println!("Empty input, please try again!");
        return Ok((None, None));
    }

    messages.push(Message {
        role: "user".into(),
        content: trimmed,
    });

    let mut payload_msgs = vec![];

    if let Some(sys) = &config.system_prompt {
        payload_msgs.push(json!({ "role": "system", "content": sys }));
    }

    for m in messages.iter() {
        payload_msgs.push(json!(m));
    }

    let client = reqwest::Client::new();
    let url = format!("{}/chat/completions", config.endpoint);

    let spinner = create_spinner("Waiting for response...");

    let start_time = Instant::now();
    let mut first_token_time: Option<Instant> = None;
    let mut token_count = 0;

    let mut request_body = json!({
        "model": config.model.clone(),
        "messages": payload_msgs,
        "stream": true,
        "stream_options": {
            "include_usage": true
        },
        "temperature": config.temperature
    });

    if let Some(seed_value) = config.seed {
        request_body["seed"] = json!(seed_value);
    }

    let mut request = client.post(&url).json(&request_body);
    if let Some(key) = &config.api_key {
        request = request.header("Authorization", format!("Bearer {}", key));
    }

    let response = request.send().await;

    let response = match response {
        Ok(r) => r,
        Err(e) => {
            spinner.finish_and_clear();
            eprintln!(
                "{}",
                format!("Error: Failed to connect to endpoint: {}", e).red()
            );
            eprintln!("{}", format!("Endpoint: {}", url).yellow());
            return Err(e.into());
        }
    };

    // Check status code
    if !response.status().is_success() {
        spinner.finish_and_clear();
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unable to read error".to_string());
        eprintln!("{}", format!("Error: HTTP {} from endpoint", status).red());
        eprintln!("{}", format!("Endpoint: {}", url).yellow());
        if !error_text.is_empty() {
            eprintln!("{}", format!("Response: {}", error_text).yellow());
        }
        if !url.ends_with("/v1/chat/completions") {
            eprintln!(
                "{}",
                "Hint: Endpoint should typically end with /v1".yellow()
            );
        }
        return Err(anyhow::anyhow!("Request failed with status {}", status));
    }

    let mut stream = response.bytes_stream();
    let mut assistant_reply = String::new();
    let mut received_any_data = false;
    let mut md_streamer = MarkdownStreamer::new();
    let mut actual_token_count: Option<usize> = None;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);

        for line in text.lines() {
            if let Some(payload) = line.strip_prefix("data: ") {
                received_any_data = true;
                if payload == "[DONE]" {
                    break;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) {
                    // Check for usage information
                    if let Some(usage) = v.get("usage") {
                        if let Some(total) = usage["total_tokens"].as_u64() {
                            actual_token_count = Some(total as usize);
                        }
                    }

                    if let Some(token) = v["choices"][0]["delta"]["content"].as_str() {
                        // Clear spinner on first token
                        if first_token_time.is_none() {
                            spinner.finish_and_clear();
                            first_token_time = Some(Instant::now());
                        }

                        md_streamer.process_token(token)?;
                        assistant_reply.push_str(token);
                        token_count += 1;
                    }
                }
            }
        }
    }

    // Flush any remaining buffered content
    md_streamer.flush()?;

    spinner.finish_and_clear();

    // Check if we received any streaming data
    if !received_any_data && assistant_reply.is_empty() {
        eprintln!(
            "{}",
            "Error: No streaming data received from endpoint".red()
        );
        eprintln!("{}", format!("Endpoint: {}", url).yellow());
        if !url.ends_with("/v1/chat/completions") {
            eprintln!(
                "{}",
                "Hint: Endpoint should typically end with /v1".yellow()
            );
        }
        return Err(anyhow::anyhow!("No response data received"));
    }

    println!();

    let total_time = start_time.elapsed();
    let ttft = first_token_time.map(|t| t.duration_since(start_time));

    // Use actual token count if available, otherwise use chunk count
    let final_token_count = actual_token_count.unwrap_or(token_count);
    let tokens_are_actual = actual_token_count.is_some();

    // Print statistics
    let mut stats = Vec::new();
    if let Some(ttft_duration) = ttft {
        stats.push(format!("TTFT: {:.2}s", ttft_duration.as_secs_f64()));
    }
    stats.push(format!("Total: {:.2}s", total_time.as_secs_f64()));
    if tokens_are_actual {
        stats.push(format!("Tokens: {}", final_token_count));
    } else {
        stats.push(format!("Tokens: ~{}", final_token_count));
    }

    if let Some(ttft_duration) = ttft {
        let generation_time = total_time.as_secs_f64() - ttft_duration.as_secs_f64();
        if generation_time > 0.0 && final_token_count > 0 {
            let tps = final_token_count as f64 / generation_time;
            stats.push(format!("Speed: {:.1} tok/s", tps));
        }
    }

    println!("{}", format!("[{}]", stats.join(" | ")).dimmed());

    // Calculate hash of response
    let mut hasher = Sha256::new();
    hasher.update(assistant_reply.as_bytes());
    let response_hash = hex::encode(hasher.finalize());

    let metrics = ttft.map(|ttft_duration| PromptMetrics {
        prompt: messages.last().unwrap().content.clone(),
        ttft: ttft_duration,
        total_time,
        tokens: final_token_count,
        tokens_actual: tokens_are_actual,
        speed: if let Some(ttft_duration) = ttft {
            let generation_time = total_time.as_secs_f64() - ttft_duration.as_secs_f64();
            if generation_time > 0.0 && final_token_count > 0 {
                final_token_count as f64 / generation_time
            } else {
                0.0
            }
        } else {
            0.0
        },
        response_length: assistant_reply.len(),
        response_hash: response_hash.clone(),
    });

    messages.push(Message {
        role: "assistant".into(),
        content: assistant_reply,
    });

    Ok((metrics, actual_token_count))
}

async fn run_benchmark(benchmark_file: &PathBuf, args: &Args) -> Result<()> {
    let yaml_content = fs::read_to_string(benchmark_file)?;
    let benchmark: BenchmarkConfig = serde_yaml::from_str(&yaml_content)?;

    println!(
        "{}",
        format!("Running benchmark: {}", benchmark.name)
            .green()
            .bold()
    );
    println!(
        "Model: {} | Temperature: {} | Seed: {}",
        args.model.as_deref().unwrap_or("local-model").cyan(),
        benchmark.temperature.to_string().cyan(),
        benchmark
            .seed
            .map(|s| s.to_string())
            .unwrap_or("None".to_string())
            .cyan()
    );
    println!("Endpoint: {}", args.endpoint.as_deref().unwrap_or("http://localhost:1234/v1").cyan());
    println!("Prompts: {}", benchmark.prompts.len());
    println!();

    let config = SessionConfig {
        endpoint: args.endpoint.clone().unwrap_or_default(),
        model: args.model.clone().unwrap_or_default(),
        api_key: args.api_key.clone(),
        system_prompt: None,
        temperature: benchmark.temperature,
        seed: benchmark.seed,
        last_actual_total_tokens: None,
    };

    let mut all_metrics = Vec::new();
    let mut messages = Vec::new();

    // Start system metrics monitoring
    let mut metrics_monitor = MetricsMonitor::new().ok();
    if let Some(ref mut monitor) = metrics_monitor {
        if let Err(e) = monitor.start() {
            eprintln!(
                "{}",
                format!("Warning: Failed to start metrics monitoring: {}", e).yellow()
            );
            metrics_monitor = None;
        }
    }

    for (idx, prompt) in benchmark.prompts.iter().enumerate() {
        println!(
            "{}",
            format!("Prompt {}: \"{}\"", idx + 1, prompt)
                .yellow()
                .bold()
        );
        println!("{}", "━".repeat(80).dimmed());

        let (metrics, _total) = handle_prompt(prompt.clone(), &mut messages, &config).await?;
        if let Some(m) = metrics {
            all_metrics.push(m);
        }

        println!();
    }

    // Stop system metrics monitoring and get stats
    let system_metrics = if let Some(mut monitor) = metrics_monitor {
        match monitor.stop() {
            Ok(stats) => Some(stats),
            Err(e) => {
                eprintln!(
                    "{}",
                    format!("Warning: Failed to collect system metrics: {}", e).yellow()
                );
                None
            }
        }
    } else {
        None
    };

    // Calculate summary statistics
    let summary = calculate_summary(&all_metrics);

    // Display results
    display_benchmark_results(&all_metrics, &summary, &system_metrics);

    // Save to JSON if requested
    if let Some(output_file) = &args.benchmark_output {
        let results = BenchmarkResults {
            name: benchmark.name,
            model: args.model.clone().unwrap_or_default(),
            endpoint: args.endpoint.clone().unwrap_or_default(),
            temperature: benchmark.temperature,
            seed: benchmark.seed,
            timestamp: Utc::now().to_rfc3339(),
            prompts: all_metrics,
            summary,
            system_metrics,
        };
        fs::write(output_file, serde_json::to_string_pretty(&results)?)?;
        println!();
        println!(
            "{}",
            format!("Results saved to: {}", output_file.display()).green()
        );
    }

    Ok(())
}

fn calculate_summary(metrics: &[PromptMetrics]) -> BenchmarkSummary {
    let total_prompts = metrics.len();
    let total_time: f64 = metrics.iter().map(|m| m.total_time.as_secs_f64()).sum();
    let total_tokens: usize = metrics.iter().map(|m| m.tokens).sum();
    let total_tokens_all_actual = metrics.iter().all(|m| m.tokens_actual);

    // Calculate hash of all response hashes
    let mut hasher = Sha256::new();
    for m in metrics {
        hasher.update(m.response_hash.as_bytes());
    }
    let responses_hash = hex::encode(hasher.finalize());

    let ttfts: Vec<f64> = metrics.iter().map(|m| m.ttft.as_secs_f64()).collect();
    let speeds: Vec<f64> = metrics.iter().map(|m| m.speed).collect();

    BenchmarkSummary {
        total_prompts,
        total_time,
        total_tokens,
        total_tokens_all_actual,
        responses_hash,
        ttft_avg: calculate_mean(&ttfts),
        ttft_median: calculate_median(&ttfts),
        ttft_min: ttfts.iter().cloned().fold(f64::INFINITY, f64::min),
        ttft_max: ttfts.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        ttft_stddev: calculate_stddev(&ttfts),
        speed_avg: calculate_mean(&speeds),
        speed_median: calculate_median(&speeds),
        speed_min: speeds.iter().cloned().fold(f64::INFINITY, f64::min),
        speed_max: speeds.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        speed_stddev: calculate_stddev(&speeds),
    }
}

fn calculate_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn calculate_median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

fn calculate_stddev(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let mean = calculate_mean(values);
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

fn display_benchmark_results(
    metrics: &[PromptMetrics],
    summary: &BenchmarkSummary,
    system_metrics: &Option<SystemMetricsStats>,
) {
    println!("{}", "━".repeat(80).green());
    println!("{}", "PER-PROMPT RESULTS".green().bold());
    println!("{}", "━".repeat(80).green());

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["#", "TTFT", "Time", "Tokens", "Speed", "Len", "Hash"]);

    for (idx, m) in metrics.iter().enumerate() {
        let tokens_str = if m.tokens_actual {
            format!("{}", m.tokens)
        } else {
            format!("~{}", m.tokens)
        };

        table.add_row(vec![
            format!("{}", idx + 1),
            format!("{:.2}s", m.ttft.as_secs_f64()),
            format!("{:.2}s", m.total_time.as_secs_f64()),
            tokens_str,
            format!("{:.1} t/s", m.speed),
            format!("{}", m.response_length),
            format!("{:.8}", m.response_hash),
        ]);
    }

    println!("{}", table);
    println!();

    println!("{}", "━".repeat(80).green());
    println!("{}", "BENCHMARK SUMMARY".green().bold());
    println!("{}", "━".repeat(80).green());
    println!("  Prompts:      {}", summary.total_prompts);
    println!("  Total Time:   {:.2}s", summary.total_time);
    if summary.total_tokens_all_actual {
        println!("  Total Tokens: {} (actual)", summary.total_tokens);
    } else {
        println!("  Total Tokens: ~{} (estimated)", summary.total_tokens);
    }
    println!("  Responses Hash: {:.16}...", summary.responses_hash);
    println!();
    println!("{}", "  TTFT (Time to First Token):".cyan().bold());
    println!(
        "    Avg: {:.2}s | Median: {:.2}s | Min: {:.2}s | Max: {:.2}s | StdDev: {:.3}s",
        summary.ttft_avg,
        summary.ttft_median,
        summary.ttft_min,
        summary.ttft_max,
        summary.ttft_stddev
    );
    println!();
    println!("{}", "  Speed (tokens/sec):".cyan().bold());
    println!(
        "    Avg: {:.1} t/s | Median: {:.1} t/s | Min: {:.1} t/s | Max: {:.1} t/s | StdDev: {:.2} t/s",
        summary.speed_avg, summary.speed_median, summary.speed_min, summary.speed_max, summary.speed_stddev
    );
    println!("{}", "━".repeat(80).green());

    // Display system metrics if available
    if let Some(sm) = system_metrics {
        println!();
        println!("{}", "━".repeat(80).green());
        println!("{}", "SYSTEM METRICS".green().bold());
        println!("{}", "━".repeat(80).green());
        println!(
            "  Samples: {} (~{:.1}s)",
            sm.sample_count, sm.duration_seconds
        );
        println!();

        println!("{}", "  Efficiency Cores:".cyan().bold());
        println!(
            "    Freq (MHz):  Min: {:.0} | Mean: {:.0} | Median: {:.0} | Max: {:.0}",
            sm.efficiency_cores.freq_mhz_min,
            sm.efficiency_cores.freq_mhz_mean,
            sm.efficiency_cores.freq_mhz_median,
            sm.efficiency_cores.freq_mhz_max
        );
        println!(
            "    Usage (%):   Min: {:.1} | Mean: {:.1} | Median: {:.1} | Max: {:.1}",
            sm.efficiency_cores.usage_percent_min,
            sm.efficiency_cores.usage_percent_mean,
            sm.efficiency_cores.usage_percent_median,
            sm.efficiency_cores.usage_percent_max
        );
        println!();

        println!("{}", "  Performance Cores:".cyan().bold());
        println!(
            "    Freq (MHz):  Min: {:.0} | Mean: {:.0} | Median: {:.0} | Max: {:.0}",
            sm.performance_cores.freq_mhz_min,
            sm.performance_cores.freq_mhz_mean,
            sm.performance_cores.freq_mhz_median,
            sm.performance_cores.freq_mhz_max
        );
        println!(
            "    Usage (%):   Min: {:.1} | Mean: {:.1} | Median: {:.1} | Max: {:.1}",
            sm.performance_cores.usage_percent_min,
            sm.performance_cores.usage_percent_mean,
            sm.performance_cores.usage_percent_median,
            sm.performance_cores.usage_percent_max
        );
        println!();

        println!("{}", "  GPU:".cyan().bold());
        println!(
            "    Freq (MHz):  Min: {:.0} | Mean: {:.0} | Median: {:.0} | Max: {:.0}",
            sm.gpu.freq_mhz_min, sm.gpu.freq_mhz_mean, sm.gpu.freq_mhz_median, sm.gpu.freq_mhz_max
        );
        println!(
            "    Usage (%):   Min: {:.1} | Mean: {:.1} | Median: {:.1} | Max: {:.1}",
            sm.gpu.usage_percent_min,
            sm.gpu.usage_percent_mean,
            sm.gpu.usage_percent_median,
            sm.gpu.usage_percent_max
        );
        println!();

        println!("{}", "  Memory:".cyan().bold());
        println!(
            "    RAM (GB):    Min: {:.2} | Mean: {:.2} | Median: {:.2} | Max: {:.2}",
            sm.memory.ram_usage_gb_min,
            sm.memory.ram_usage_gb_mean,
            sm.memory.ram_usage_gb_median,
            sm.memory.ram_usage_gb_max
        );
        println!(
            "    Swap (GB):   Min: {:.2} | Mean: {:.2} | Median: {:.2} | Max: {:.2}",
            sm.memory.swap_usage_gb_min,
            sm.memory.swap_usage_gb_mean,
            sm.memory.swap_usage_gb_median,
            sm.memory.swap_usage_gb_max
        );
        println!();

        println!("{}", "  Power Consumption:".cyan().bold());
        println!(
            "    CPU (W):     Min: {:.2} | Mean: {:.2} | Median: {:.2} | Max: {:.2} | Total: {:.2} Wh",
            sm.power.cpu_watts_min,
            sm.power.cpu_watts_mean,
            sm.power.cpu_watts_median,
            sm.power.cpu_watts_max,
            sm.power.cpu_watts_total / 3600.0
        );
        println!(
            "    GPU (W):     Min: {:.2} | Mean: {:.2} | Median: {:.2} | Max: {:.2} | Total: {:.2} Wh",
            sm.power.gpu_watts_min,
            sm.power.gpu_watts_mean,
            sm.power.gpu_watts_median,
            sm.power.gpu_watts_max,
            sm.power.gpu_watts_total / 3600.0
        );
        println!(
            "    ANE (W):     Min: {:.2} | Mean: {:.2} | Median: {:.2} | Max: {:.2} | Total: {:.2} Wh",
            sm.power.ane_watts_min,
            sm.power.ane_watts_mean,
            sm.power.ane_watts_median,
            sm.power.ane_watts_max,
            sm.power.ane_watts_total / 3600.0
        );
        println!("{}", "━".repeat(80).green());
    }
}
