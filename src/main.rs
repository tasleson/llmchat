use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use rustyline::{history::DefaultHistory, Editor};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    env, fs,
    io::{self, Read, Write},
    path::PathBuf,
    process::Command,
    time::Instant,
};
use tempfile::NamedTempFile;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    session: Option<PathBuf>,

    #[arg(short, long, default_value = "local-model")]
    model: String,

    #[arg(long, default_value = "http://localhost:1234/v1")]
    endpoint: String,

    /// Optional one-shot prompt
    prompt: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let mut messages = load_session(&args.session)?;
    let mut system_prompt: Option<String> = None;

    // --- One-shot from CLI argument ---
    if let Some(ref p) = args.prompt {
        handle_prompt(p.clone(), &mut messages, &args, &system_prompt).await?;
        save_session(&args.session, &messages)?;
        return Ok(());
    }

    // --- One-shot from pipe ---
    if !atty::is(atty::Stream::Stdin) {
        let mut input = String::new();
        io::stdin().read_to_string(&mut input)?;
        handle_prompt(input, &mut messages, &args, &system_prompt).await?;
        save_session(&args.session, &messages)?;
        return Ok(());
    }

    // --- Interactive REPL ---
    let mut rl = Editor::<(), DefaultHistory>::new()?;
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
                    if handle_command(input, &mut messages, &mut system_prompt, &args).await? {
                        break;
                    }
                    save_session(&args.session, &messages)?;
                    continue;
                }

                handle_prompt(input.to_string(), &mut messages, &args, &system_prompt).await?;
                save_session(&args.session, &messages)?;
            }
            Err(_) => break,
        }
    }

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

fn open_editor() -> Result<String> {
    let tmp = NamedTempFile::new()?;
    let editor = env::var("EDITOR").unwrap_or_else(|_| "vim".into());
    Command::new(editor).arg(tmp.path()).status()?;
    Ok(fs::read_to_string(tmp.path())?)
}

async fn handle_command(
    cmd: &str,
    messages: &mut Vec<Message>,
    system_prompt: &mut Option<String>,
    args: &Args,
) -> Result<bool> {
    let parts: Vec<&str> = cmd.splitn(2, ' ').collect();

    match parts[0] {
        "/exit" => return Ok(true),
        "/clear" => {
            messages.clear();
            println!("Session cleared.");
        }
        "/edit" => {
            let content = open_editor()?;
            handle_prompt(content, messages, args, system_prompt).await?;
        }
        "/system" => {
            if parts.len() > 1 {
                *system_prompt = Some(parts[1].to_string());
                println!("System prompt set.");
            }
        }
        _ => println!("Unknown command, try /exit, /clear, /edit, /system"),
    }

    Ok(false)
}

async fn handle_prompt(
    input: String,
    messages: &mut Vec<Message>,
    args: &Args,
    system_prompt: &Option<String>,
) -> Result<()> {
    messages.push(Message {
        role: "user".into(),
        content: input,
    });

    let mut payload_msgs = vec![];

    if let Some(sys) = system_prompt {
        payload_msgs.push(json!({ "role": "system", "content": sys }));
    }

    for m in messages.iter() {
        payload_msgs.push(json!(m));
    }

    let client = reqwest::Client::new();
    let url = format!("{}/chat/completions", args.endpoint);

    // Create spinner
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );
    spinner.set_message("Waiting for response...");
    spinner.enable_steady_tick(std::time::Duration::from_millis(80));

    let start_time = Instant::now();
    let mut first_token_time: Option<Instant> = None;
    let mut token_count = 0;

    let response = client
        .post(&url)
        .json(&json!({
            "model": args.model,
            "messages": payload_msgs,
            "stream": true
        }))
        .send()
        .await;

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
                    if let Some(token) = v["choices"][0]["delta"]["content"].as_str() {
                        // Clear spinner on first token
                        if first_token_time.is_none() {
                            spinner.finish_and_clear();
                            first_token_time = Some(Instant::now());
                        }

                        print!("{}", token.bright_cyan());
                        io::stdout().flush().unwrap();
                        assistant_reply.push_str(token);
                        token_count += 1;
                    }
                }
            }
        }
    }

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

    // Print statistics
    let mut stats = Vec::new();
    if let Some(ttft_duration) = ttft {
        stats.push(format!("TTFT: {:.2}s", ttft_duration.as_secs_f64()));
    }
    stats.push(format!("Total: {:.2}s", total_time.as_secs_f64()));
    stats.push(format!("Tokens: {}", token_count));

    if let Some(ttft_duration) = ttft {
        let generation_time = total_time.as_secs_f64() - ttft_duration.as_secs_f64();
        if generation_time > 0.0 && token_count > 0 {
            let tps = token_count as f64 / generation_time;
            stats.push(format!("Speed: {:.1} tok/s", tps));
        }
    }

    println!("{}", format!("[{}]", stats.join(" | ")).dimmed());

    messages.push(Message {
        role: "assistant".into(),
        content: assistant_reply,
    });

    Ok(())
}
