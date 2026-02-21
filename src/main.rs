use clap::Parser;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    env, fs,
    io::{self, Write},
    path::PathBuf,
    process::Command,
};
use tempfile::NamedTempFile;

#[derive(Parser)]
struct Args {
    /// Continue existing session file
    #[arg(short, long)]
    session: Option<PathBuf>,

    /// Model name (must match LM Studio loaded model)
    #[arg(short, long, default_value = "local-model")]
    model: String,

    /// LM Studio endpoint
    #[arg(long, default_value = "http://localhost:1234/v1")]
    endpoint: String,
}

#[derive(Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut messages: Vec<Message> = if let Some(ref path) = args.session {
        if path.exists() {
            let data = fs::read_to_string(path)?;
            serde_json::from_str(&data)?
        } else {
            vec![]
        }
    } else {
        vec![]
    };

    // --- Open editor ---
    let tmp = NamedTempFile::new()?;
    let editor = env::var("EDITOR").unwrap_or_else(|_| "vim".into());

    Command::new(editor).arg(tmp.path()).status()?;

    let prompt = fs::read_to_string(tmp.path())?;
    if prompt.trim().is_empty() {
        println!("Empty prompt. Aborting.");
        return Ok(());
    }

    messages.push(Message {
        role: "user".into(),
        content: prompt.clone(),
    });

    // --- Send request ---
    let client = reqwest::Client::new();
    let url = format!("{}/chat/completions", args.endpoint);

    let response = client
        .post(url)
        .json(&json!({
            "model": args.model,
            "messages": messages,
            "stream": true
        }))
        .send()
        .await?;

    let mut stream = response.bytes_stream();

    let mut assistant_reply = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);

        for line in text.lines() {
            if let Some(payload) = line.strip_prefix("data: ") {
                if payload == "[DONE]" {
                    break;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) {
                    if let Some(token) = v["choices"][0]["delta"]["content"].as_str() {
                        print!("{}", token);
                        io::stdout().flush().unwrap();
                        assistant_reply.push_str(token);
                    }
                }
            }
        }
    }

    println!();

    messages.push(Message {
        role: "assistant".into(),
        content: assistant_reply,
    });

    // --- Save session ---
    if let Some(path) = args.session {
        fs::write(path, serde_json::to_string_pretty(&messages)?)?;
    }

    Ok(())
}
