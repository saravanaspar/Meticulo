mod api;
mod app;
mod hardware;
mod service;
mod ui;

use app::App;
use std::env;

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("--install") => {
            if let Err(e) = service::install_service() {
                eprintln!("Failed to install service: {e}");
                std::process::exit(1);
            }
            println!("Meticulo service installed and enabled on startup.");
        }
        Some("--uninstall") => {
            if let Err(e) = service::uninstall_service() {
                eprintln!("Failed to uninstall service: {e}");
                std::process::exit(1);
            }
            println!("Meticulo service removed.");
        }
        Some("--daemon") | Some("-d") => {
            if let Err(e) = service::run_daemon().await {
                eprintln!("Daemon error: {e}");
                std::process::exit(1);
            }
        }
        Some("--help") | Some("-h") => {
            print_help();
        }
        _ => {
            let mut app = App::new().await;
            if let Err(e) = app.run().await {
                // Restore terminal before printing error
                let _ = crossterm::terminal::disable_raw_mode();
                let _ = crossterm::execute!(
                    std::io::stdout(),
                    crossterm::terminal::LeaveAlternateScreen
                );
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
    }
}

fn print_help() {
    println!(
        r#"
  ╔══════════════════════════════════════╗
  ║         M E T I C U L O  TUI        ║
  ╚══════════════════════════════════════╝

  USAGE:
    meticulo-tui              Launch chat interface (auto-starts server)
    meticulo-tui --install    Install as system service
    meticulo-tui --uninstall  Remove system service
    meticulo-tui --daemon     Run as background daemon
    meticulo-tui --help       Show this help

  SHORTCUTS (in TUI):
    Ctrl+K       Select model (with search)
    Ctrl+P       Pull new model (search 100+ models)
    Ctrl+R       Recommended models for your hardware
    Ctrl+S       Inference settings (temp, top-k, top-p, system prompt)
    Ctrl+N       New conversation
    Ctrl+L       Clear chat
    Tab          Switch panels
    Enter        Send message / confirm
    PgUp/PgDn    Scroll chat
    Esc          Close popup
    Ctrl+C       Quit

  PORT: 22434 (avoids conflict with Ollama on 11434)
"#
    );
}
