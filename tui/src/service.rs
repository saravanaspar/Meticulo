use std::io;

/// Install Meticulo as a system service (auto-start on boot)
pub fn install_service() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "linux")]
    {
        install_systemd_service()?;
    }
    #[cfg(target_os = "windows")]
    {
        install_windows_service()?;
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        return Err("Service installation not supported on this OS".into());
    }
    Ok(())
}

/// Remove the system service
pub fn uninstall_service() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "linux")]
    {
        uninstall_systemd_service()?;
    }
    #[cfg(target_os = "windows")]
    {
        uninstall_windows_service()?;
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        return Err("Service removal not supported on this OS".into());
    }
    Ok(())
}

/// Run as a background daemon (called by the service manager or --daemon flag)
pub async fn run_daemon() -> Result<(), Box<dyn std::error::Error>> {
    // The daemon just ensures the meticulo server stays running
    // and restarts it if it crashes
    use std::process::Command;
    use tokio::time::{sleep, Duration};

    eprintln!("[meticulo-daemon] Starting...");

    loop {
        // Check if meticulo serve is running
        let health_ok = check_server_health().await;
        if !health_ok {
            eprintln!("[meticulo-daemon] Server not responding, starting...");
            // Try to start the server
            let _ = Command::new("meticulo")
                .args(["serve"])
                .spawn();
            // Wait for it to come up
            sleep(Duration::from_secs(5)).await;
        }
        sleep(Duration::from_secs(10)).await;
    }
}

async fn check_server_health() -> bool {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .unwrap();

    client
        .get("http://127.0.0.1:22434/api/health")
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

// ── Linux systemd ──────────────────────────────────────────
#[cfg(target_os = "linux")]
fn install_systemd_service() -> Result<(), Box<dyn std::error::Error>> {
    let exe_path = std::env::current_exe()?;
    let exe = exe_path.display();
    let user = std::env::var("USER").unwrap_or_else(|_| "root".into());

    let unit = format!(
        r#"[Unit]
Description=Meticulo LLM Service
After=network.target

[Service]
Type=simple
User={user}
ExecStart={exe} --daemon
Restart=on-failure
RestartSec=5
Environment=HOME=/home/{user}

[Install]
WantedBy=multi-user.target
"#
    );

    // Try user service first, fall back to system
    let user_dir = dirs::config_dir()
        .map(|d| d.join("systemd/user"))
        .unwrap_or_else(|| {
            std::path::PathBuf::from(format!("/home/{user}/.config/systemd/user"))
        });

    if let Ok(()) = install_systemd_user_service(&user_dir, &unit) {
        return Ok(());
    }

    // Fall back to system service (needs root)
    let service_path = "/etc/systemd/system/meticulo.service";
    std::fs::write(service_path, &unit).map_err(|e| {
        if e.kind() == io::ErrorKind::PermissionDenied {
            format!("Permission denied. Run with sudo or as root:\n  sudo {} --install", exe).into()
        } else {
            Box::new(e) as Box<dyn std::error::Error>
        }
    })?;

    run_cmd("systemctl", &["daemon-reload"])?;
    run_cmd("systemctl", &["enable", "meticulo"])?;
    run_cmd("systemctl", &["start", "meticulo"])?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn install_systemd_user_service(
    user_dir: &std::path::Path,
    unit: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(user_dir)?;
    let path = user_dir.join("meticulo.service");
    std::fs::write(&path, unit)?;

    run_cmd("systemctl", &["--user", "daemon-reload"])?;
    run_cmd("systemctl", &["--user", "enable", "meticulo"])?;
    run_cmd("systemctl", &["--user", "start", "meticulo"])?;

    // Enable lingering so service runs even when logged out
    let user = std::env::var("USER").unwrap_or_default();
    let _ = run_cmd("loginctl", &["enable-linger", &user]);

    eprintln!("Installed as user service: {}", path.display());
    Ok(())
}

#[cfg(target_os = "linux")]
fn uninstall_systemd_service() -> Result<(), Box<dyn std::error::Error>> {
    // Try user service first
    let _ = run_cmd("systemctl", &["--user", "stop", "meticulo"]);
    let _ = run_cmd("systemctl", &["--user", "disable", "meticulo"]);

    if let Some(config) = dirs::config_dir() {
        let path = config.join("systemd/user/meticulo.service");
        let _ = std::fs::remove_file(&path);
    }

    // Try system service
    let _ = run_cmd("systemctl", &["stop", "meticulo"]);
    let _ = run_cmd("systemctl", &["disable", "meticulo"]);
    let _ = std::fs::remove_file("/etc/systemd/system/meticulo.service");

    let _ = run_cmd("systemctl", &["daemon-reload"]);
    let _ = run_cmd("systemctl", &["--user", "daemon-reload"]);

    Ok(())
}

// ── Windows service ────────────────────────────────────────
#[cfg(target_os = "windows")]
fn install_windows_service() -> Result<(), Box<dyn std::error::Error>> {
    let exe_path = std::env::current_exe()?;
    let exe = exe_path.display().to_string();

    // Create a scheduled task that runs at startup
    // This is more reliable than Windows Services for user-level apps
    // and shows in the system tray area
    let task_xml = format!(
        r#"<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
  </Triggers>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Hidden>true</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
  </Settings>
  <Actions>
    <Exec>
      <Command>{exe}</Command>
      <Arguments>--daemon</Arguments>
    </Exec>
  </Actions>
</Task>"#
    );

    // Write task XML to temp file
    let temp_dir = std::env::temp_dir();
    let xml_path = temp_dir.join("meticulo_task.xml");
    std::fs::write(&xml_path, &task_xml)?;

    // Register with Task Scheduler
    let output = std::process::Command::new("schtasks")
        .args([
            "/Create",
            "/TN", "Meticulo",
            "/XML", &xml_path.display().to_string(),
            "/F",
        ])
        .output()?;

    std::fs::remove_file(&xml_path).ok();

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Failed to create scheduled task: {err}").into());
    }

    // Also add to startup registry for system tray icon
    add_to_windows_startup(&exe)?;

    Ok(())
}

#[cfg(target_os = "windows")]
fn add_to_windows_startup(exe: &str) -> Result<(), Box<dyn std::error::Error>> {
    use winreg::enums::*;
    use winreg::RegKey;

    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let (key, _) = hkcu.create_subkey("SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run")?;
    key.set_value("Meticulo", &format!("{exe} --daemon"))?;
    Ok(())
}

#[cfg(target_os = "windows")]
fn uninstall_windows_service() -> Result<(), Box<dyn std::error::Error>> {
    // Remove scheduled task
    let _ = std::process::Command::new("schtasks")
        .args(["/Delete", "/TN", "Meticulo", "/F"])
        .output();

    // Remove from startup registry
    #[cfg(target_os = "windows")]
    {
        use winreg::enums::*;
        use winreg::RegKey;
        let hkcu = RegKey::predef(HKEY_CURRENT_USER);
        if let Ok(key) = hkcu.open_subkey_with_flags(
            "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run",
            KEY_WRITE,
        ) {
            let _ = key.delete_value("Meticulo");
        }
    }

    Ok(())
}

fn run_cmd(cmd: &str, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let output = std::process::Command::new(cmd).args(args).output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("{cmd} failed: {stderr}").into());
    }
    Ok(())
}
