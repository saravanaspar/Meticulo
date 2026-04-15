use std::process::Command;
use sysinfo::System;

#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub cpu_name: String,
    pub cpu_cores: usize,
    pub cpu_arch: String,
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub gpu: Option<GpuInfo>,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub vram_mb: u64,
    pub backend: String,
}

#[derive(Debug, Clone)]
pub struct ModelRecommendation {
    pub name: String,
    pub params: String,
    pub quant: String,
    pub size_gb: f64,
    pub reason: String,
}

impl HardwareInfo {
    pub fn detect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        let cpu_name = sys
            .cpus()
            .first()
            .map(|c| c.brand().to_string())
            .unwrap_or_else(|| "Unknown CPU".into());

        let cpu_cores = sys.physical_core_count().unwrap_or(1);
        let cpu_arch = std::env::consts::ARCH.to_string();
        let total_ram_mb = sys.total_memory() / (1024 * 1024);
        let available_ram_mb = sys.available_memory() / (1024 * 1024);

        let gpu = detect_gpu();

        HardwareInfo {
            cpu_name,
            cpu_cores,
            cpu_arch,
            total_ram_mb,
            available_ram_mb,
            gpu,
        }
    }

    pub fn recommend_models(&self) -> Vec<ModelRecommendation> {
        let vram_mb = self.gpu.as_ref().map(|g| g.vram_mb).unwrap_or(0);
        let ram_mb = self.total_ram_mb;
        let mut recs = Vec::new();

        if vram_mb > 0 {
            match vram_mb {
                0..=2048 => {
                    recs.push(ModelRecommendation {
                        name: "llama3.2:1b".into(),
                        params: "1B".into(),
                        quant: "Q4_K_M".into(),
                        size_gb: 0.7,
                        reason: "Minimal VRAM".into(),
                    });
                    recs.push(ModelRecommendation {
                        name: "qwen2.5:1.5b".into(),
                        params: "1.5B".into(),
                        quant: "Q4_K_M".into(),
                        size_gb: 1.0,
                        reason: "Code capable".into(),
                    });
                    recs.push(ModelRecommendation {
                        name: "gemma3:1b".into(),
                        params: "1B".into(),
                        quant: "Q4_K_M".into(),
                        size_gb: 0.8,
                        reason: "Fast, multilingual".into(),
                    });
                }
                2049..=4096 => {
                    recs.push(ModelRecommendation {
                        name: "llama3.2:3b".into(),
                        params: "3B".into(),
                        quant: "Q4_K_M".into(),
                        size_gb: 2.0,
                        reason: "Best quality/VRAM".into(),
                    });
                    recs.push(ModelRecommendation {
                        name: "qwen2.5:3b".into(),
                        params: "3B".into(),
                        quant: "Q4_K_M".into(),
                        size_gb: 2.0,
                        reason: "Coding champion".into(),
                    });
                    recs.push(ModelRecommendation {
                        name: "phi4-mini:3.8b".into(),
                        params: "3.8B".into(),
                        quant: "Q4_K_M".into(),
                        size_gb: 2.5,
                        reason: "Reasoning powerhouse".into(),
                    });
                }
                4097..=6144 => {
                    recs.push(ModelRecommendation {
                        name: "llama3.1:8b".into(),
                        params: "8B".into(),
                        quant: "Q4_K_M".into(),
                        size_gb: 4.9,
                        reason: "Best overall".into(),
                    });
                    recs.push(ModelRecommendation {
                        name: "qwen2.5:7b".into(),
                        params: "7B".into(),
                        quant: "Q4_K_M".into(),
                        size_gb: 4.7,
                        reason: "Coding king".into(),
                    });
                    recs.push(ModelRecommendation {
                        name: "mistral:7b".into(),
                        params: "7B".into(),
                        quant: "Q4_K_M".into(),
                        size_gb: 4.4,
                        reason: "Lightning fast".into(),
                    });
                }
                6145..=8192 => {
                    recs.push(ModelRecommendation {
                        name: "llama3.1:8b".into(),
                        params: "8B".into(),
                        quant: "Q4_K_M".into(),
                        size_gb: 4.9,
                        reason: "Premium quality".into(),
                    });
                    recs.push(ModelRecommendation {
                        name: "qwen2.5:14b".into(),
                        params: "14B".into(),
                        quant: "Q4_K_M".into(),
                        size_gb: 9.0,
                        reason: "14B fits perfectly".into(),
                    });
                    recs.push(ModelRecommendation {
                        name: "deepseek-r1:14b".into(),
                        params: "14B".into(),
                        quant: "Q4_K_M".into(),
                        size_gb: 9.0,
                        reason: "Advanced reasoning".into(),
                    });
                }
                8193..=12288 => {
                    recs.push(ModelRecommendation {
                        name: "qwen2.5:14b".into(),
                        params: "14B".into(),
                        quant: "Q4_K_M".into(),
                        size_gb: 9.0,
                        reason: "Sweet spot".into(),
                    });
                    recs.push(ModelRecommendation {
                        name: "deepseek-r1:14b".into(),
                        params: "14B".into(),
                        quant: "Q4_K_M".into(),
                        size_gb: 9.0,
                        reason: "Top reasoning".into(),
                    });
                    recs.push(ModelRecommendation {
                        name: "qwen2.5:32b".into(),
                        params: "32B".into(),
                        quant: "Q4_K_M".into(),
                        size_gb: 20.0,
                        reason: "32B class!".into(),
                    });
                }
                _ => {
                    recs.push(ModelRecommendation {
                        name: "qwen2.5:32b".into(),
                        params: "32B".into(),
                        quant: "Q4_K_M".into(),
                        size_gb: 20.0,
                        reason: "32B fits easily".into(),
                    });
                    recs.push(ModelRecommendation {
                        name: "deepseek-r1:32b".into(),
                        params: "32B".into(),
                        quant: "Q4_K_M".into(),
                        size_gb: 20.0,
                        reason: "Full reasoning power".into(),
                    });
                    recs.push(ModelRecommendation {
                        name: "llama3.1:70b".into(),
                        params: "70B".into(),
                        quant: "Q4_K_M".into(),
                        size_gb: 40.0,
                        reason: "Flagship model".into(),
                    });
                }
            }
            return recs;
        }

        if ram_mb <= 1024 {
            recs.push(ModelRecommendation {
                name: "qwen2.5:0.5b".into(),
                params: "500M".into(),
                quant: "Q4_K_M".into(),
                size_gb: 0.4,
                reason: "Pi-compatible".into(),
            });
        } else if ram_mb <= 2048 {
            recs.push(ModelRecommendation {
                name: "qwen2.5:0.5b".into(),
                params: "500M".into(),
                quant: "Q4_K_M".into(),
                size_gb: 0.4,
                reason: "Only option".into(),
            });
            recs.push(ModelRecommendation {
                name: "qwen2.5:1.5b".into(),
                params: "1.5B".into(),
                quant: "Q4_K_M".into(),
                size_gb: 1.0,
                reason: "Stretch".into(),
            });
        } else if ram_mb <= 4096 {
            recs.push(ModelRecommendation {
                name: "qwen2.5:1.5b".into(),
                params: "1.5B".into(),
                quant: "Q4_K_M".into(),
                size_gb: 1.0,
                reason: "Best 4GB".into(),
            });
            recs.push(ModelRecommendation {
                name: "llama3.2:1b".into(),
                params: "1B".into(),
                quant: "Q4_K_M".into(),
                size_gb: 0.7,
                reason: "Light".into(),
            });
        } else if ram_mb <= 8192 {
            recs.push(ModelRecommendation {
                name: "llama3.2:3b".into(),
                params: "3B".into(),
                quant: "Q4_K_M".into(),
                size_gb: 2.0,
                reason: "Quality".into(),
            });
            recs.push(ModelRecommendation {
                name: "qwen2.5:3b".into(),
                params: "3B".into(),
                quant: "Q4_K_M".into(),
                size_gb: 2.0,
                reason: "Coding".into(),
            });
        } else if ram_mb <= 16384 {
            recs.push(ModelRecommendation {
                name: "llama3.1:8b".into(),
                params: "8B".into(),
                quant: "Q4_K_M".into(),
                size_gb: 4.9,
                reason: "Full 8B".into(),
            });
            recs.push(ModelRecommendation {
                name: "qwen2.5:7b".into(),
                params: "7B".into(),
                quant: "Q4_K_M".into(),
                size_gb: 4.7,
                reason: "Coding".into(),
            });
        } else if ram_mb <= 32768 {
            recs.push(ModelRecommendation {
                name: "llama3.1:8b".into(),
                params: "8B".into(),
                quant: "Q8_0".into(),
                size_gb: 8.5,
                reason: "High quality".into(),
            });
            recs.push(ModelRecommendation {
                name: "qwen2.5:14b".into(),
                params: "14B".into(),
                quant: "Q4_K_M".into(),
                size_gb: 9.0,
                reason: "14B".into(),
            });
        } else {
            recs.push(ModelRecommendation {
                name: "llama3.1:70b".into(),
                params: "70B".into(),
                quant: "Q4_K_M".into(),
                size_gb: 40.0,
                reason: "Powerhouse".into(),
            });
            recs.push(ModelRecommendation {
                name: "qwen2.5:32b".into(),
                params: "32B".into(),
                quant: "Q4_K_M".into(),
                size_gb: 20.0,
                reason: "Massive".into(),
            });
        }
        recs
    }

    pub fn summary_line(&self) -> String {
        let gpu_str = match &self.gpu {
            Some(g) => format!("{} ({}MB VRAM, {})", g.name, g.vram_mb, g.backend),
            None => "No GPU detected".into(),
        };
        format!(
            "{} | {}C | {}MB RAM | {}",
            self.cpu_name.trim(),
            self.cpu_cores,
            self.total_ram_mb,
            gpu_str
        )
    }
}

fn detect_gpu() -> Option<GpuInfo> {
    // Try NVIDIA first
    if let Some(gpu) = detect_nvidia() {
        return Some(gpu);
    }
    // Try AMD ROCm
    if let Some(gpu) = detect_amd() {
        return Some(gpu);
    }
    None
}

fn detect_nvidia() -> Option<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().next()?;
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() >= 2 {
        let name = parts[0].trim().to_string();
        let vram_mb: u64 = parts[1].trim().parse().unwrap_or(0);
        Some(GpuInfo {
            name,
            vram_mb,
            backend: "CUDA".into(),
        })
    } else {
        None
    }
}

fn detect_amd() -> Option<GpuInfo> {
    let output = Command::new("rocm-smi")
        .args(["--showproductname", "--showmeminfo", "vram"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    // Basic parsing - extract GPU name and VRAM
    let name = text
        .lines()
        .find(|l| l.contains("Card series"))
        .and_then(|l| l.split(':').nth(1))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "AMD GPU".into());

    let vram_mb = text
        .lines()
        .find(|l| l.contains("Total Memory"))
        .and_then(|l| l.split_whitespace().rev().nth(1))
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    Some(GpuInfo {
        name,
        vram_mb,
        backend: "ROCm".into(),
    })
}

pub fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.0} MB", bytes as f64 / 1_048_576.0)
    } else {
        format!("{:.0} KB", bytes as f64 / 1024.0)
    }
}
