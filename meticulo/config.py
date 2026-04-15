"""Configuration and paths for Meticulo."""

import os
import json
from pathlib import Path
from typing import Dict


def get_meticulo_home() -> Path:
    """Get the Meticulo home directory."""
    return Path(os.environ.get("METICULO_HOME", Path.home() / ".meticulo"))


def get_models_dir() -> Path:
    return get_meticulo_home() / "models"


def get_blobs_dir() -> Path:
    return get_meticulo_home() / "blobs"


def get_config_path() -> Path:
    return get_meticulo_home() / "config.json"


def get_custom_models_path() -> Path:
    """User-managed model source registry file."""
    return get_meticulo_home() / "custom_models.json"


def get_popular_models_path() -> Path:
    """Packaged model catalog file shipped with Meticulo."""
    return Path(__file__).resolve().parent / "popular_models.json"


def get_engine_dir() -> Path:
    return get_meticulo_home() / "engine"


def get_server_path() -> Path:
    """Get the path to meticulo-server binary."""
    build_dir = get_engine_dir() / "build" / "bin"
    for name in ["meticulo-server", "server"]:
        p = build_dir / name
        if p.exists():
            return p
    import shutil
    sys_path = shutil.which("meticulo-server")
    if sys_path:
        return Path(sys_path)
    return build_dir / "meticulo-server"


def get_cli_path() -> Path:
    """Get the path to meticulo-cli binary."""
    build_dir = get_engine_dir() / "build" / "bin"
    for name in ["meticulo-cli", "main"]:
        p = build_dir / name
        if p.exists():
            return p
    import shutil
    sys_path = shutil.which("meticulo-cli")
    if sys_path:
        return Path(sys_path)
    return build_dir / "meticulo-cli"


def get_completion_path() -> Path:
    """Get the path to llama-completion binary."""
    return get_engine_dir() / "build" / "bin" / "llama-completion"


def ensure_dirs():
    """Create all necessary directories."""
    get_models_dir().mkdir(parents=True, exist_ok=True)
    get_blobs_dir().mkdir(parents=True, exist_ok=True)


def _load_registry_file(path: Path) -> Dict[str, str]:
    """Load a JSON dict of model aliases to model sources."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}
    return {}


def load_custom_models() -> Dict[str, str]:
    """Load user-defined model aliases from ~/.meticulo/custom_models.json."""
    return _load_registry_file(get_custom_models_path())


def save_custom_models(models: Dict[str, str]):
    """Persist user-defined model aliases."""
    path = get_custom_models_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(models, f, indent=2, sort_keys=True)


def add_custom_model(alias: str, source: str):
    """Add or update a user-defined model alias."""
    models = load_custom_models()
    models[alias] = source
    save_custom_models(models)


def get_model_registry() -> Dict[str, str]:
    """Merged model registry: built-in + packaged popular + user custom."""
    merged: Dict[str, str] = dict(MODEL_REGISTRY)
    merged.update(_load_registry_file(get_popular_models_path()))
    merged.update(load_custom_models())
    return merged


def load_config() -> dict:
    """Load user configuration."""
    path = get_config_path()
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"auto_update": True}  # Default enabled


def save_config(config: dict):
    """Save user configuration."""
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure auto_update has a default
    if "auto_update" not in config:
        config["auto_update"] = True
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


# Default server settings - no artificial limits
DEFAULTS = {
    "host": "127.0.0.1",
    "port": 22434,             # API server port (22434 to avoid Ollama's 11434)
    "engine_port": 22435,     # Backend engine port (port + 1)
    "n_gpu_layers": -1,          # Offload ALL layers to GPU by default
    "ctx_size": 0,               # 0 = use model's full context size
    "n_batch": 2048,             # Large batch for throughput
    "n_ubatch": 512,             # Physical batch size
    "n_parallel": 1,             # Parallel sequences
    "flash_attn": True,          # Flash attention for speed
    "mlock": True,               # Lock model in RAM
    "no_mmap": False,            # Use mmap by default
    "cont_batching": True,       # Continuous batching
}


# Known model registries (HuggingFace repos with GGUF files)
MODEL_REGISTRY = {
    # Llama models
    "llama3.2:1b": "bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "llama3.2:3b": "bartowski/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "llama3.1:8b": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "llama3.1:70b": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF:Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
    # Mistral
    "mistral:7b": "bartowski/Mistral-7B-Instruct-v0.3-GGUF:Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    "mixtral:8x7b": "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF:Mixtral-8x7B-Instruct-v0.1-Q4_K_M.gguf",
    # Phi
    "phi3:mini": "bartowski/Phi-3.1-mini-4k-instruct-GGUF:Phi-3.1-mini-4k-instruct-Q4_K_M.gguf",
    "phi3:medium": "bartowski/Phi-3-medium-4k-instruct-GGUF:Phi-3-medium-4k-instruct-Q4_K_M.gguf",
    # Qwen
    "qwen2.5:7b": "bartowski/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "qwen2.5:14b": "bartowski/Qwen2.5-14B-Instruct-GGUF:Qwen2.5-14B-Instruct-Q4_K_M.gguf",
    "qwen2.5:32b": "bartowski/Qwen2.5-32B-Instruct-GGUF:Qwen2.5-32B-Instruct-Q4_K_M.gguf",
    "qwen2.5:72b": "bartowski/Qwen2.5-72B-Instruct-GGUF:Qwen2.5-72B-Instruct-Q4_K_M.gguf",
    # Gemma
    "gemma2:9b": "bartowski/gemma-2-9b-it-GGUF:gemma-2-9b-it-Q4_K_M.gguf",
    "gemma2:27b": "bartowski/gemma-2-27b-it-GGUF:gemma-2-27b-it-Q4_K_M.gguf",
    # CodeLlama
    "codellama:7b": "bartowski/CodeLlama-7B-Instruct-GGUF:CodeLlama-7B-Instruct-Q4_K_M.gguf",
    "codellama:13b": "bartowski/CodeLlama-13B-Instruct-GGUF:CodeLlama-13B-Instruct-Q4_K_M.gguf",
    # DeepSeek
    "deepseek-coder:6.7b": "bartowski/deepseek-coder-6.7B-instruct-GGUF:deepseek-coder-6.7B-instruct-Q4_K_M.gguf",
    "deepseek-r1:7b": "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
    "deepseek-r1:14b": "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF:DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
    "deepseek-r1:32b": "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF:DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
}
