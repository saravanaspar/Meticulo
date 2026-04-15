"""Model manager - download, list, delete GGUF models."""

import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from .config import (
    add_custom_model,
    ensure_dirs,
    get_blobs_dir,
    get_model_registry,
    get_models_dir,
)


def _sizeof_fmt(num: float) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def _clean_model_name(filename: str) -> str:
    """
    Generic Ollama-style name: family:size-variant
    Examples: llama3.2:1b, qwen2.5:7b, gemma4:e2b
    """
    name = filename.replace(".gguf", "")
    
    # Remove quantization suffix
    name = re.sub(r'[-_]?q\d+_(?:k_\w+|0)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[-_]?fp16', '', name, flags=re.IGNORECASE)
    
    # Split by common separators
    parts = re.split(r'[-_\s]+', name)
    family = parts[0].lower() if parts else ""
    
    # If second part looks like a version number, append it
    if len(parts) > 1 and re.match(r'^\d+\.?\d*$', parts[1]):
        family += parts[1]
    
    # Extract size
    # First try to find e2b, e4b pattern (gemma4 style - before normal 1b, 2b)
    size_match = re.search(r'[eE](\d+)[bB]?', name)
    if size_match:
        size = f"e{size_match.group(1)}b"
    else:
        # Look forNxN pattern (mixtral 8x7b)
        mx_match = re.search(r'(\d+)[xX](\d+)[bB]?', name)
        if mx_match:
            size = f"{mx_match.group(1)}x{mx_match.group(2)}b"
        else:
            # Standard Nx pattern
            size_match = re.search(r'(\d+)[bB](?!\w)', name)
            size = size_match.group(1) + "b" if size_match else ""
    
    # Extract variant
    variant = ""
    if re.search(r'instruct', name, re.I):
        variant = "instruct"
    elif re.search(r'-it\b', name, re.I):
        variant = "it"
    elif re.search(r'\bchat\b', name, re.I):
        variant = "chat"
    elif re.search(r'\bpython\b', name, re.I):
        variant = "python"
    elif re.search(r'\bcode\b', name, re.I):
        variant = "code"
    elif re.search(r'\bmini\b', name, re.I):
        variant = "mini"
    
    # Build result
    if size:
        result = f"{family}:{size}"
    else:
        result = family
    
    if variant and variant not in result:
        result = f"{result}-{variant}"
    
    return result


def _download_with_progress(url: str, dest: Path, headers: Optional[dict] = None):
    """Download a file with a progress bar."""
    import urllib.request

    req = urllib.request.Request(url)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)

    with urllib.request.urlopen(req) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 1024 * 1024  # 1MB chunks

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".partial")

        start_time = time.time()
        with open(tmp, "wb") as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                elapsed = time.time() - start_time
                speed = downloaded / elapsed if elapsed > 0 else 0

                if total > 0:
                    pct = downloaded / total * 100
                    bar_len = 40
                    filled = int(bar_len * downloaded // total)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    sys.stdout.write(
                        f"\r  {bar} {pct:5.1f}% | "
                        f"{_sizeof_fmt(downloaded)}/{_sizeof_fmt(total)} | "
                        f"{_sizeof_fmt(speed)}/s"
                    )
                else:
                    sys.stdout.write(
                        f"\r  Downloaded: {_sizeof_fmt(downloaded)} | "
                        f"{_sizeof_fmt(speed)}/s"
                    )
                sys.stdout.flush()

        print()  # newline after progress bar
        tmp.rename(dest)


def resolve_model_source(model_name: str) -> tuple[str, str, str]:
    """
    Resolve a model name to (repo_id, filename, display_name).

    Supports:
      - Short names from registry: "llama3.1:8b"
      - HuggingFace URLs: "https://huggingface.co/user/repo/resolve/main/file.gguf"
      - HuggingFace repo:file format: "bartowski/Llama-3.1-8B-GGUF:file.gguf"
      - Direct GGUF file paths: "/path/to/model.gguf"
    """
    registry = get_model_registry()

    # Check registry first
    if model_name in registry:
        entry = registry[model_name]
        repo_and_file = entry.split(":")
        repo_id = repo_and_file[0]
        filename = repo_and_file[1]
        return repo_id, filename, model_name

    # HuggingFace URL
    if model_name.startswith("https://huggingface.co/"):
        parts = model_name.replace("https://huggingface.co/", "").split("/")
        if len(parts) >= 5 and parts[2] == "resolve":
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = "/".join(parts[4:])
            display = _clean_model_name(filename)
            return repo_id, filename, display

    # repo:file format
    if ":" in model_name and "/" in model_name.split(":")[0]:
        repo_id, filename = model_name.split(":", 1)
        display = _clean_model_name(filename)
        return repo_id, filename, display

    # Could also be just a repo name - we'll list files later
    if "/" in model_name and not model_name.startswith("/"):
        return model_name, "", model_name

    # Local file path
    if os.path.exists(model_name) and model_name.endswith(".gguf"):
        return "", model_name, Path(model_name).stem

    raise ValueError(
        f"Unknown model: '{model_name}'\n"
        f"Use 'meticulo list --available' to see available models, or provide:\n"
        f"  - A HuggingFace repo:  user/repo:filename.gguf\n"
        f"  - A HuggingFace URL:   https://huggingface.co/user/repo/resolve/main/file.gguf\n"
        f"  - A local GGUF file:   /path/to/model.gguf"
    )


def list_hf_repo_gguf_files(repo_id: str) -> list[dict]:
    """List GGUF files available in a HuggingFace repo."""
    import urllib.request

    api_url = f"https://huggingface.co/api/models/{quote(repo_id, safe='/')}"
    try:
        req = urllib.request.Request(api_url)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        raise RuntimeError(f"Failed to query HuggingFace API for {repo_id}: {e}")

    siblings = data.get("siblings", [])
    gguf_files = []
    for s in siblings:
        fname = s.get("rfilename", "")
        if fname.endswith(".gguf"):
            gguf_files.append({"filename": fname, "size": s.get("size")})

    return gguf_files


def pull_model(model_name: str, quant: Optional[str] = None) -> Path:
    """
    Download a model and return the path to the GGUF file.

    Args:
        model_name: Model identifier (registry name, HF repo:file, URL, or local path)
        quant: Optional quantization preference (e.g. "Q4_K_M", "Q8_0")
    """
    ensure_dirs()

    repo_id, filename, display = resolve_model_source(model_name)
    was_direct_hf_link = model_name.startswith("https://huggingface.co/")

    # Local file - just symlink/copy
    if not repo_id and os.path.exists(filename):
        local_path = Path(filename).resolve()
        model_dir = get_models_dir() / display
        model_dir.mkdir(parents=True, exist_ok=True)
        link_path = model_dir / local_path.name
        if not link_path.exists():
            link_path.symlink_to(local_path)
        _save_model_manifest(display, str(link_path), {"source": "local", "path": str(local_path)})
        print(f"Linked local model: {local_path}")
        return link_path

    # If no filename specified, list available and pick best match
    if not filename:
        print(f"Listing GGUF files in {repo_id}...")
        files = list_hf_repo_gguf_files(repo_id)
        if not files:
            raise RuntimeError(f"No GGUF files found in {repo_id}")

        target_quant = quant or "Q4_K_M"
        # Pick the file matching the quant preference
        match = None
        for f in files:
            if target_quant in f["filename"]:
                match = f
                break
        if not match:
            # Fall back to first GGUF file
            match = files[0]
            print(f"  Quant '{target_quant}' not found, using: {match['filename']}")

        filename = match["filename"]
        display = _clean_model_name(filename)

    # Check if already downloaded
    model_dir = get_models_dir() / display
    model_path = model_dir / filename
    if model_path.exists():
        print(f"Model already downloaded: {model_path}")
        return model_path

    # Download from HuggingFace
    url = f"https://huggingface.co/{quote(repo_id, safe='/')}/resolve/main/{quote(filename, safe='/')}"

    print(f"Pulling {display}")
    print(f"  From: {repo_id}")
    print(f"  File: {filename}")
    print()

    model_dir.mkdir(parents=True, exist_ok=True)
    _download_with_progress(url, model_path)

    # Save manifest
    _save_model_manifest(display, str(model_path), {
        "source": "huggingface",
        "repo_id": repo_id,
        "filename": filename,
        "url": url,
        "size": model_path.stat().st_size,
        "pulled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    print(f"\nDone! Model saved to: {model_path}")

    # If user pulled from a raw HF URL, persist a local alias automatically.
    if was_direct_hf_link:
        alias = display  # Already cleaned by _clean_model_name
        try:
            add_custom_model(alias, model_name)
            print(f"Saved alias: {alias} -> custom_models.json")
        except Exception:
            pass

    return model_path


def _normalize_alias(name: str) -> str:
    """Generate a friendly alias for user-provided model sources."""
    alias = name.strip().lower()
    for ch in (" ", "/", "\\", "(", ")", "[", "]", "{", "}", "'", '"'):
        alias = alias.replace(ch, "-")
    while "--" in alias:
        alias = alias.replace("--", "-")
    return alias.strip("-")


def add_model_alias(alias: str, source: str):
    """Add a custom model alias backed by HF URL or repo:file source."""
    # Validate source is resolvable
    resolve_model_source(source)
    add_custom_model(alias, source)


def _save_model_manifest(name: str, model_path: str, metadata: dict):
    """Save model metadata."""
    model_dir = get_models_dir() / name
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": name,
        "model_path": model_path,
        **metadata,
    }
    with open(model_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def list_models() -> list[dict]:
    """List all downloaded models."""
    ensure_dirs()
    models = []
    models_dir = get_models_dir()
    if not models_dir.exists():
        return models

    for entry in sorted(models_dir.iterdir()):
        manifest_path = entry / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            # Get actual file size
            model_path = Path(manifest.get("model_path", ""))
            if model_path.exists():
                manifest["size"] = model_path.stat().st_size
                manifest["size_human"] = _sizeof_fmt(model_path.stat().st_size)
            models.append(manifest)
    return models


def get_model_path(model_name: str) -> Optional[Path]:
    """Get the path to a downloaded model's GGUF file. Returns None if not found."""
    ensure_dirs()

    # Direct path to GGUF file
    if model_name.endswith(".gguf") and os.path.exists(model_name):
        return Path(model_name)

    # Check by directory name first (including colon-separated names like qwen2.5:0.5b)
    model_dir = get_models_dir() / model_name
    manifest_path = model_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        p = Path(manifest.get("model_path", ""))
        if p.exists():
            return p

    registry = get_model_registry()

    # Check registry name mapping
    if model_name in registry:
        entry = registry[model_name]
        filename = entry.split(":")[1]
        display = _clean_model_name(filename)
        model_dir = get_models_dir() / display
        model_path = model_dir / filename
        if model_path.exists():
            return model_path
        return None

    # Check by display name
    model_dir = get_models_dir() / model_name
    manifest_path = model_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        p = Path(manifest.get("model_path", ""))
        if p.exists():
            return p

    # Search all models
    for m in list_models():
        if m.get("name") == model_name:
            p = Path(m.get("model_path", ""))
            if p.exists():
                return p

    return None


def delete_model(model_name: str) -> bool:
    """Delete a downloaded model."""
    ensure_dirs()

    # Find the model directory
    candidates = []

    registry = get_model_registry()

    # Check registry mapping
    if model_name in registry:
        entry = registry[model_name]
        filename = entry.split(":")[1]
        display = filename.replace(".gguf", "")
        candidates.append(get_models_dir() / display)

    # Check direct name
    candidates.append(get_models_dir() / model_name)

    for model_dir in candidates:
        if model_dir.exists():
            shutil.rmtree(model_dir)
            print(f"Deleted: {model_name}")
            return True

    print(f"Model not found: {model_name}")
    return False


def list_available_models():
    """Print available models from registry with hardware recommendations."""
    registry = get_model_registry()
    
    # Filter out internal keys and separators
    priority_order = {
        "llama": 1,
        "gemma": 2,
        "qwen": 3,
        "phi": 4,
        "mistral": 5,
        "deepseek": 6,
    }
    
    def sort_key(name):
        name_lower = name.lower()
        for prefix, prio in priority_order.items():
            if name_lower.startswith(prefix):
                return (0, prio, name_lower)
        if name_lower.startswith("__") or "=====" in name:
            return (2, 999, name_lower)
        return (1, 99, name_lower)
    
    sorted_models = sorted(registry.items(), key=lambda x: sort_key(x[0]))
    
    # Filter out separators, notes, and embedding models for regular listing
    regular_models = [
        (n, s) for n, s in sorted_models 
        if not n.startswith("__") 
        and "=====" not in n 
        and "embedding" not in n.lower()
    ]
    
    print("Available models (use 'meticulo pull <name>'):\n")
    print("  Recommended for your hardware (4GB VRAM, 32GB RAM):")
    print("  " + "─" * 48)
    print("  llama3.2:1b      Meta Llama 3.2 - 1B (lightweight)")
    print("  qwen2.5:1.5b     Qwen 2.5 - 1.5B (great for code)")
    print("  gemma3:1b        Google Gemma 3 - 1B (multilingual)")
    print("  phi4-mini:3.8b   Microsoft Phi-4 Mini (reasoning)")
    print("  qwen2.5:3b       Qwen 2.5 - 3B (better quality)")
    print("  llama3.2:3b      Meta Llama 3.2 - 3B (balanced)")
    print()
    print("  All available models:")
    print("  " + "─" * 48)
    
    for name, source in regular_models:
        repo = source.split(":")[0]
        print(f"  {name:<25} {repo}")
    
    print()


def list_embedding_models():
    """Print available embedding models."""
    registry = get_model_registry()
    
    # Get all embedding models
    embedding_models = [
        (n, s) for n, s in registry.items()
        if "embedding" in n.lower() or "embed-text" in n.lower()
    ]
    
    print("Available embedding models (use 'meticulo pull <name>'):\n")
    print("  Lightweight (run alongside chat models):")
    print("  " + "─" * 48)
    print("  nomic-embed-text:v1.5-137m   Nomic - 137M (fast)")
    print("  qwen3-embedding:0.6b         Qwen - 0.6B (multilingual)")
    print("  bge-small-en:v1.5-33m        BGE - 33M (english)")
    print("  all-minilm-l6:22m            MiniLM - 22M (fast)")
    print()
    print("  Higher quality:")
    print("  " + "─" * 48)
    print("  qwen3-embedding:4b           Qwen - 4B (balanced)")
    print("  bge-base-en:v1.5-109m        BGE - 109M (better)")
    print("  nomic-embed-text:v2-moe-475m Nomic - 475M (best)")
    print()
    print("  Run with: meticulo serve-embed --model <name>")
    print("  Then use http://localhost:8081/v1/embeddings")
    
    print(f"\nOr pull any GGUF from HuggingFace:")
    print(f"  meticulo pull user/repo:filename.gguf")
    print(f"  meticulo pull https://huggingface.co/user/repo/resolve/main/file.gguf")
