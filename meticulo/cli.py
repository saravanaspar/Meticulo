"""Meticulo CLI - Simple command-line interface for local LLM management."""

import argparse
import sys
import os
from pathlib import Path

from . import __version__


def cmd_pull(args):
    """Pull/download a model."""
    from .model_manager import pull_model
    try:
        pull_model(args.model, quant=args.quant)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_add_model(args):
    """Add a custom model alias from raw HuggingFace source."""
    from .model_manager import add_model_alias
    try:
        add_model_alias(args.name, args.source)
        print(f"Added alias: {args.name} -> {args.source}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_run(args):
    """Run a model interactively or start the server."""
    from .model_manager import get_model_path, pull_model
    from .runner import get_runner

    model_path = get_model_path(args.model)
    if not model_path:
        # Model not found - ask before pulling
        response = input(f"Model '{args.model}' not found. Pull it now? [y/N] ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return
        try:
            model_path = pull_model(args.model)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    runner = get_runner()

    kwargs = {}
    if args.gpu_layers is not None:
        kwargs["n_gpu_layers"] = args.gpu_layers
    if args.ctx_size is not None:
        kwargs["ctx_size"] = args.ctx_size
    if args.threads is not None:
        kwargs["threads"] = args.threads
    if args.batch_size is not None:
        kwargs["n_batch"] = args.batch_size
    if args.no_flash_attn:
        kwargs["flash_attn"] = False
    if args.no_mlock:
        kwargs["mlock"] = False
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature

    if args.prompt:
        # Single completion mode
        result = runner.run_completion(
            model_path,
            args.prompt,
            max_tokens=args.max_tokens or -1,
            temperature=args.temperature if args.temperature is not None else 0.7,
            **kwargs,
        )
        print(result)
    else:
        # Interactive chat mode
        runner.run_interactive(model_path, **kwargs)


def cmd_serve(args):
    """Start the API server."""
    from .server import start_api_server
    server_kwargs = {}
    if args.gpu_layers is not None:
        server_kwargs["n_gpu_layers"] = args.gpu_layers
    if args.ctx_size is not None:
        server_kwargs["ctx_size"] = args.ctx_size
    if args.threads is not None:
        server_kwargs["threads"] = args.threads
    if args.batch_size is not None:
        server_kwargs["n_batch"] = args.batch_size
    if args.no_flash_attn:
        server_kwargs["flash_attn"] = False
    if args.no_mlock:
        server_kwargs["mlock"] = False

    start_api_server(
        host=args.host,
        port=args.port,
        model=args.model,
        server_kwargs=server_kwargs,
    )


def cmd_serve_embed(args):
    """Start the embedding server."""
    from .server import start_embedding_server
    server_kwargs = {}
    if args.gpu_layers is not None:
        server_kwargs["n_gpu_layers"] = args.gpu_layers
    if args.threads is not None:
        server_kwargs["threads"] = args.threads
    if args.no_mlock:
        server_kwargs["mlock"] = False

    start_embedding_server(
        host=args.host,
        port=args.port,
        model=args.model,
        server_kwargs=server_kwargs,
    )


def cmd_list(args):
    """List downloaded models."""
    if args.embeddings:
        from .model_manager import list_embedding_models
        list_embedding_models()
        return
    
    if args.available:
        from .model_manager import list_available_models
        list_available_models()
        return

    from .model_manager import list_models
    models = list_models()
    if not models:
        print("No models downloaded.")
        print("Use 'meticulo pull <model>' to download a model.")
        print("Use 'meticulo list --available' to see available models.")
        return

    print(f"\n{'Name':<40} {'Size':<12} {'Source':<15} {'Pulled'}")
    print(f"{'─' * 40} {'─' * 12} {'─' * 15} {'─' * 20}")
    for m in models:
        name = m.get("name", "unknown")
        size = m.get("size_human", "?")
        source = m.get("source", "?")
        pulled = m.get("pulled_at", "?")[:10] if m.get("pulled_at") else "?"
        print(f"{name:<40} {size:<12} {source:<15} {pulled}")
    print()


def cmd_rm(args):
    """Remove a downloaded model."""
    from .model_manager import delete_model
    if not args.force:
        confirm = input(f"Delete model '{args.model}'? [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return
    delete_model(args.model)


def cmd_ps(args):
    """Show running models and servers."""
    from .runner import get_runner
    from .config import get_meticulo_home
    import os
    
    runner = get_runner()
    
    print(f"\n{'═' * 60}")
    print("  Meticulo Status")
    print(f"{'═' * 60}\n")
    
    # Check for running processes by looking at port files or processes
    home = get_meticulo_home()
    port_file = home / "running_server.txt"
    
    has_chat = False
    has_embed = False
    
    if runner.is_running():
        has_chat = True
        model_size = runner.model_path.stat().st_size / (1024*1024) if runner.model_path.exists() else 0
        print(f"  Chat Server:    RUNNING")
        print(f"    Model:       {runner.model_path.name}")
        print(f"    Size:        {model_size:.0f} MB")
        print(f"    Endpoint:    http://{runner.host}:{runner.port}")
    else:
        print(f"  Chat Server:    Stopped")
    
    # Check for embedding server (by looking for processes)
    import subprocess
    result = subprocess.run(["pgrep", "-f", "meticulo-server"], capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        # Check if it's embedding mode
        embed_result = subprocess.run(["pgrep", "-f", "--embedding"], capture_output=True, text=True)
        if embed_result.returncode == 0:
            has_embed = True
            print(f"  Embed Server:   RUNNING")
            print(f"    Port:        8081")
        else:
            print(f"  Embed Server:   Stopped")
    else:
        print(f"  Embed Server:   Stopped")
    
    # Show process info with memory
    print(f"\n  Process Info:")
    ps_result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    meticulo_procs = [l for l in ps_result.stdout.split('\n') if 'meticulo' in l.lower() and 'grep' not in l.lower()]
    if meticulo_procs:
        print(f"    {'PID':<8} {'CPU':<6} {'MEM':<6} {'RSS (MB)':<10} {'COMMAND'}")
        for p in meticulo_procs[:5]:
            parts = p.split()
            if len(parts) >= 11:
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                # Get RSS memory in MB
                rss_mb = 0
                try:
                    with open(f'/proc/{pid}/status') as f:
                        for line in f:
                            if line.startswith('VmRSS:'):
                                rss_kb = int(line.split()[1])
                                rss_mb = rss_kb / 1024
                                break
                except:
                    pass
                cmd = ' '.join(parts[10:13])
                cmd = ' '.join(parts[10:13])
                print(f"    {pid:<8} {cpu:<6} {mem:<6} {rss_mb:<10.0f} {cmd}")
    else:
        print("    (no processes)")
    
    # Show memory info when running
    if runner.is_running():
        print(f"\n  Memory Usage:")
        # Try to get GPU memory
        try:
            import subprocess
            nvidia = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            if nvidia.returncode == 0 and nvidia.stdout.strip():
                used, total = nvidia.stdout.strip().split(',')
                print(f"    GPU Memory:   {used.strip()} MB / {total.strip()} MB")
        except:
            pass
        print(f"    Use 'nvidia-smi' for detailed GPU memory")
    
    print(f"\n{'═' * 60}\n")


def cmd_stop(args):
    """Stop running model."""
    from .runner import get_runner
    runner = get_runner()
    runner.stop()


def cmd_setup(args):
    """Set up meticulo engine (build from source)."""
    from .setup_engine import setup_engine
    setup_engine(
        force=args.force,
        backend=args.backend,
    )


def cmd_tui(args):
    """Launch the TUI."""
    import subprocess
    tui_path = Path(__file__).parent.parent / "tui" / "target" / "release" / "meticulo-tui"
    if not tui_path.exists():
        print("TUI not built. Building...")
        result = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=Path(__file__).parent.parent / "tui",
            capture_output=True,
        )
        if result.returncode != 0:
            print("Failed to build TUI")
            sys.exit(1)
    
    if args.install:
        subprocess.run([str(tui_path), "--install"])
    elif args.uninstall:
        subprocess.run([str(tui_path), "--uninstall"])
    else:
        subprocess.run([str(tui_path)])


def cmd_info(args):
    """Show system info and configuration."""
    from .config import (
        get_meticulo_home,
        get_models_dir,
        get_engine_dir,
        load_config,
        DEFAULTS,
    )
    from .hardware import get_hardware_profile
    
    profile = get_hardware_profile()
    config = load_config()
    
    print(f"\n{'═' * 60}")
    print("  Meticulo Configuration")
    print(f"{'═' * 60}\n")
    
    print("  Paths:")
    print(f"    Home:       {get_meticulo_home()}")
    print(f"    Models:     {get_models_dir()}")
    print(f"    Engine:     {get_engine_dir()}")
    print()
    
    print("  Settings:")
    for k, v in config.items():
        print(f"    {k}: {v}")
    print()
    
    print("  Hardware:")
    print(f"    Backend:    {profile.best_backend}")
    print(f"    CPU:        {profile.cpu.name[:40] if profile.cpu.name else 'unknown'}")
    print(f"    Cores:      {profile.cpu.cores_physical} physical, {profile.cpu.cores_logical} logical")
    if profile.can_gpu and profile.gpus:
        for gpu in profile.gpus:
            print(f"    GPU:        {gpu.name} ({gpu.vram_mb/1024:.1f} GB)")
    print()
    
    print(f"{'═' * 60}\n")


def print_config():
    """Print current configuration."""
    from .config import load_config
    import json
    config = load_config()
    print(json.dumps(config, indent=2))


def set_config(key: str, value: str):
    """Set a nested config value using dot notation."""
    from .config import load_config, save_config
    config = load_config()
    
    # Handle nested keys like "inference.n_gpu_layers"
    parts = key.split(".")
    current = config
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    
    if value.lower() == "true":
        current[parts[-1]] = True
    elif value.lower() == "false":
        current[parts[-1]] = False
    elif value.isdigit():
        current[parts[-1]] = int(value)
    else:
        current[parts[-1]] = value
    
    save_config(config)
    print(f"Set {key} = {current[parts[-1]]}")


def cmd_preset(args):
    """Manage inference presets."""
    from .config import list_presets, get_preset, save_preset, delete_preset
    import json
    
    if args.preset_action == "list":
        presets = list_presets()
        if not presets:
            print("No presets saved. Use 'meticulo preset save <name> [--flags...]' to create one.")
            return
        print("Presets:")
        for name in presets:
            print(f"  {name}")
    
    elif args.preset_action == "get":
        if not args.name:
            print("Error: preset name required")
            return
        preset = get_preset(args.name)
        if preset:
            print(json.dumps(preset, indent=2))
        else:
            print(f"Preset '{args.name}' not found")
    
    elif args.preset_action == "save":
        if not args.name:
            print("Error: preset name required")
            return
        preset = {
            "n_gpu_layers": args.gpu_layers if args.gpu_layers is not None else -1,
            "ctx_size": args.ctx_size if args.ctx_size else 0,
            "n_batch": args.batch_size if args.batch_size else 2048,
            "n_ubatch": args.ubatch_size if args.ubatch_size else 512,
            "n_parallel": args.parallel if args.parallel else 1,
            "flash_attn": not args.no_flash_attn,
            "mlock": not args.no_mlock,
            "no_mmap": args.no_mmap,
        }
        save_preset(args.name, preset)
        print(f"Saved preset '{args.name}'")
    
    elif args.preset_action == "delete":
        if not args.name:
            print("Error: preset name required")
            return
        delete_preset(args.name)
        print(f"Deleted preset '{args.name}'")


def cmd_import(args):
    """Import models from Ollama."""
    from .config import scan_ollama_models, import_ollama_models, load_config
    import json
    
    if args.import_action == "scan":
        models = scan_ollama_models()
        if not models:
            print("No Ollama models found")
            return
        print("Ollama models:")
        for m in models:
            print(f"  {m['name']} -> {m['path']}")
    
    elif args.import_action == "import":
        imported = import_ollama_models()
        if imported:
            print(f"Imported {len(imported)} model(s)")
            for name in imported:
                print(f"  {name}")
        else:
            print("No new models to import")
    
    elif args.import_action == "list":
        config = load_config()
        ollama_dir = config.get("import", {}).get("ollama_dir", "~/.ollama/models")
        print(f"Ollama models directory: {ollama_dir}")


def main():
    parser = argparse.ArgumentParser(
        prog="meticulo",
        description="Meticulo - Zero-overhead local LLM runner",
    )
    parser.add_argument("--version", action="version", version=f"meticulo {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # === pull ===
    p_pull = subparsers.add_parser("pull", help="Download a model")
    p_pull.add_argument("model", help="Model name (e.g., llama3.1:8b, user/repo:file.gguf)")
    p_pull.add_argument("--quant", "-q", help="Quantization preference (e.g., Q4_K_M, Q8_0)")
    p_pull.set_defaults(func=cmd_pull)

    # === add-model ===
    p_add_model = subparsers.add_parser(
        "add-model",
        help="Add model alias from HuggingFace source (stored in ~/.meticulo/custom_models.json)",
    )
    p_add_model.add_argument("name", help="Alias name (e.g., gemma3:12b)")
    p_add_model.add_argument(
        "source",
        help="HuggingFace source: user/repo:file.gguf or full https://huggingface.co/.../file.gguf URL",
    )
    p_add_model.set_defaults(func=cmd_add_model)

    # === run ===
    p_run = subparsers.add_parser("run", help="Run a model (interactive chat or single prompt)")
    p_run.add_argument("model", help="Model name or path to GGUF file")
    p_run.add_argument("--prompt", "-p", help="Single prompt (non-interactive mode)")
    p_run.add_argument("--max-tokens", type=int, help="Max tokens to generate (-1 = unlimited)")
    p_run.add_argument("--temperature", "-t", type=float, help="Temperature (default: 0.7)")
    p_run.add_argument("--top-k", type=int, help="Top-k sampling")
    p_run.add_argument("--top-p", type=float, help="Top-p sampling")
    p_run.add_argument("--gpu-layers", "-ngl", type=int, help="GPU layers (-1 = all, default)")
    p_run.add_argument("--ctx-size", "-c", type=int, help="Context size (0 = model default)")
    p_run.add_argument("--threads", type=int, help="CPU threads (default: all cores)")
    p_run.add_argument("--batch-size", type=int, help="Batch size (default: 2048)")
    p_run.add_argument("--no-flash-attn", action="store_true", help="Disable flash attention")
    p_run.add_argument("--no-mlock", action="store_true", help="Disable memory locking")
    p_run.set_defaults(func=cmd_run)

    # === serve ===
    p_serve = subparsers.add_parser("serve", help="Start the API server (OpenAI + Ollama compatible)")
    p_serve.add_argument("--model", "-m", help="Model to load on startup")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=22434, help="Port (default: 22434)")
    p_serve.add_argument("--gpu-layers", "-ngl", type=int, help="GPU layers (-1 = all, default)")
    p_serve.add_argument("--ctx-size", "-c", type=int, help="Context size (0 = model default)")
    p_serve.add_argument("--threads", type=int, help="CPU threads")
    p_serve.add_argument("--batch-size", type=int, help="Batch size")
    p_serve.add_argument("--no-flash-attn", action="store_true", help="Disable flash attention")
    p_serve.add_argument("--no-mlock", action="store_true", help="Disable memory locking")
    p_serve.set_defaults(func=cmd_serve)

    # === serve-embed ===
    p_serve_embed = subparsers.add_parser("serve-embed", help="Start embedding API server")
    p_serve_embed.add_argument("model", nargs="?", help="Model name (from registry or HuggingFace)")
    p_serve_embed.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    p_serve_embed.add_argument("--port", type=int, default=8081, help="Port (default: 8081)")
    p_serve_embed.add_argument("--gpu-layers", "-ngl", type=int, help="GPU layers (-1 = all, default)")
    p_serve_embed.add_argument("--threads", type=int, help="CPU threads")
    p_serve_embed.add_argument("--no-mlock", action="store_true", help="Disable memory locking")
    p_serve_embed.set_defaults(func=cmd_serve_embed)

    # === list ===
    p_list = subparsers.add_parser("list", aliases=["ls"], help="List downloaded models")
    p_list.add_argument("--available", "-a", action="store_true", help="Show available models from registry")
    p_list.add_argument("--embeddings", "-e", action="store_true", help="Show embedding models")
    p_list.set_defaults(func=cmd_list)

    # === rm ===
    p_rm = subparsers.add_parser("rm", aliases=["delete"], help="Delete a downloaded model")
    p_rm.add_argument("model", help="Model name to delete")
    p_rm.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    p_rm.set_defaults(func=cmd_rm)

    # === ps ===
    p_ps = subparsers.add_parser("ps", help="Show running model")
    p_ps.set_defaults(func=cmd_ps)

    # === stop ===
    p_stop = subparsers.add_parser("stop", help="Stop running model")
    p_stop.set_defaults(func=cmd_stop)

    # === setup ===
    p_setup = subparsers.add_parser("setup", help="Build and install meticulo engine")
    p_setup.add_argument("--force", "-f", action="store_true", help="Force rebuild")
    p_setup.add_argument("--backend", "-b",
                         choices=["cuda", "metal", "rocm", "vulkan", "sycl", "musa", "cpu"],
                         help="Override auto-detected GPU backend")
    p_setup.set_defaults(func=cmd_setup)

    # === info ===
    p_info = subparsers.add_parser("info", help="Show system info and configuration")
    p_info.set_defaults(func=cmd_info)

    # === tui ===
    p_tui = subparsers.add_parser("tui", help="Launch the TUI")
    p_tui.add_argument("--install", action="store_true", help="Install as system service (Linux systemd)")
    p_tui.add_argument("--uninstall", action="store_true", help="Remove system service")
    p_tui.set_defaults(func=cmd_tui)

    # === config ===
    p_config = subparsers.add_parser("config", help="Manage configuration")
    p_config_sub = p_config.add_subparsers(dest="config_action")
    
    p_config_show = p_config_sub.add_parser("show", help="Show current config")
    p_config_show.set_defaults(func=lambda a: print_config())
    
    p_config_set = p_config_sub.add_parser("set", help="Set config value (supports dot notation)")
    p_config_set.add_argument("key", help="Config key (e.g., auto_update, server.port)")
    p_config_set.add_argument("value", help="Value (true/false/string)")
    p_config_set.set_defaults(func=lambda a: set_config(a.key, a.value))
    
    p_config.set_defaults(func=lambda a: print_config())

    # === preset ===
    p_preset = subparsers.add_parser("preset", help="Manage inference presets")
    p_preset_sub = p_preset.add_subparsers(dest="preset_action")
    
    p_preset_list = p_preset_sub.add_parser("list", help="List saved presets")
    p_preset_list.set_defaults(func=cmd_preset)
    
    p_preset_get = p_preset_sub.add_parser("get", help="Get preset details")
    p_preset_get.add_argument("name", help="Preset name")
    p_preset_get.set_defaults(func=cmd_preset, name=None)
    
    p_preset_save = p_preset_sub.add_parser("save", help="Save a new preset")
    p_preset_save.add_argument("name", help="Preset name")
    p_preset_save.add_argument("--gpu-layers", "-ngl", type=int, help="GPU layers")
    p_preset_save.add_argument("--ctx-size", "-c", type=int, help="Context size")
    p_preset_save.add_argument("--batch-size", type=int, help="Batch size")
    p_preset_save.add_argument("--ubatch-size", type=int, help="Micro-batch size")
    p_preset_save.add_argument("--parallel", "-p", type=int, help="Parallel sequences")
    p_preset_save.add_argument("--no-flash-attn", action="store_true", help="Disable flash attention")
    p_preset_save.add_argument("--no-mlock", action="store_true", help="Disable memory locking")
    p_preset_save.add_argument("--no-mmap", action="store_true", help="Disable mmap")
    p_preset_save.set_defaults(func=cmd_preset, name=None)
    
    p_preset_delete = p_preset_sub.add_parser("delete", help="Delete a preset")
    p_preset_delete.add_argument("name", help="Preset name")
    p_preset_delete.set_defaults(func=cmd_preset, name=None)
    
    p_preset.set_defaults(func=cmd_preset, name=None)

    # === import ===
    p_import = subparsers.add_parser("import", help="Import models from Ollama")
    p_import_sub = p_import.add_subparsers(dest="import_action")
    
    p_import_scan = p_import_sub.add_parser("scan", help="Scan for Ollama models")
    p_import_scan.set_defaults(func=cmd_import)
    
    p_import_do = p_import_sub.add_parser("import", help="Import Ollama models to Meticulo")
    p_import_do.set_defaults(func=cmd_import)
    
    p_import_list = p_import_sub.add_parser("list", help="Show Ollama models directory")
    p_import_list.set_defaults(func=cmd_import)
    
    p_import.set_defaults(func=cmd_import)

    args = parser.parse_args()

    # Auto-update check (skip for setup/info commands)
    if args.command and args.command not in ("setup", "info", "tui", "config"):
        from .config import load_config, save_config
        config = load_config()
        auto_update = config.get("auto_update", True)
        
        if auto_update:
            meticulo_dir = Path(__file__).parent.parent
            git_dir = meticulo_dir / ".git"
            if git_dir.exists():
                import subprocess
                try:
                    result = subprocess.run(
                        ["git", "fetch", "--dry-run"],
                        cwd=meticulo_dir,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.stdout or result.stderr:
                        print("\n[Update] New version found, updating...")
                        
                        # Pull latest changes
                        pull_result = subprocess.run(
                            ["git", "pull", "--ff-only"],
                            cwd=meticulo_dir,
                            capture_output=True,
                            text=True
                        )
                        
                        # Reinstall package
                        import sys
                        reinstall_result = subprocess.run(
                            [sys.executable, "-m", "pip", "install", "-e", str(meticulo_dir)],
                            capture_output=True,
                            text=True
                        )
                        
                        if reinstall_result.returncode == 0:
                            print("[Update] Updated successfully!")
                            # Re-run the command with updated code
                            os.execv(sys.executable, [sys.executable] + sys.argv)
                        else:
                            print(f"[Update] Failed: {reinstall_result.stderr}")
                except Exception as e:
                    pass
                except Exception:
                    pass
    
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
