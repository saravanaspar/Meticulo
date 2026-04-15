"""
Meticulo engine process runner - hardware-optimized, zero overhead.

This is the core of Meticulo's performance advantage over Ollama.
Instead of adding a Go runtime layer, we:
1. Detect hardware precisely (GPU type, VRAM, CPU features, RAM)
2. Auto-tune all engine parameters for that hardware
3. Launch the raw C++ binary directly - zero middleware
4. Set environment variables for optimal GPU utilization

Why C++ (llama.cpp) and not Python inference?
- Python inference (torch, transformers) adds ~200MB+ runtime overhead
- Python GIL limits true multi-threaded CPU inference
- llama.cpp uses hand-tuned CUDA kernels, Metal shaders, AVX intrinsics
- llama.cpp's GGUF format enables quantized inference Python can't match
- Our Python is ONLY for management - inference stays in C++
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .config import DEFAULTS, get_server_path, get_cli_path, get_completion_path
from .hardware import get_hardware_profile, get_optimal_settings, HardwareProfile


class LlamaProcess:
    """Manages a meticulo engine server or CLI process with hardware-aware tuning."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.model_path: Optional[Path] = None
        self.port: int = DEFAULTS["engine_port"]
        self.host: str = DEFAULTS["host"]
        self._hw_profile: Optional[HardwareProfile] = None

    def _get_profile(self) -> HardwareProfile:
        """Get cached hardware profile."""
        if self._hw_profile is None:
            self._hw_profile = get_hardware_profile()
        return self._hw_profile

    def _build_env(self, profile: HardwareProfile) -> dict:
        """Build environment variables for optimal GPU utilization."""
        env = dict(os.environ)

        if profile.best_backend == "cuda":
            # Enable unified memory — model spills to system RAM instead of OOM crash
            env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"

        elif profile.best_backend == "rocm":
            env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
            # Override GFX version for unsupported AMD GPUs
            for gpu in profile.gpus:
                if gpu.vendor == "amd" and gpu.compute_capability:
                    gfx = gpu.compute_capability
                    if gfx.startswith("gfx103"):
                        env.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
                    elif gfx.startswith("gfx110"):
                        env.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

        return env

    def _build_server_args(
        self,
        model_path: Path,
        host: str = None,
        port: int = None,
        n_gpu_layers: int = None,
        ctx_size: int = None,
        n_batch: int = None,
        n_ubatch: int = None,
        n_parallel: int = None,
        flash_attn: bool = None,
        mlock: bool = None,
        no_mmap: bool = None,
        cont_batching: bool = None,
        threads: int = None,
        threads_batch: int = None,
        embedding: bool = None,
        extra_args: list = None,
    ) -> list[str]:
        """Build command-line arguments for meticulo-server with hardware-aware defaults."""
        server_bin = str(get_server_path())
        profile = self._get_profile()

        # Get hardware-optimized defaults
        model_size = model_path.stat().st_size if model_path.exists() else 0
        hw = get_optimal_settings(profile, model_size)

        self.host = host or DEFAULTS["host"]
        self.port = port or DEFAULTS["port"]

        args = [
            server_bin,
            "--model", str(model_path),
            "--host", self.host,
            "--port", str(self.port),
        ]

        # GPU layers — user override > hardware-detected > default
        ngl = n_gpu_layers if n_gpu_layers is not None else hw.get("n_gpu_layers", DEFAULTS["n_gpu_layers"])
        args.extend(["--gpu-layers", str(ngl)])

        # Context size — 0 = model's full native context (not Ollama's 4096!)
        ctx = ctx_size if ctx_size is not None else hw.get("ctx_size", 0)
        if ctx > 0:
            args.extend(["--ctx-size", str(ctx)])

        # Batch sizes — tuned per VRAM/RAM
        batch = n_batch if n_batch is not None else hw.get("n_batch", DEFAULTS["n_batch"])
        args.extend(["--batch-size", str(batch)])

        ubatch = n_ubatch if n_ubatch is not None else hw.get("n_ubatch", DEFAULTS["n_ubatch"])
        args.extend(["--ubatch-size", str(ubatch)])

        # Parallel sequences
        parallel = n_parallel if n_parallel is not None else DEFAULTS["n_parallel"]
        if parallel > 1:
            args.extend(["--parallel", str(parallel)])

        # Flash attention — on by default (Ollama doesn't do this)
        fa = flash_attn if flash_attn is not None else hw.get("flash_attn", True)
        if fa:
            args.extend(["--flash-attn", "on"])
        else:
            args.extend(["--flash-attn", "off"])

        # Memory locking — adaptive based on RAM
        ml = mlock if mlock is not None else hw.get("mlock", True)
        if ml:
            args.append("--mlock")

        # mmap
        nmmap = no_mmap if no_mmap is not None else DEFAULTS["no_mmap"]
        if nmmap:
            args.append("--no-mmap")

        # Continuous batching
        cb = cont_batching if cont_batching is not None else hw.get("cont_batching", True)
        if cb:
            args.append("--cont-batching")

        # Thread count
        t = threads if threads is not None else hw.get("threads", profile.cpu.cores_physical)
        tb = threads_batch if threads_batch is not None else hw.get("threads_batch", t)
        args.extend(["--threads", str(t)])
        args.extend(["--threads-batch", str(tb)])

        # Embedding mode
        if embedding:
            args.append("--embedding")

        # Extra passthrough args
        if extra_args:
            args.extend(extra_args)

        return args

    def start_server(self, model_path: Path, **kwargs) -> bool:
        """Start meticulo-server with hardware-optimized settings."""
        if self.process and self.process.poll() is None:
            print("Server already running. Stop it first with .stop()")
            return False

        self.model_path = model_path
        profile = self._get_profile()

        args = self._build_server_args(model_path, **kwargs)
        env = self._build_env(profile)

        # Display startup info
        model_size_mb = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0
        print(f"\nStarting meticulo-server...")
        print(f"  Model:    {model_path.name} ({model_size_mb:.0f} MB)")
        print(f"  Endpoint: http://{self.host}:{self.port}")
        print(f"  Backend:  {profile.best_backend.upper()}")

        if profile.can_gpu and profile.gpus:
            gpu = profile.gpus[0]
            vram_str = f"{gpu.vram_mb / 1024:.1f} GB" if gpu.vram_mb else "unknown"
            print(f"  GPU:      {gpu.name} ({vram_str} VRAM)")

        # Show key tuned settings
        ngl_idx = args.index("--n-gpu-layers") + 1 if "--n-gpu-layers" in args else -1
        ngl_val = args[ngl_idx] if ngl_idx > 0 else "?"
        threads_idx = args.index("--threads") + 1 if "--threads" in args else -1
        threads_val = args[threads_idx] if threads_idx > 0 else "?"
        batch_idx = args.index("--batch-size") + 1 if "--batch-size" in args else -1
        batch_val = args[batch_idx] if batch_idx > 0 else "?"

        print(f"  GPU layers: {ngl_val} (-1=all) | Threads: {threads_val} | Batch: {batch_val}")
        print(f"  Context: {'model default (no limit)' if '--ctx-size' not in args else args[args.index('--ctx-size') + 1]}")
        print(f"  Flash attn: {'on' if '--flash-attn' in args else 'off'} | mlock: {'on' if '--mlock' in args else 'off'}")

        if profile.best_backend == "cuda":
            print(f"  CUDA unified memory: on (OOM protection)")

        print(f"\n  $ {' '.join(args)}\n")

        # Start the process
        self.process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        return self._wait_for_ready()

    def _wait_for_ready(self, timeout: int = 300) -> bool:
        """Wait for the server to become ready. Longer timeout for large models."""
        import urllib.request
        import urllib.error

        health_url = f"http://{self.host}:{self.port}/health"
        start = time.time()

        print("Waiting for server to load model", end="", flush=True)

        while time.time() - start < timeout:
            if self.process and self.process.poll() is not None:
                output = self.process.stdout.read() if self.process.stdout else ""
                print(f"\n\nServer failed to start!")
                if output:
                    lines = output.strip().split("\n")
                    for line in lines[-20:]:
                        print(f"  {line}")
                self._suggest_fix(output)
                return False

            try:
                req = urllib.request.Request(health_url)
                with urllib.request.urlopen(req, timeout=2) as resp:
                    data = json.loads(resp.read().decode())
                    status = data.get("status", "")
                    if status == "ok":
                        elapsed = time.time() - start
                        print(f"\nServer ready! (loaded in {elapsed:.1f}s)")
                        return True
                    elif status == "loading model":
                        print(".", end="", flush=True)
            except (urllib.error.URLError, ConnectionRefusedError, OSError):
                print(".", end="", flush=True)

            time.sleep(1)

        print(f"\nTimeout waiting for server after {timeout}s")
        return False

    def _suggest_fix(self, output: str):
        """Suggest fixes based on common error patterns."""
        out = output.lower()
        if "out of memory" in out or "oom" in out or "cuda error" in out:
            print("\n  Suggestion: Model too large for GPU VRAM.")
            print("  Try: meticulo run <model> --gpu-layers 20  (partial offload)")
            print("  Or:  meticulo run <model> --gpu-layers 0   (CPU only)")
        elif "mlock" in out and ("failed" in out or "error" in out):
            print("\n  Suggestion: mlock failed (insufficient permissions).")
            print("  Try: meticulo run <model> --no-mlock")
        elif "flash" in out and "not supported" in out:
            print("\n  Suggestion: Flash attention not supported on this GPU.")
            print("  Try: meticulo run <model> --no-flash-attn")

    def stop(self):
        """Stop the running server."""
        if self.process and self.process.poll() is None:
            print("Stopping server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            print("Server stopped.")
        self.process = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def run_interactive(self, model_path: Path, **kwargs):
        """
        Simple interactive REPL - clean input/output, no junk.
        """
        profile = self._get_profile()
        model_size = model_path.stat().st_size if model_path.exists() else 0
        hw = get_optimal_settings(profile, model_size)

        ngl = kwargs.get("n_gpu_layers", hw.get("n_gpu_layers", -1))
        ctx = kwargs.get("ctx_size", hw.get("ctx_size", 0))
        threads = kwargs.get("threads", hw.get("threads", profile.cpu.cores_physical))
        flash_attn = kwargs.get("flash_attn", hw.get("flash_attn", True))
        mlock = kwargs.get("mlock", hw.get("mlock", True))
        n_batch = kwargs.get("n_batch", hw.get("n_batch", 2048))
        temperature = kwargs.get("temperature", 0.7)
        top_k = kwargs.get("top_k")
        top_p = kwargs.get("top_p")

        args = [
            str(get_completion_path()),
            "--model", str(model_path),
            "--gpu-layers", str(ngl),
            "--threads", str(threads),
            "--threads-batch", str(threads),
            "--batch-size", str(n_batch),
            "--temp", str(temperature),
            "--no-display-prompt",
            "--no-perf",
            "--log-disable",
            "--interactive-first",
        ]

        if top_k is not None:
            args.extend(["--top-k", str(top_k)])
        if top_p is not None:
            args.extend(["--top-p", str(top_p)])

        if ctx > 0:
            args.extend(["--ctx-size", str(ctx)])

        if flash_attn:
            args.extend(["--flash-attn", "on"])
        else:
            args.extend(["--flash-attn", "off"])
        if mlock:
            args.append("--mlock")

        env = self._build_env(profile)

        print(f"\n> Loading {model_path.name}...", flush=True)

        proc = subprocess.Popen(args, env=env, stdin=sys.stdin, stdout=sys.stdout, stderr=subprocess.DEVNULL)
        proc.wait()
        return proc.returncode

    def run_completion(
        self,
        model_path: Path,
        prompt: str,
        max_tokens: int = -1,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Run a single completion. max_tokens=-1 means generate until EOS."""
        cli_bin = str(get_cli_path())
        profile = self._get_profile()
        model_size = model_path.stat().st_size if model_path.exists() else 0
        hw = get_optimal_settings(profile, model_size)

        ngl = kwargs.get("n_gpu_layers", hw.get("n_gpu_layers", -1))
        threads = kwargs.get("threads", hw.get("threads", profile.cpu.cores_physical))
        top_k = kwargs.get("top_k")
        top_p = kwargs.get("top_p")

        args = [
            cli_bin,
            "--model", str(model_path),
            "--n-gpu-layers", str(ngl),
            "--threads", str(threads),
            "--threads-batch", str(threads),
            "--prompt", prompt,
            "--temp", str(temperature),
            "--no-display-prompt",
        ]

        if top_k is not None:
            args.extend(["--top-k", str(top_k)])
        if top_p is not None:
            args.extend(["--top-p", str(top_p)])

        if max_tokens > 0:
            args.extend(["--n-predict", str(max_tokens)])
        else:
            args.extend(["--n-predict", "-1"])

        if kwargs.get("flash_attn", hw.get("flash_attn", True)):
            args.extend(["--flash-attn", "on"])
        else:
            args.extend(["--flash-attn", "off"])
        if kwargs.get("mlock", hw.get("mlock", True)):
            args.append("--mlock")

        ctx = kwargs.get("ctx_size", hw.get("ctx_size", 0))
        if ctx > 0:
            args.extend(["--ctx-size", str(ctx)])

        env = self._build_env(profile)
        result = subprocess.run(args, capture_output=True, text=True, env=env)
        return result.stdout


# Singleton process manager
_instance: Optional[LlamaProcess] = None


def get_runner() -> LlamaProcess:
    """Get the global LlamaProcess instance."""
    global _instance
    if _instance is None:
        _instance = LlamaProcess()
    return _instance
