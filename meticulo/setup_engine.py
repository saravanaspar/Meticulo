"""
Automated meticulo engine build and installation.

Supports ALL GPU backends that meticulo engine supports:
- CUDA (NVIDIA)
- Metal (Apple Silicon)
- HIP/ROCm (AMD)
- Vulkan (universal GPU fallback)
- SYCL (Intel Arc / Data Center)
- MUSA (Moore Threads)
- CPU optimized (AVX2/AVX-512/NEON/SVE)

If pre-built binaries are available, downloads them instead of compiling.
"""

import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from .config import get_engine_dir, get_server_path, get_cli_path
from .hardware import get_hardware_profile, print_hardware_report, _has_cuda_toolkit


def _run(cmd: list[str], cwd: Path = None, env: dict = None):
    """Run a command and stream output."""
    merged_env = {**os.environ, **(env or {})}
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def _get_system_tag() -> str:
    """Get system tag for pre-built binary downloads."""
    system = platform.system().lower()
    arch = platform.machine()
    
    if system == "darwin":
        if arch == "arm64":
            return "darwin-arm64"
        return "darwin-x86_64"
    elif system == "linux":
        if arch == "x86_64":
            return "linux-x86_64"
        elif arch == "aarch64":
            return "linux-arm64"
    return None


def _download_prebuilt(backend: str, system_tag: str) -> bool:
    """Try to download pre-built binary. Returns True if successful."""
    if not system_tag:
        return False
    
    # Check GitHub releases for pre-built (future: upload your own releases)
    # For now, we'll compile but make it faster
    return False


def _get_cmake_gpu_flags(backend: str, profile=None) -> list[str]:
    """Get cmake flags for the detected GPU backend."""
    flags = []

    if backend == "cuda":
        # First check if nvcc is available
        import shutil
        nvcc_path = shutil.which("nvcc")
        
        if nvcc_path:
            flags.append("-DGGML_CUDA=ON")
            flags.append(f"-DCMAKE_CUDA_COMPILER={nvcc_path}")
            # Auto-detect CUDA architectures for faster compilation
            if profile and profile.gpus:
                archs = set()
                for gpu in profile.gpus:
                    if gpu.vendor == "nvidia" and gpu.compute_capability:
                        cc = gpu.compute_capability.replace(".", "")
                        archs.add(cc)
                if archs:
                    flags.append(f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(sorted(archs))}")
        else:
            # nvcc not found - check common CUDA installation paths
            cuda_paths = [
                "/usr/local/cuda",
                "/usr/local/cuda-12",
                "/usr/local/cuda-11",
                "/opt/cuda",
            ]
            cuda_root = None
            for path in cuda_paths:
                if os.path.exists(path):
                    cuda_root = path
                    break
            
            if cuda_root:
                nvcc = os.path.join(cuda_root, "bin", "nvcc")
                flags.append(f"-DCUDAToolkit_ROOT={cuda_root}")
                flags.append("-DGGML_CUDA=ON")
                if os.path.exists(nvcc):
                    flags.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")
                if profile and profile.gpus:
                    archs = set()
                    for gpu in profile.gpus:
                        if gpu.vendor == "nvidia" and gpu.compute_capability:
                            cc = gpu.compute_capability.replace(".", "")
                            archs.add(cc)
                    if archs:
                        flags.append(f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(sorted(archs))}")
            else:
                # No CUDA toolkit - fall back to CPU
                print("  ⚠ CUDA toolkit not found, falling back to CPU build")
                flags.append("-DGGML_CUDA=OFF")

    elif backend == "metal":
        flags.append("-DGGML_METAL=ON")

    elif backend == "rocm":
        flags.append("-DGGML_HIP=ON")
        # Set AMD GPU architecture if detected
        if profile and profile.gpus:
            for gpu in profile.gpus:
                if gpu.vendor == "amd" and gpu.compute_capability:
                    flags.append(f"-DAMDGPU_TARGETS={gpu.compute_capability}")
                    break

    elif backend == "vulkan":
        flags.append("-DGGML_VULKAN=ON")

    elif backend == "sycl":
        flags.append("-DGGML_SYCL=ON")
        flags.append("-DCMAKE_C_COMPILER=icx")
        flags.append("-DCMAKE_CXX_COMPILER=icpx")

    elif backend == "musa":
        flags.append("-DGGML_MUSA=ON")

    return flags


def _get_cmake_cpu_flags(profile=None) -> list[str]:
    """Get cmake flags for CPU optimizations."""
    flags = []
    if profile and profile.cpu:
        cpu = profile.cpu
        # Enable native optimizations if building for this machine only
        flags.append("-DGGML_NATIVE=ON")
        # ARM-specific optimizations
        if cpu.has_sve:
            flags.append("-DGGML_CPU_ARM_ARCH=armv8.6-a+sve+sve2")
    return flags


def setup_engine(
    force: bool = False,
    backend: str = None,
    **kwargs,
):
    """
    Clone and build the meticulo engine from source.
    Auto-detects GPU and builds with the best available acceleration.

    Args:
        force: Force rebuild even if binaries exist
        backend: Override auto-detected backend (cuda/metal/rocm/vulkan/sycl/musa/cpu)
    """
    engine_dir = get_engine_dir()
    build_dir = engine_dir / "build"

    # Check if already built
    server_path = get_server_path()
    cli_path = get_cli_path()
    if server_path.exists() and cli_path.exists() and not force:
        print(f"Engine already built at {engine_dir}")
        print(f"  meticulo-server: {server_path}")
        print(f"  meticulo-cli: {cli_path}")
        print("Use --force to rebuild.")
        return

    # Check dependencies
    print("Checking build dependencies...")
    for tool in ["git", "cmake"]:
        if not shutil.which(tool):
            print(f"  ✗ {tool} not found. Please install it first.")
            sys.exit(1)
        print(f"  ✓ {tool}")

    # Need either make or ninja
    build_tool = None
    if shutil.which("ninja"):
        build_tool = "ninja"
        print(f"  ✓ ninja (preferred)")
    elif shutil.which("make"):
        build_tool = "make"
        print(f"  ✓ make")
    else:
        print(f"  ✗ No build tool found. Install make or ninja.")
        sys.exit(1)

    # Check for C++ compiler
    cxx = shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
    if not cxx:
        print("  ✗ C++ compiler not found. Install g++ or clang++.")
        sys.exit(1)
    print(f"  ✓ C++ compiler: {cxx}")

    # ccache for faster rebuilds
    has_ccache = shutil.which("ccache") is not None
    if has_ccache:
        print(f"  ✓ ccache (faster rebuilds)")

    # Detect hardware
    print("\nDetecting hardware...")
    profile = get_hardware_profile(force_refresh=True)
    print_hardware_report(profile)

    # Determine backend
    if backend:
        gpu_backend = backend
    else:
        gpu_backend = profile.best_backend

    # Clone or update
    if engine_dir.exists():
        if force:
            print("Updating engine...")
            _run(["git", "pull", "--rebase"], cwd=engine_dir)
        else:
            print(f"Using existing engine at {engine_dir}")
    else:
        print("Cloning engine...")
        engine_dir.parent.mkdir(parents=True, exist_ok=True)
        _run([
            "git", "clone",
            "--depth", "1",
            "https://github.com/ggerganov/llama.cpp.git",
            str(engine_dir),
        ])

    # Build
    print(f"\nBuilding engine with {gpu_backend.upper()} backend...")

    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    cmake_args = [
        "cmake", "..",
        "-DCMAKE_BUILD_TYPE=Release",
    ]

    # Use ninja if available
    if build_tool == "ninja":
        cmake_args.append("-GNinja")

    # ccache
    if has_ccache:
        cmake_args.append("-DCMAKE_C_COMPILER_LAUNCHER=ccache")
        cmake_args.append("-DCMAKE_CXX_COMPILER_LAUNCHER=ccache")

    # GPU backend flags
    gpu_flags = _get_cmake_gpu_flags(gpu_backend, profile)
    cmake_args.extend(gpu_flags)

    # CPU optimization flags
    cpu_flags = _get_cmake_cpu_flags(profile)
    cmake_args.extend(cpu_flags)

    print(f"  cmake flags: {' '.join(cmake_args[1:])}")

    # SYCL requires sourcing oneAPI environment
    env = None
    if gpu_backend == "sycl":
        setvars = "/opt/intel/oneapi/setvars.sh"
        if os.path.exists(setvars):
            print("  Sourcing Intel oneAPI environment...")
            # Source the environment and capture variables
            result = subprocess.run(
                ["bash", "-c", f"source {setvars} --force 2>/dev/null && env"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                env = {}
                for line in result.stdout.split("\n"):
                    if "=" in line:
                        key, _, val = line.partition("=")
                        env[key] = val

    _run(cmake_args, cwd=build_dir, env=env)

    # Build with all cores
    cpu_count = os.cpu_count() or 4
    build_cmd = ["cmake", "--build", ".", "--config", "Release", "-j", str(cpu_count)]
    _run(build_cmd, cwd=build_dir, env=env)

    # Verify
    server_found = False
    cli_found = False

    for server_candidate in [build_dir / "bin" / "meticulo-server", build_dir / "bin" / "server"]:
        if server_candidate.exists():
            server_found = True
            print(f"\n✓ Built meticulo-server: {server_candidate}")
            break

    for cli_candidate in [build_dir / "bin" / "meticulo-cli", build_dir / "bin" / "main"]:
        if cli_candidate.exists():
            cli_found = True
            print(f"✓ Built meticulo-cli: {cli_candidate}")
            break

    if server_found and cli_found:
        print(f"\nEngine built successfully with {gpu_backend.upper()} acceleration!")
        print(f"\nRun 'meticulo info' to verify the installation.")
    else:
        print("\nWarning: Some binaries may not have built correctly.")
        print("Check the build output above for errors.")
        bin_dir = build_dir / "bin"
        if bin_dir.exists():
            print(f"\nFiles in {bin_dir}:")
            for f in sorted(bin_dir.iterdir()):
                print(f"  {f.name}")
