"""
Comprehensive hardware detection for Meticulo.

Detects ALL GPU types (NVIDIA/CUDA, AMD/ROCm, Intel/SYCL, Apple/Metal, Vulkan),
CPU capabilities, and RAM to auto-tune llama.cpp for maximum performance.

Why this matters vs Ollama:
- Ollama defaults to 4096 token context regardless of hardware
- Ollama doesn't fully exploit GPU VRAM for context sizing
- Ollama's Go runtime adds overhead before inference even starts
- We detect everything and auto-configure for raw hardware speed
"""

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    name: str = ""
    vendor: str = ""          # nvidia, amd, intel, apple, moore_threads
    backend: str = ""         # cuda, rocm, metal, vulkan, sycl, musa, opencl
    vram_mb: int = 0
    vram_free_mb: int = 0
    compute_capability: str = ""  # NVIDIA: "8.9", AMD: "gfx1100"
    driver_version: str = ""
    cuda_version: str = ""
    index: int = 0


@dataclass
class CPUInfo:
    """Information about the CPU."""
    name: str = ""
    cores_physical: int = 0
    cores_logical: int = 0
    arch: str = ""               # x86_64, aarch64, arm64
    features: list = field(default_factory=list)  # avx, avx2, avx512, neon, sve
    has_avx: bool = False
    has_avx2: bool = False
    has_avx512: bool = False
    has_f16c: bool = False
    has_neon: bool = False       # ARM
    has_sve: bool = False        # ARM SVE
    has_amx: bool = False        # Intel AMX


@dataclass
class MemoryInfo:
    """System memory information."""
    total_mb: int = 0
    available_mb: int = 0
    swap_total_mb: int = 0
    swap_free_mb: int = 0


@dataclass
class HardwareProfile:
    """Complete hardware profile."""
    gpus: list = field(default_factory=list)
    cpu: CPUInfo = field(default_factory=CPUInfo)
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    os_name: str = ""
    os_version: str = ""
    best_backend: str = ""       # The recommended backend to use
    can_gpu: bool = False
    total_vram_mb: int = 0


def _run_cmd(cmd: list[str], timeout: int = 10) -> Optional[str]:
    """Run a command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def detect_nvidia_gpus() -> list[GPUInfo]:
    """Detect NVIDIA GPUs using nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return []

    gpus = []

    # Query GPU details
    output = _run_cmd([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version",
        "--format=csv,noheader,nounits"
    ])
    if not output:
        return []

    # Get CUDA version
    cuda_ver = ""
    smi_output = _run_cmd(["nvidia-smi"])
    if smi_output:
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", smi_output)
        if match:
            cuda_ver = match.group(1)

    for line in output.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            gpu = GPUInfo(
                index=int(parts[0]),
                name=parts[1],
                vendor="nvidia",
                backend="cuda",
                vram_mb=int(float(parts[2])),
                vram_free_mb=int(float(parts[3])),
                compute_capability=parts[4],
                driver_version=parts[5],
                cuda_version=cuda_ver,
            )
            gpus.append(gpu)

    return gpus


def detect_amd_gpus() -> list[GPUInfo]:
    """Detect AMD GPUs using rocm-smi or rocminfo."""
    gpus = []

    # Try rocm-smi first
    if shutil.which("rocm-smi"):
        output = _run_cmd(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
        if output:
            try:
                data = json.loads(output)
                for idx, (card_id, card_data) in enumerate(data.items()):
                    if isinstance(card_data, dict):
                        name = card_data.get("Card Series", card_data.get("Card series", f"AMD GPU {idx}"))
                        vram_total = int(card_data.get("VRAM Total Memory (B)", 0)) // (1024 * 1024)
                        vram_used = int(card_data.get("VRAM Total Used Memory (B)", 0)) // (1024 * 1024)
                        gpu = GPUInfo(
                            index=idx,
                            name=str(name),
                            vendor="amd",
                            backend="rocm",
                            vram_mb=vram_total,
                            vram_free_mb=vram_total - vram_used,
                        )
                        gpus.append(gpu)
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

    # Try rocminfo for compute capability
    if shutil.which("rocminfo"):
        output = _run_cmd(["rocminfo"])
        if output:
            gfx_match = re.findall(r"(gfx\w+)", output)
            for i, gpu in enumerate(gpus):
                if i < len(gfx_match):
                    gpu.compute_capability = gfx_match[i]

    # Fallback: check /sys for AMD GPUs
    if not gpus:
        amd_gpu_dir = Path("/sys/class/drm")
        if amd_gpu_dir.exists():
            for card in sorted(amd_gpu_dir.iterdir()):
                vendor_path = card / "device" / "vendor"
                if vendor_path.exists():
                    try:
                        vendor_id = vendor_path.read_text().strip()
                        if vendor_id == "0x1002":  # AMD vendor ID
                            name_path = card / "device" / "product_name"
                            name = name_path.read_text().strip() if name_path.exists() else f"AMD GPU"
                            # Try to get VRAM from /sys
                            vram = 0
                            mem_path = card / "device" / "mem_info_vram_total"
                            if mem_path.exists():
                                vram = int(mem_path.read_text().strip()) // (1024 * 1024)
                            gpu = GPUInfo(
                                index=len(gpus),
                                name=name,
                                vendor="amd",
                                backend="rocm" if shutil.which("rocm-smi") or os.path.exists("/opt/rocm") else "vulkan",
                                vram_mb=vram,
                            )
                            gpus.append(gpu)
                    except (OSError, ValueError):
                        pass

    return gpus


def detect_intel_gpus() -> list[GPUInfo]:
    """Detect Intel GPUs (Arc, integrated, Data Center)."""
    gpus = []

    # Check for Intel GPU via sycl-ls (Intel oneAPI)
    if shutil.which("sycl-ls"):
        output = _run_cmd(["sycl-ls"])
        if output:
            for i, line in enumerate(output.split("\n")):
                if "intel" in line.lower() and "gpu" in line.lower():
                    name_match = re.search(r"\[(.+?)\]", line)
                    name = name_match.group(1) if name_match else f"Intel GPU {i}"
                    gpu = GPUInfo(
                        index=len(gpus),
                        name=name,
                        vendor="intel",
                        backend="sycl",
                    )
                    gpus.append(gpu)

    # Check for Intel GPU via /sys
    if not gpus:
        drm_dir = Path("/sys/class/drm")
        if drm_dir.exists():
            for card in sorted(drm_dir.iterdir()):
                vendor_path = card / "device" / "vendor"
                if vendor_path.exists():
                    try:
                        vendor_id = vendor_path.read_text().strip()
                        if vendor_id == "0x8086":  # Intel vendor ID
                            device_path = card / "device" / "device"
                            device_id = device_path.read_text().strip() if device_path.exists() else ""
                            # Determine if it's a discrete GPU (Arc) or integrated
                            name = "Intel GPU"
                            # Check for Intel Arc series
                            if shutil.which("intel_gpu_top"):
                                name = "Intel Arc GPU"
                            gpu = GPUInfo(
                                index=len(gpus),
                                name=name,
                                vendor="intel",
                                backend="sycl" if shutil.which("sycl-ls") else "vulkan",
                            )
                            gpus.append(gpu)
                            break  # Usually only one Intel GPU
                    except (OSError, ValueError):
                        pass

    return gpus


def detect_apple_gpu() -> list[GPUInfo]:
    """Detect Apple Silicon GPU (Metal)."""
    if platform.system() != "Darwin":
        return []

    gpus = []

    # Get chip info via system_profiler
    output = _run_cmd(["system_profiler", "SPDisplaysDataType", "-json"])
    if output:
        try:
            data = json.loads(output)
            displays = data.get("SPDisplaysDataType", [])
            for i, display in enumerate(displays):
                name = display.get("sppci_model", "Apple GPU")
                # Apple Silicon VRAM is unified - shared with system RAM
                vram_str = display.get("spdisplays_vram_shared", display.get("spdisplays_vram", "0"))
                vram = 0
                if isinstance(vram_str, str):
                    vram_match = re.search(r"(\d+)", vram_str)
                    if vram_match:
                        vram = int(vram_match.group(1)) * 1024  # Usually in GB
                gpu = GPUInfo(
                    index=i,
                    name=name,
                    vendor="apple",
                    backend="metal",
                    vram_mb=vram,
                    vram_free_mb=vram,  # Approximate
                )
                gpus.append(gpu)
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback for Apple Silicon
    if not gpus:
        output = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        chip_name = output if output else "Apple Silicon"
        # Get total memory (unified architecture)
        mem_output = _run_cmd(["sysctl", "-n", "hw.memsize"])
        vram = int(mem_output) // (1024 * 1024) if mem_output else 0
        gpus.append(GPUInfo(
            index=0,
            name=f"{chip_name} (Metal)",
            vendor="apple",
            backend="metal",
            vram_mb=vram,
            vram_free_mb=vram,
        ))

    return gpus


def detect_vulkan_devices() -> list[GPUInfo]:
    """Detect any GPU that supports Vulkan (fallback for non-CUDA/ROCm/Metal)."""
    if not shutil.which("vulkaninfo"):
        return []

    output = _run_cmd(["vulkaninfo", "--summary"], timeout=15)
    if not output:
        return []

    gpus = []
    current_gpu = None

    for line in output.split("\n"):
        if "GPU" in line and "=" in line:
            if current_gpu:
                gpus.append(current_gpu)
            current_gpu = GPUInfo(index=len(gpus), backend="vulkan")
        if current_gpu:
            if "deviceName" in line:
                name = line.split("=")[-1].strip()
                current_gpu.name = name
                # Determine vendor from name
                name_lower = name.lower()
                if "nvidia" in name_lower or "geforce" in name_lower or "rtx" in name_lower or "gtx" in name_lower:
                    current_gpu.vendor = "nvidia"
                elif "amd" in name_lower or "radeon" in name_lower:
                    current_gpu.vendor = "amd"
                elif "intel" in name_lower:
                    current_gpu.vendor = "intel"
                else:
                    current_gpu.vendor = "other"
            elif "apiVersion" in line:
                current_gpu.driver_version = line.split("=")[-1].strip()

    if current_gpu:
        gpus.append(current_gpu)

    return gpus


def detect_moore_threads_gpus() -> list[GPUInfo]:
    """Detect Moore Threads GPUs (MUSA)."""
    if not shutil.which("mthreads-gmi"):
        return []

    gpus = []
    output = _run_cmd(["mthreads-gmi"])
    if output and "MTT" in output:
        gpu = GPUInfo(
            index=0,
            name="Moore Threads GPU",
            vendor="moore_threads",
            backend="musa",
        )
        gpus.append(gpu)

    return gpus


def detect_cpu() -> CPUInfo:
    """Detect CPU capabilities."""
    info = CPUInfo()
    info.arch = platform.machine()
    info.cores_logical = os.cpu_count() or 1

    system = platform.system()

    if system == "Linux":
        # Read /proc/cpuinfo
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()

            # CPU name
            name_match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
            if name_match:
                info.name = name_match.group(1).strip()

            # Physical cores
            physical = set()
            for match in re.finditer(r"core id\s*:\s*(\d+)", cpuinfo):
                physical.add(match.group(1))
            info.cores_physical = len(physical) if physical else info.cores_logical

            # CPU flags/features
            flags_match = re.search(r"flags\s*:\s*(.+)", cpuinfo)
            if flags_match:
                flags = flags_match.group(1).strip().split()
                info.features = flags
                info.has_avx = "avx" in flags
                info.has_avx2 = "avx2" in flags
                info.has_avx512 = any(f.startswith("avx512") for f in flags)
                info.has_f16c = "f16c" in flags
                info.has_amx = any(f.startswith("amx") for f in flags)

            # ARM features
            if "neon" in cpuinfo.lower() or "asimd" in cpuinfo.lower():
                info.has_neon = True
            if "sve" in cpuinfo.lower():
                info.has_sve = True

        except OSError:
            pass

    elif system == "Darwin":
        output = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        if output:
            info.name = output
        cores = _run_cmd(["sysctl", "-n", "hw.physicalcpu"])
        if cores:
            info.cores_physical = int(cores)
        # Apple Silicon always has NEON
        if platform.machine() == "arm64":
            info.has_neon = True

    # Fallback for physical cores
    if info.cores_physical == 0:
        info.cores_physical = info.cores_logical

    return info


def detect_memory() -> MemoryInfo:
    """Detect system memory."""
    mem = MemoryInfo()

    if platform.system() == "Linux":
        try:
            meminfo = Path("/proc/meminfo").read_text()

            total_match = re.search(r"MemTotal:\s*(\d+)\s*kB", meminfo)
            if total_match:
                mem.total_mb = int(total_match.group(1)) // 1024

            avail_match = re.search(r"MemAvailable:\s*(\d+)\s*kB", meminfo)
            if avail_match:
                mem.available_mb = int(avail_match.group(1)) // 1024

            swap_total = re.search(r"SwapTotal:\s*(\d+)\s*kB", meminfo)
            if swap_total:
                mem.swap_total_mb = int(swap_total.group(1)) // 1024

            swap_free = re.search(r"SwapFree:\s*(\d+)\s*kB", meminfo)
            if swap_free:
                mem.swap_free_mb = int(swap_free.group(1)) // 1024

        except OSError:
            pass

    elif platform.system() == "Darwin":
        output = _run_cmd(["sysctl", "-n", "hw.memsize"])
        if output:
            mem.total_mb = int(output) // (1024 * 1024)
        # macOS available memory is harder - use vm_stat
        vm_output = _run_cmd(["vm_stat"])
        if vm_output:
            free_match = re.search(r"Pages free:\s*(\d+)", vm_output)
            inactive_match = re.search(r"Pages inactive:\s*(\d+)", vm_output)
            page_size = 16384 if platform.machine() == "arm64" else 4096
            free_pages = int(free_match.group(1)) if free_match else 0
            inactive_pages = int(inactive_match.group(1)) if inactive_match else 0
            mem.available_mb = (free_pages + inactive_pages) * page_size // (1024 * 1024)

    # Fallback using os.sysconf
    if mem.total_mb == 0:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            total_pages = os.sysconf("SC_PHYS_PAGES")
            mem.total_mb = (page_size * total_pages) // (1024 * 1024)
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            mem.available_mb = (page_size * avail_pages) // (1024 * 1024)
        except (ValueError, OSError):
            pass

    return mem


def _has_cuda_toolkit() -> bool:
    """Check if CUDA toolkit (nvcc) is installed."""
    import shutil
    if shutil.which("nvcc"):
        return True
    # Check common CUDA paths
    for path in ["/usr/local/cuda/bin/nvcc", "/usr/local/cuda-12/bin/nvcc", "/opt/cuda/bin/nvcc"]:
        if os.path.exists(path):
            return True
    return False


def detect_hardware() -> HardwareProfile:
    """
    Run full hardware detection. Returns a complete HardwareProfile.

    Detection order (priority):
    1. NVIDIA CUDA (best supported, most common)
    2. Apple Metal (macOS)
    3. AMD ROCm (Linux)
    4. Intel SYCL (oneAPI)
    5. Moore Threads MUSA
    6. Vulkan (universal fallback for any GPU)
    7. CPU-only
    """
    profile = HardwareProfile()
    profile.os_name = platform.system()
    profile.os_version = platform.release()

    # Detect CPU and memory
    profile.cpu = detect_cpu()
    profile.memory = detect_memory()

    # Detect GPUs in priority order
    all_gpus = []

    # 1. NVIDIA (CUDA)
    nvidia_gpus = detect_nvidia_gpus()
    all_gpus.extend(nvidia_gpus)

    # 2. Apple Metal
    apple_gpus = detect_apple_gpu()
    all_gpus.extend(apple_gpus)

    # 3. AMD (ROCm)
    amd_gpus = detect_amd_gpus()
    # Don't double-count if already found via another method
    existing_names = {g.name for g in all_gpus}
    for g in amd_gpus:
        if g.name not in existing_names:
            all_gpus.append(g)

    # 4. Intel (SYCL)
    intel_gpus = detect_intel_gpus()
    for g in intel_gpus:
        if g.name not in existing_names:
            all_gpus.append(g)

    # 5. Moore Threads (MUSA)
    mt_gpus = detect_moore_threads_gpus()
    all_gpus.extend(mt_gpus)

    # 6. Vulkan fallback (only if nothing else found)
    if not all_gpus:
        vulkan_gpus = detect_vulkan_devices()
        all_gpus.extend(vulkan_gpus)

    profile.gpus = all_gpus

    # Calculate totals
    profile.total_vram_mb = sum(g.vram_mb for g in all_gpus)
    profile.can_gpu = len(all_gpus) > 0

    # Determine best backend - but only if we have the required toolchain
    if nvidia_gpus and _has_cuda_toolkit():
        profile.best_backend = "cuda"
    elif apple_gpus:
        profile.best_backend = "metal"
    elif amd_gpus and any(g.backend == "rocm" for g in amd_gpus):
        profile.best_backend = "rocm"
    elif intel_gpus and any(g.backend == "sycl" for g in intel_gpus):
        profile.best_backend = "sycl"
    elif mt_gpus:
        profile.best_backend = "musa"
    elif all_gpus and any(g.backend == "vulkan" for g in all_gpus):
        profile.best_backend = "vulkan"
    else:
        profile.best_backend = "cpu"

    return profile


def estimate_model_memory_mb(model_size_bytes: int) -> int:
    """Estimate memory needed to run a model (model + KV cache + overhead)."""
    model_mb = model_size_bytes / (1024 * 1024)
    # KV cache and runtime overhead typically add 20-40%
    return int(model_mb * 1.35)


def get_optimal_settings(profile: HardwareProfile, model_size_bytes: int = 0) -> dict:
    """
    Calculate optimal llama.cpp settings based on detected hardware.

    This is where Meticulo diverges from Ollama:
    - Ollama defaults to 4096 context → we use full model context
    - Ollama doesn't auto-tune batch sizes → we optimize per hardware
    - Ollama doesn't set threads optimally → we use physical core count
    - Ollama doesn't enable flash_attn by default → we do
    """
    settings = {}
    ram_mb = profile.memory.total_mb
    available_ram_mb = profile.memory.available_mb
    model_mem_mb = estimate_model_memory_mb(model_size_bytes) if model_size_bytes else 0

    # ── GPU Settings ──
    if profile.can_gpu:
        primary_gpu = profile.gpus[0]
        vram_mb = primary_gpu.vram_mb

        if vram_mb > 0 and model_mem_mb > 0:
            # Model fits entirely in VRAM → full GPU offload
            if model_mem_mb < vram_mb * 0.85:
                settings["n_gpu_layers"] = -1  # All layers to GPU
            else:
                # Partial offload - estimate layers that fit
                # Rough heuristic: each layer is ~equal fraction of model
                layer_fraction = (vram_mb * 0.80) / model_mem_mb
                settings["n_gpu_layers"] = max(1, int(layer_fraction * 80))
        else:
            settings["n_gpu_layers"] = -1  # Default: try full offload

        # Enable unified memory on CUDA to prevent OOM crashes
        if primary_gpu.backend == "cuda":
            settings["env"] = {"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"}
    else:
        settings["n_gpu_layers"] = 0  # CPU only

    # ── Context Size ──
    # Unlike Ollama's 4096 default, we let the model use its native context
    # ctx_size=0 tells llama.cpp to use the model's trained context length
    settings["ctx_size"] = 0

    # ── Thread Count ──
    # Use physical cores, not logical (hyperthreading hurts llama.cpp perf)
    physical_cores = profile.cpu.cores_physical
    if profile.can_gpu:
        # When GPU is primary, fewer CPU threads needed (avoid contention)
        settings["threads"] = max(1, physical_cores)
        settings["threads_batch"] = physical_cores
    else:
        # CPU-only: use all physical cores
        settings["threads"] = physical_cores
        settings["threads_batch"] = physical_cores

    # ── Batch Size ──
    if profile.can_gpu:
        vram = profile.gpus[0].vram_mb if profile.gpus else 0
        if vram >= 24000:       # 24GB+ (RTX 4090, A100, etc.)
            settings["n_batch"] = 4096
            settings["n_ubatch"] = 1024
        elif vram >= 8000:      # 8-24GB (RTX 3070-4080, etc.)
            settings["n_batch"] = 2048
            settings["n_ubatch"] = 512
        elif vram >= 4000:      # 4-8GB (GTX 1070, RX 580, etc.)
            settings["n_batch"] = 1024
            settings["n_ubatch"] = 256
        else:                   # <4GB or unknown VRAM
            settings["n_batch"] = 512
            settings["n_ubatch"] = 128
    else:
        # CPU: batch size based on RAM
        if ram_mb >= 32000:
            settings["n_batch"] = 2048
            settings["n_ubatch"] = 512
        elif ram_mb >= 16000:
            settings["n_batch"] = 1024
            settings["n_ubatch"] = 256
        elif ram_mb >= 8000:
            settings["n_batch"] = 512
            settings["n_ubatch"] = 128
        else:
            # Low RAM: conservative to prevent OOM
            settings["n_batch"] = 256
            settings["n_ubatch"] = 64

    # ── Flash Attention ──
    # Enable by default - massive speed boost, supported on most hardware
    settings["flash_attn"] = True

    # ── Memory Locking ──
    if profile.can_gpu or ram_mb >= 16000:
        settings["mlock"] = True
    else:
        # Low RAM system: don't lock, let OS manage paging
        settings["mlock"] = False

    # ── mmap ──
    # Use mmap on CPU-heavy workloads (faster loading, less RAM pressure)
    settings["no_mmap"] = False

    # ── CPU-Only Optimizations ──
    if not profile.can_gpu:
        # On CPU-only systems, these are critical for stability:
        if ram_mb < 8000:
            # Very low RAM: limit context to prevent crashes
            settings["ctx_size"] = 4096
            settings["mlock"] = False
            settings["n_batch"] = 128
            settings["n_ubatch"] = 32

        if ram_mb < 4000:
            # Extremely low RAM
            settings["ctx_size"] = 2048
            settings["n_batch"] = 64
            settings["n_ubatch"] = 16

    # ── Continuous Batching ──
    settings["cont_batching"] = True

    return settings


def print_hardware_report(profile: HardwareProfile):
    """Print a formatted hardware detection report."""
    print(f"\n{'═' * 60}")
    print(f"  Meticulo Hardware Detection Report")
    print(f"{'═' * 60}")

    # OS
    print(f"\n  OS:      {profile.os_name} {profile.os_version}")

    # CPU
    cpu = profile.cpu
    print(f"\n  CPU:     {cpu.name or 'Unknown'}")
    print(f"  Cores:   {cpu.cores_physical} physical / {cpu.cores_logical} logical")
    print(f"  Arch:    {cpu.arch}")

    features = []
    if cpu.has_avx:
        features.append("AVX")
    if cpu.has_avx2:
        features.append("AVX2")
    if cpu.has_avx512:
        features.append("AVX-512")
    if cpu.has_f16c:
        features.append("F16C")
    if cpu.has_amx:
        features.append("AMX")
    if cpu.has_neon:
        features.append("NEON")
    if cpu.has_sve:
        features.append("SVE")
    if features:
        print(f"  ISA:     {', '.join(features)}")

    # Memory
    mem = profile.memory
    print(f"\n  RAM:     {mem.total_mb / 1024:.1f} GB total / {mem.available_mb / 1024:.1f} GB available")
    if mem.swap_total_mb > 0:
        print(f"  Swap:    {mem.swap_total_mb / 1024:.1f} GB total / {mem.swap_free_mb / 1024:.1f} GB free")

    # GPUs
    if profile.gpus:
        print(f"\n  GPUs ({len(profile.gpus)}):")
        for gpu in profile.gpus:
            vram_str = f"{gpu.vram_mb / 1024:.1f} GB VRAM" if gpu.vram_mb else "VRAM unknown"
            free_str = f" ({gpu.vram_free_mb / 1024:.1f} GB free)" if gpu.vram_free_mb else ""
            cc_str = f" [CC {gpu.compute_capability}]" if gpu.compute_capability else ""
            print(f"    [{gpu.index}] {gpu.name} — {vram_str}{free_str}{cc_str}")
            print(f"        Backend: {gpu.backend.upper()} | Driver: {gpu.driver_version or 'N/A'}")
            if gpu.cuda_version:
                print(f"        CUDA: {gpu.cuda_version}")
    else:
        print(f"\n  GPUs:    None detected (CPU-only mode)")

    # Recommendation
    print(f"\n  Backend: {profile.best_backend.upper()}")
    if profile.can_gpu:
        print(f"  Mode:    GPU-accelerated inference")
    else:
        print(f"  Mode:    CPU-only inference (optimize with quantized models)")

    print(f"{'═' * 60}\n")


# Cache the detection result
_cached_profile: Optional[HardwareProfile] = None


def get_hardware_profile(force_refresh: bool = False) -> HardwareProfile:
    """Get hardware profile (cached after first detection)."""
    global _cached_profile
    if _cached_profile is None or force_refresh:
        _cached_profile = detect_hardware()
    return _cached_profile
