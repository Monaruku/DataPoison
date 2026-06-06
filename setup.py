"""
DataPoison setup script.

Comprehensive system detection and dependency installation.
Detects Python version, OS, GPU architecture, VRAM, system RAM,
and installs the best compatible versions of every dependency.

Usage:
    python setup.py
"""
import platform
import subprocess
import sys
import os
import importlib
import re


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def run(cmd: str, capture: bool = False) -> str:
    """Run a shell command and return stdout."""
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if result.returncode != 0 and capture:
        return ""
    return result.stdout.strip() if capture else ""


def pip_install(packages: str, index_url: str = None, upgrade: bool = False):
    """Install pip packages with optional custom index."""
    cmd = f"{sys.executable} -m pip install {packages}"
    if index_url:
        cmd += f" --index-url {index_url}"
    if upgrade:
        cmd += " --upgrade"
    subprocess.run(cmd, shell=True)


def get_installed_version(package_name: str) -> str | None:
    """Return installed version of a package, or None if not installed."""
    try:
        mod = importlib.import_module(package_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return None


def parse_version(version_str: str) -> tuple:
    """Parse a version string like '2.11.0+cu128' into a comparable tuple."""
    clean = re.split(r"[+a-zA-Z]", version_str)[0]
    try:
        return tuple(int(x) for x in clean.split("."))
    except (ValueError, AttributeError):
        return (0,)


# ─────────────────────────────────────────────────────────────────────────
# System Detection
# ─────────────────────────────────────────────────────────────────────────

def detect_system() -> dict:
    """Detect OS, architecture, and Python version."""
    return {
        "os": platform.system(),             # Windows, Linux, Darwin
        "os_version": platform.version(),
        "arch": platform.machine(),           # AMD64, x86_64, arm64
        "python_version": sys.version_info,   # (3, 12, 10)
        "python_str": platform.python_version(),
        "python_path": sys.executable,
    }


def detect_hardware() -> dict:
    """Detect RAM and CPU info."""
    info = {"ram_gb": None, "cpu_name": None, "cpu_cores": os.cpu_count()}

    # RAM detection
    try:
        if platform.system() == "Windows":
            # Try PowerShell first (wmic is deprecated on Windows 11 25H2+)
            output = run(
                'powershell -Command "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"',
                capture=True,
            )
            if output and output.isdigit():
                info["ram_gb"] = round(int(output) / (1024 ** 3), 1)
            else:
                # Fallback: wmic
                output = run(
                    "wmic ComputerSystem get TotalPhysicalMemory", capture=True
                )
                if output:
                    lines = [l.strip() for l in output.split("\n") if l.strip()]
                    if len(lines) >= 2 and lines[1].isdigit():
                        info["ram_gb"] = round(int(lines[1]) / (1024 ** 3), 1)
        elif platform.system() == "Linux":
            output = run("cat /proc/meminfo", capture=True)
            match = re.search(r"MemTotal:\s+(\d+)", output)
            if match:
                info["ram_gb"] = round(int(match.group(1)) / (1024 ** 2), 1)
        elif platform.system() == "Darwin":
            output = run("sysctl -n hw.memsize", capture=True)
            if output and output.isdigit():
                info["ram_gb"] = round(int(output) / (1024 ** 3), 1)
    except Exception:
        pass

    # CPU name
    try:
        if platform.system() == "Windows":
            # PowerShell (works on Windows 25H2+)
            output = run(
                'powershell -Command "(Get-CimInstance Win32_Processor).Name"',
                capture=True,
            )
            if output:
                info["cpu_name"] = output.strip()
            else:
                # Fallback
                output = run("wmic cpu get name", capture=True)
                if output:
                    lines = [l.strip() for l in output.split("\n") if l.strip()]
                    if len(lines) >= 2:
                        info["cpu_name"] = lines[1]
        else:
            info["cpu_name"] = platform.processor() or "Unknown"
    except Exception:
        pass

    return info


def detect_gpu() -> dict:
    """Detect NVIDIA GPU, VRAM, compute capability, and driver."""
    info = {
        "has_gpu": False, "name": None, "vram_gb": None,
        "compute_capability": None, "sm": None, "driver": None,
        "cuda_driver_version": None,
    }

    try:
        output = run(
            "nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version "
            "--format=csv,noheader",
            capture=True,
        )
        if output and "NVIDIA" in output.upper():
            parts = [p.strip() for p in output.split(",")]
            info["has_gpu"] = True
            info["name"] = parts[0] if parts else "Unknown NVIDIA GPU"
            if len(parts) >= 2:
                mem_match = re.search(r"(\d+)", parts[1])
                if mem_match:
                    info["vram_gb"] = round(int(mem_match.group(1)) / 1024, 1)
            if len(parts) >= 3:
                cc = parts[2]
                info["compute_capability"] = cc
                try:
                    major, minor = cc.split(".")
                    info["sm"] = f"sm_{major}{minor}"
                except ValueError:
                    pass
            if len(parts) >= 4:
                info["driver"] = parts[3]

        # CUDA driver version from nvidia-smi header
        smi_full = run("nvidia-smi", capture=True)
        match = re.search(r"CUDA Version:\s*([\d.]+)", smi_full)
        if match:
            info["cuda_driver_version"] = match.group(1)
    except Exception:
        pass

    return info


# ─────────────────────────────────────────────────────────────────────────
# Dependency Resolution
# ─────────────────────────────────────────────────────────────────────────

# Minimum required versions
MIN_VERSIONS = {
    "torch": (2, 2, 0),
    "torchvision": (0, 17, 0),
    "customtkinter": (5, 2, 0),
    "PIL": (10, 2, 0),         # Pillow
    "numpy": (1, 26, 0),
    "piexif": (1, 1, 3),
    "tqdm": (4, 66, 0),
}

# Map package import names to pip package names
PIP_NAMES = {
    "PIL": "Pillow",
    "torch": "torch",
    "torchvision": "torchvision",
    "customtkinter": "customtkinter",
    "numpy": "numpy",
    "piexif": "piexif",
    "tqdm": "tqdm",
}


def get_pytorch_config(gpu_info: dict) -> dict:
    """
    Determine the best PyTorch version and index URL for this system.

    Returns dict with: index_url, cuda_label, reason
    """
    sm = gpu_info.get("sm")
    vram = gpu_info.get("vram_gb") or 0
    cuda_driver = gpu_info.get("cuda_driver_version")

    if not gpu_info["has_gpu"]:
        return {
            "index_url": None,
            "cuda_label": "CPU-only",
            "reason": "No NVIDIA GPU detected",
        }

    # Map SM to minimum required CUDA toolkit version
    # Blackwell sm_120 needs CUDA 12.8+
    # Ada sm_89, Hopper sm_90 work with CUDA 12.1+ but 12.8 is best
    # Ampere sm_80-86 work with CUDA 11.8+ but 12.8 is best
    if sm and sm.startswith("sm_12"):
        return {
            "index_url": "https://download.pytorch.org/whl/cu128",
            "cuda_label": "CUDA 12.8",
            "reason": f"Blackwell architecture ({sm}) requires CUDA 12.8+",
        }
    elif sm:
        # For all modern GPUs, CUDA 12.8 gives best compatibility + perf
        return {
            "index_url": "https://download.pytorch.org/whl/cu128",
            "cuda_label": "CUDA 12.8",
            "reason": f"{sm} architecture — CUDA 12.8 for best compatibility",
        }
    else:
        return {
            "index_url": "https://download.pytorch.org/whl/cu128",
            "cuda_label": "CUDA 12.8",
            "reason": "Unknown GPU architecture — defaulting to CUDA 12.8",
        }


def check_dependencies() -> dict:
    """Check which dependencies are installed and at what versions."""
    status = {}
    for import_name, min_ver in MIN_VERSIONS.items():
        installed = get_installed_version(import_name)
        pip_name = PIP_NAMES.get(import_name, import_name)
        if installed:
            current = parse_version(installed)
            needs_upgrade = current < min_ver
            status[import_name] = {
                "pip_name": pip_name,
                "installed": installed,
                "min_required": ".".join(str(x) for x in min_ver),
                "ok": not needs_upgrade,
                "action": "upgrade" if needs_upgrade else "skip",
            }
        else:
            status[import_name] = {
                "pip_name": pip_name,
                "installed": None,
                "min_required": ".".join(str(x) for x in min_ver),
                "ok": False,
                "action": "install",
            }
    return status


# ─────────────────────────────────────────────────────────────────────────
# Installation
# ─────────────────────────────────────────────────────────────────────────

def install_base_deps(dep_status: dict):
    """Install or upgrade non-PyTorch dependencies."""
    to_install = []
    for import_name, info in dep_status.items():
        if import_name in ("torch", "torchvision"):
            continue  # Handled separately
        if info["action"] in ("install", "upgrade"):
            to_install.append(info["pip_name"])

    if not to_install:
        print("  All base dependencies are up to date.")
        return

    print(f"  Installing/upgrading: {', '.join(to_install)}")
    # Build install spec with minimum version constraints
    specs = []
    for name in to_install:
        # Find the min version for this pip name
        for imp, pip_n in PIP_NAMES.items():
            if pip_n == name:
                min_v = MIN_VERSIONS[imp]
                ver_str = ".".join(str(x) for x in min_v)
                specs.append(f"{name}>={ver_str}")
                break
    if specs:
        pip_install(" ".join(specs), upgrade=True)
    else:
        pip_install(" ".join(to_install), upgrade=True)


def install_pytorch(dep_status: dict, pytorch_config: dict):
    """Install or verify PyTorch with correct CUDA build."""
    torch_info = dep_status.get("torch", {})
    tv_info = dep_status.get("torchvision", {})

    index_url = pytorch_config["index_url"]
    need_install = (
        torch_info.get("action") in ("install", "upgrade")
        or tv_info.get("action") in ("install", "upgrade")
    )

    # Even if installed, check if it's the right CUDA build
    if torch_info.get("installed"):
        current_ver = torch_info["installed"]
        if index_url and f"cu{index_url.split('cu')[-1]}" not in current_ver:
            print(f"  PyTorch {current_ver} found but missing CUDA support.")
            print(f"  Reinstalling with {pytorch_config['cuda_label']}...")
            need_install = True
        elif torch_info.get("ok"):
            print(f"  PyTorch {current_ver} already installed with correct build.")
            need_install = False

    if not need_install:
        return

    label = pytorch_config["cuda_label"]
    print(f"  Installing PyTorch with {label} support...")
    if index_url:
        pip_install("torch torchvision", index_url=index_url, upgrade=True)
    else:
        pip_install("torch torchvision", upgrade=True)


# ─────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────

def verify_install():
    """Verify all dependencies and report status."""
    print("\n[4/5] Verifying installation...\n")

    verify_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_verify_install.py")
    with open(verify_script, "w") as f:
        f.write("""
import sys
OK = True
def check(name, import_name, min_ver_str):
    global OK
    try:
        mod = __import__(import_name)
        ver = getattr(mod, '__version__', 'ok')
        print(f"  {name:<16s} {ver}")
    except ImportError:
        print(f"  {name:<16s} NOT INSTALLED")
        OK = False

print("  Installed versions:")
check("PyTorch", "torch", "2.2.0")
check("TorchVision", "torchvision", "0.17.0")
check("CustomTkinter", "customtkinter", "5.2.0")
check("Pillow", "PIL", "10.2.0")
check("NumPy", "numpy", "1.26.0")
check("piexif", "piexif", "1.1.3")
check("tqdm", "tqdm", "4.66.0")

print()
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Compute:  GPU ({name}, {mem:.1f} GB) - CUDA {torch.version.cuda}")
    else:
        print(f"  Compute:  CPU only")
except Exception as e:
    print(f"  Compute:  Error - {e}")
    OK = False

if OK:
    print("\\n  Status: ALL CHECKS PASSED")
else:
    print("\\n  Status: SOME CHECKS FAILED - re-run setup.py")
    sys.exit(1)
""")

    run(f"{sys.executable} {verify_script}")

    try:
        os.remove(verify_script)
    except OSError:
        pass


# ─────────────────────────────────────────────────────────────────────────
# Model Pre-Download
# ─────────────────────────────────────────────────────────────────────────

def download_models():
    """Pre-download all surrogate model weights so they're available offline."""
    print("\n[5/5] Pre-downloading surrogate models...\n")

    download_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_download_models.py")
    with open(download_script, "w") as f:
        f.write("""
import sys
import os

MODELS = {
    "ResNet-18":    lambda: __import__('torchvision').models.resnet18(
                        weights=__import__('torchvision').models.ResNet18_Weights.DEFAULT),
    "MobileNet-V2":  lambda: __import__('torchvision').models.mobilenet_v2(
                        weights=__import__('torchvision').models.MobileNet_V2_Weights.DEFAULT),
    "EfficientNet-B0": lambda: __import__('torchvision').models.efficientnet_b0(
                        weights=__import__('torchvision').models.EfficientNet_B0_Weights.DEFAULT),
}

all_ok = True
for name, loader in MODELS.items():
    try:
        print(f"  {name:<20s} ", end="", flush=True)
        model = loader()
        # Get approximate size from param count
        params = sum(p.numel() for p in model.parameters())
        size_mb = params * 4 / 1e6  # ~4 bytes per float32 param
        print(f"OK ({size_mb:.0f} MB)")
        del model
    except Exception as e:
        print(f"FAILED ({e})")
        all_ok = False

if all_ok:
    print("\\n  All models downloaded successfully.")
else:
    print("\\n  WARNING: Some models failed to download.")
    print("  They will be retried on first use when launching the app.")
""")

    run(f"{sys.executable} {download_script}")

    try:
        os.remove(download_script)
    except OSError:
        pass


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  DataPoison Setup")
    print("  Protecting artwork from unauthorized AI training")
    print("=" * 62)

    # ── Step 1: System Detection ─────────────────────────────────────
    print("\n[1/5] Detecting system specs...\n")

    system = detect_system()
    hardware = detect_hardware()
    gpu = detect_gpu()

    # Python version
    py = system["python_version"]
    py_ok = py >= (3, 10)
    print(f"  Python:     {system['python_str']} ({system['arch']}) "
          f"{'OK' if py_ok else 'WARNING: 3.10+ recommended'}")

    # OS
    print(f"  OS:         {system['os']} {system['os_version'][:30]}")

    # CPU
    cpu = hardware.get("cpu_name", "Unknown")
    cores = hardware.get("cpu_cores", "?")
    if cpu:
        print(f"  CPU:        {cpu} ({cores} cores)")

    # RAM
    ram = hardware.get("ram_gb")
    if ram:
        ram_ok = ram >= 8
        print(f"  RAM:        {ram} GB {'OK' if ram_ok else 'WARNING: 8+ GB recommended'}")

    # GPU
    if gpu["has_gpu"]:
        cc = gpu.get("compute_capability", "?")
        vram = f"{gpu['vram_gb']} GB" if gpu.get("vram_gb") else "?"
        driver = gpu.get("driver", "?")
        cuda_drv = gpu.get("cuda_driver_version", "?")
        print(f"  GPU:        {gpu['name']}")
        print(f"  VRAM:       {vram}")
        print(f"  Compute:    {cc} ({gpu.get('sm', '?')})")
        print(f"  Driver:     {driver} (CUDA {cuda_drv})")
    else:
        print("  GPU:        None detected (CPU mode)")

    # ── Step 2: Dependency Check ─────────────────────────────────────
    print("\n[2/5] Checking existing dependencies...\n")

    dep_status = check_dependencies()
    for import_name, info in dep_status.items():
        pip_name = info["pip_name"]
        if info["installed"]:
            mark = "OK" if info["ok"] else "NEEDS UPGRADE"
            print(f"  {pip_name:<18s} {info['installed']:<20s} {mark}")
        else:
            print(f"  {pip_name:<18s} {'not installed':<20s} WILL INSTALL")

    # Determine PyTorch config
    pytorch_config = get_pytorch_config(gpu)
    print(f"\n  PyTorch target: {pytorch_config['cuda_label']}")
    print(f"  Reason: {pytorch_config['reason']}")

    # ── Step 3: Install ──────────────────────────────────────────────
    print("\n[3/5] Installing dependencies...\n")

    print("  Base packages:")
    install_base_deps(dep_status)

    print("\n  PyTorch:")
    install_pytorch(dep_status, pytorch_config)

    # ── Step 4: Verify ───────────────────────────────────────────────
    verify_install()

    # ── Step 5: Pre-download models ───────────────────────────────────
    download_models()

    # ── Done ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Setup complete! Launch with:")
    print("    python main.py")
    print("=" * 62)


if __name__ == "__main__":
    main()
