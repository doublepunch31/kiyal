import json
import platform
import statistics
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import torch

import kiyal

REPO_ROOT = Path(__file__).resolve().parent.parent
# Unreviewed output of benchmark runs, ignored by git. Results worth keeping are copied to
# benchmarks/results/.
RAW_DIR = REPO_ROOT / "benchmarks" / "raw"


def synchronize(device: torch.device) -> None:
    """Waits until the device has finished the work queued on it.

    GPU calls return before the work is done, so a timestamp taken without this would
    leave out whatever is still running.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def capture_environment(device: torch.device, dtype: torch.dtype) -> dict[str, object]:
    """Describes the machine and software a result was measured on."""
    environment: dict[str, object] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "kiyal_version": kiyal.__version__,
        "kiyal_commit": _command_output(["git", "rev-parse", "HEAD"]),
        # True when the measured code differs from the commit.
        "kiyal_uncommitted_changes": bool(_command_output(["git", "status", "--porcelain"])),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "cpu_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        environment |= {
            "gpu": properties.name,
            "gpu_memory_gb": round(properties.total_memory / 2**30, 1),
            "cuda": torch.version.cuda,
            "driver": _command_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            ),
        }
    return environment


def save_result(result: dict[str, object], path: Path) -> Path:
    """Writes a result as JSON, creating the directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def _command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command, cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    lines = completed.stdout.strip().splitlines()
    return lines[0] if lines else ""


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        name = _command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()
