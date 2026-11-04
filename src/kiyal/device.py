from collections.abc import Mapping
from typing import Final, cast

import torch

SUPPORTED_DEVICE_TYPES: Final = ("cuda", "mps", "cpu")

DTYPES: Final[Mapping[str, torch.dtype]] = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}

# Ampere (compute capability 8.0) is the first generation with bf16 tensor cores.
# Older GPUs such as the T4 (7.5) only emulate bf16, so they default to fp16.
BF16_MIN_CUDA_CAPABILITY: Final = (8, 0)


class DeviceUnavailableError(RuntimeError):
    """Raised when a requested device is not usable on this machine."""


def _is_available(device_type: str) -> bool:
    if device_type == "cuda":
        return torch.cuda.is_available()
    if device_type == "mps":
        return torch.backends.mps.is_available()
    return device_type == "cpu"


def resolve_device(name: str = "auto") -> torch.device:
    """Returns the device to run on.

    Args:
        name: "auto" picks the first available of cuda, mps and cpu. Anything else is
            parsed as a torch device string such as "cuda", "cuda:1", "mps" or "cpu".

    Raises:
        ValueError: If the name is not a supported device string.
        DeviceUnavailableError: If the device does not exist on this machine.
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    try:
        device = torch.device(name)
    except RuntimeError as e:
        raise ValueError(f"invalid device {name!r}") from e
    if device.type not in SUPPORTED_DEVICE_TYPES:
        expected = ", ".join(["auto", *SUPPORTED_DEVICE_TYPES])
        raise ValueError(f"unsupported device {name!r}; expected one of {expected}")

    if not _is_available(device.type):
        raise DeviceUnavailableError(f"device {name!r} is not available on this machine")
    if device.type == "cuda":
        # The torch stubs declare index as int, but it is None when the string has no index.
        index = cast(int | None, device.index)
        count = torch.cuda.device_count()
        if index is not None and index >= count:
            raise DeviceUnavailableError(
                f"device {name!r} does not exist; this machine has {count} CUDA device(s)"
            )
    return device


def default_dtype(device_type: str, cuda_capability: tuple[int, int] | None = None) -> torch.dtype:
    """Returns the default compute dtype for a device type.

    Args:
        device_type: "cuda", "mps" or "cpu".
        cuda_capability: The GPU's (major, minor) compute capability. Required for cuda.

    Raises:
        ValueError: If the device type is unsupported or a cuda capability is missing.
    """
    if device_type == "cuda":
        if cuda_capability is None:
            raise ValueError("cuda_capability is required to pick a dtype for cuda")
        if cuda_capability >= BF16_MIN_CUDA_CAPABILITY:
            return torch.bfloat16
        return torch.float16
    if device_type == "mps":
        return torch.float16
    if device_type == "cpu":
        return torch.float32
    raise ValueError(f"unsupported device type {device_type!r}")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Returns the compute dtype for a device.

    Args:
        name: "auto" for the device default, or a key of DTYPES such as "bf16".
        device: The device the model will run on.

    Raises:
        ValueError: If the name is not "auto" or a key of DTYPES.
    """
    if name == "auto":
        capability = torch.cuda.get_device_capability(device) if device.type == "cuda" else None
        return default_dtype(device.type, capability)
    if name not in DTYPES:
        expected = ", ".join(["auto", *DTYPES])
        raise ValueError(f"unsupported dtype {name!r}; expected one of {expected}")
    return DTYPES[name]
