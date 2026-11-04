import pytest
import torch

from kiyal.device import DeviceUnavailableError, default_dtype, resolve_device, resolve_dtype


def set_available(monkeypatch: pytest.MonkeyPatch, *, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_prefers_cuda_then_mps_then_cpu(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: str
) -> None:
    set_available(monkeypatch, cuda=cuda, mps=mps)
    assert resolve_device("auto") == torch.device(expected)


def test_auto_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    set_available(monkeypatch, cuda=False, mps=False)
    assert resolve_device() == torch.device("cpu")


def test_explicit_cpu_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    set_available(monkeypatch, cuda=True, mps=True)
    assert resolve_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("name", ["cuda", "cuda:0", "mps"])
def test_unavailable_device_raises(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    set_available(monkeypatch, cuda=False, mps=False)
    with pytest.raises(DeviceUnavailableError, match="not available"):
        resolve_device(name)


def test_cuda_index_within_device_count(monkeypatch: pytest.MonkeyPatch) -> None:
    set_available(monkeypatch, cuda=True, mps=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_device("cuda:1") == torch.device("cuda", 1)


def test_cuda_index_beyond_device_count_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    set_available(monkeypatch, cuda=True, mps=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(DeviceUnavailableError, match="1 CUDA device"):
        resolve_device("cuda:1")


@pytest.mark.parametrize(
    ("name", "message"),
    [("gpu", "invalid device"), ("", "invalid device"), ("meta", "unsupported device")],
)
def test_unsupported_device_name_raises(name: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_device(name)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((7, 0), torch.float16),
        ((7, 5), torch.float16),
        ((8, 0), torch.bfloat16),
        ((8, 9), torch.bfloat16),
        ((9, 0), torch.bfloat16),
    ],
    ids=["volta-v100", "turing-t4", "ampere-a100", "ada-l4", "hopper-h100"],
)
def test_cuda_default_dtype_follows_capability(
    capability: tuple[int, int], expected: torch.dtype
) -> None:
    assert default_dtype("cuda", capability) == expected


def test_mps_defaults_to_fp16() -> None:
    assert default_dtype("mps") == torch.float16


def test_cpu_defaults_to_fp32() -> None:
    assert default_dtype("cpu") == torch.float32


def test_cuda_default_dtype_requires_capability() -> None:
    with pytest.raises(ValueError, match="cuda_capability"):
        default_dtype("cuda")


def test_default_dtype_rejects_unknown_device_type() -> None:
    with pytest.raises(ValueError, match="meta"):
        default_dtype("meta")


def test_auto_dtype_on_cpu_is_fp32() -> None:
    assert resolve_dtype("auto", torch.device("cpu")) == torch.float32


def test_auto_dtype_on_cuda_reads_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    queried: list[torch.device] = []

    def fake_capability(device: torch.device) -> tuple[int, int]:
        queried.append(device)
        return (7, 5)

    monkeypatch.setattr(torch.cuda, "get_device_capability", fake_capability)
    assert resolve_dtype("auto", torch.device("cuda", 1)) == torch.float16
    assert queried == [torch.device("cuda", 1)]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("fp32", torch.float32),
        ("float32", torch.float32),
        ("fp16", torch.float16),
        ("float16", torch.float16),
        ("bf16", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
    ],
)
def test_explicit_dtype_overrides_device_default(name: str, expected: torch.dtype) -> None:
    assert resolve_dtype(name, torch.device("cpu")) == expected


def test_unsupported_dtype_raises() -> None:
    with pytest.raises(ValueError, match="'int8'"):
        resolve_dtype("int8", torch.device("cpu"))
