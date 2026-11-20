import dataclasses
from pathlib import Path

import pytest
import torch
from torch import Tensor

from kiyal.cache.paged import PagedKVPool
from kiyal.config import EngineConfig
from kiyal.engine.llm_engine import LLMEngine
from kiyal.engine.model_runner import (
    cuda_kv_cache_bytes,
    kv_block_bytes,
    kv_blocks_for_memory,
    profile_forward,
)
from kiyal.model.attention import AttentionMetadata
from kiyal.model.config import ModelConfig
from kiyal.model.llama import LlamaModel
from kiyal.model.loader import load_model
from kiyal.tokenizer import Tokenizer

GIB = 2**30
# A block of 16 tokens of the tiny model in fp32: 2 layers, keys and values, 2 KV heads of
# 16 dimensions, 4 bytes each.
TINY_BLOCK_BYTES = 2 * 2 * 2 * 16 * 16 * 4


@pytest.fixture(scope="module")
def model(tiny_model_dir: Path) -> LlamaModel:
    return load_model(tiny_model_dir, device=torch.device("cpu"), dtype=torch.float32)


def test_a_block_holds_keys_and_values_of_every_layer(tiny_config: ModelConfig) -> None:
    assert kv_block_bytes(tiny_config, 16, torch.float32) == TINY_BLOCK_BYTES
    assert kv_block_bytes(tiny_config, 16, torch.float16) == TINY_BLOCK_BYTES // 2
    # SmolLM2-135M: 30 layers, 3 KV heads of 64 dimensions, in fp16.
    smollm2 = dataclasses.replace(
        tiny_config, num_hidden_layers=30, num_key_value_heads=3, head_dim=64
    )
    assert kv_block_bytes(smollm2, 16, torch.float16) == 368_640


def test_the_pool_gets_kv_cache_gb_off_cuda(model: LlamaModel) -> None:
    config = EngineConfig(kv_cache_gb=100 * TINY_BLOCK_BYTES / GIB)
    assert kv_blocks_for_memory(model, config, 128) == 100
    # A part of a block does not count.
    config = EngineConfig(kv_cache_gb=100.5 * TINY_BLOCK_BYTES / GIB)
    assert kv_blocks_for_memory(model, config, 128) == 100


def test_the_engine_pool_is_never_larger_than_the_batch_can_use(
    model: LlamaModel, tiny_tokenizer: Tokenizer
) -> None:
    # 4 sequences of the tiny model's 128 tokens take 4 x 8 blocks, far below 1 GiB.
    engine = LLMEngine(model, tiny_tokenizer, EngineConfig(max_num_seqs=4))
    assert engine.num_kv_blocks == 32


def test_the_engine_pool_fits_in_kv_cache_gb(model: LlamaModel, tiny_tokenizer: Tokenizer) -> None:
    config = EngineConfig(max_num_seqs=4, kv_cache_gb=20 * TINY_BLOCK_BYTES / GIB)
    assert LLMEngine(model, tiny_tokenizer, config).num_kv_blocks == 20
    # Explicit blocks override the memory.
    config = dataclasses.replace(config, num_kv_blocks=24)
    assert LLMEngine(model, tiny_tokenizer, config).num_kv_blocks == 24


def test_too_little_memory_for_one_sequence_raises(
    model: LlamaModel, tiny_tokenizer: Tokenizer
) -> None:
    config = EngineConfig(kv_cache_gb=7 * TINY_BLOCK_BYTES / GIB)
    with pytest.raises(ValueError, match=r"pool of 7 x 16 tokens .* give the pool more blocks"):
        LLMEngine(model, tiny_tokenizer, config)


def record_forward_sizes(model: LlamaModel, monkeypatch: pytest.MonkeyPatch) -> list[list[int]]:
    """Makes the model record the length of every sequence it runs, per forward pass."""
    calls: list[list[int]] = []
    forward_flat = model.forward_flat

    def record(
        input_ids: Tensor,
        positions: Tensor,
        pool: PagedKVPool,
        metadata: AttentionMetadata,
        logits_indices: Tensor,
    ) -> Tensor:
        ends = (logits_indices + 1).tolist()
        calls.append([end - start for start, end in zip([0, *ends[:-1]], ends, strict=True)])
        assert len(input_ids) == ends[-1]
        return forward_flat(input_ids, positions, pool, metadata, logits_indices)

    monkeypatch.setattr(model, "forward_flat", record)
    return calls


def test_profiling_runs_the_tokens_as_prompts_of_at_most_max_len(
    model: LlamaModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = record_forward_sizes(model, monkeypatch)
    profile_forward(model, 100, 40, block_size=16)
    profile_forward(model, 80, 40, block_size=16)
    assert calls == [[40, 40, 20], [40, 40]]


def test_cuda_sizing_leaves_room_for_the_profiled_peak_and_other_memory(
    model: LlamaModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A 10 GiB device with 8 GiB free, 1 GiB of it held by this process's allocator, so
    # 1 GiB is used outside it. The weights and the profiled step peak at 2 GiB.
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (8 * GIB, 10 * GIB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 1 * GIB)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 2 * GIB)
    calls = record_forward_sizes(model, monkeypatch)
    config = EngineConfig(max_num_batched_tokens=100, gpu_memory_utilization=0.85)
    assert cuda_kv_cache_bytes(model, config, 64) == int(10 * GIB * 0.85) - 2 * GIB - GIB
    # The largest step: the token budget, in prompts of at most max_len.
    assert calls == [[64, 36]]
    # A preempted sequence of max_len tokens can run alone, past the budget.
    calls.clear()
    cuda_kv_cache_bytes(model, config, 128)
    assert calls == [[128]]
