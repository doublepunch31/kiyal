from pathlib import Path

import pytest
import torch

from kiyal.model.llama import LlamaModel
from kiyal.model.loader import load_model
from kiyal.tokenizer import Tokenizer


@pytest.fixture(scope="session")
def smollm2(smollm2_checkpoint: Path) -> LlamaModel:
    """SmolLM2-135M in fp32 on the CPU, loaded once for all equivalence tests."""
    return load_model(smollm2_checkpoint, device=torch.device("cpu"), dtype=torch.float32)


@pytest.fixture(scope="session")
def smollm2_kiyal_tokenizer(smollm2_checkpoint: Path) -> Tokenizer:
    return Tokenizer.from_dir(smollm2_checkpoint)
