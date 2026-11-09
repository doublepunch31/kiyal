from pathlib import Path

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kiyal.model.loader import resolve_model_dir

# Revisions are pinned so that a new upload cannot change the reference. CI keys its Hugging Face
# cache on a hash of this file, so changing a pin here also refreshes the cache.
SMOLLM2_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
SMOLLM2_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"


@pytest.fixture(scope="session")
def smollm2_checkpoint() -> Path:
    return resolve_model_dir(SMOLLM2_REPO, revision=SMOLLM2_REVISION)


@pytest.fixture(scope="session")
def smollm2_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(SMOLLM2_REPO, revision=SMOLLM2_REVISION)
