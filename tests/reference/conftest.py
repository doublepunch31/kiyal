import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tests.pinned_models import SMOLLM2_REPO, SMOLLM2_REVISION


@pytest.fixture(scope="session")
def smollm2_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(SMOLLM2_REPO, revision=SMOLLM2_REVISION)
