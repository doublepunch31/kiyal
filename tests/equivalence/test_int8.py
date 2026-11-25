from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from benchmarks.perplexity import compare, load_wikitext2, sliding_windows
from kiyal.config import EngineConfig
from kiyal.engine.sequence import SamplingParams
from kiyal.model.llama import LlamaModel
from kiyal.model.loader import load_model
from kiyal.quant.int8 import QuantLinear
from kiyal.tokenizer import Tokenizer
from tests.equivalence.continuous import (
    CPU,
    Setup,
    assert_continuous_logits_match,
    assert_same_generation,
)
from tests.equivalence.prompts import TINY_TEXTS, random_prompts

# Unlike the other rungs, int8 weights change the outputs, so the SmolLM2 check is on how
# much: perplexity on the start of WikiText-2 may grow by at most MAX_PERPLEXITY_INCREASE,
# and the int8 model's most likely next token must equal the fp32 model's at no fewer than
# MIN_TOP1_AGREEMENT of the positions. The reference is fp32, since the ladder runs on the
# CPU. Measured on an Apple M4: +0.21% and 97.3%. Two broken quantizers fail both limits:
# truncating instead of rounding gave +1.03% and 93.9%, one scale for the whole weight
# instead of one per row gave +2.67% and 89.5%.
WIKITEXT_TOKENS = 16384
WINDOW = 1024
STRIDE = 512
MAX_PERPLEXITY_INCREASE = 0.0075
MIN_TOP1_AGREEMENT = 0.95
# The int8 model is still one model, so through the engine it must give what it gives alone,
# within the earlier rungs' tolerance.
TINY_ATOL = 2e-5


@pytest.fixture(scope="module")
def tiny_int8(make_tiny_checkpoint: Callable[..., Path], tiny_tokenizer: Tokenizer) -> LlamaModel:
    checkpoint = make_tiny_checkpoint(
        vocab_size=tiny_tokenizer.vocab_size,
        bos_token_id=tiny_tokenizer.bos_token_id,
        eos_token_id=tiny_tokenizer.eos_token_id,
    )
    model = load_model(checkpoint, device=CPU, dtype=torch.float32, quantization="int8")
    assert isinstance(model.get_submodule("layers.0.mlp.down_proj"), QuantLinear)
    return model


def test_int8_logits_through_the_engine_match_the_prompts_alone(
    tiny_int8: LlamaModel, tiny_tokenizer: Tokenizer
) -> None:
    config = EngineConfig(
        max_num_seqs=3, max_num_batched_tokens=48, block_size=4, enable_prefix_caching=False
    )
    setup = Setup(tiny_int8, tiny_tokenizer, config, TINY_ATOL)
    prompts = random_prompts(tiny_int8.config.vocab_size, [1, 7, 40, 19, 33])
    assert_continuous_logits_match(setup, prompts, [30, 12, 40, 25, 18])


def test_int8_generation_through_the_engine_matches_the_prompts_alone(
    tiny_int8: LlamaModel, tiny_tokenizer: Tokenizer
) -> None:
    config = EngineConfig(max_num_seqs=2, block_size=4)
    setup = Setup(tiny_int8, tiny_tokenizer, config, TINY_ATOL)
    prompts = [tiny_tokenizer.encode(text) for text in TINY_TEXTS.values()]
    params = [SamplingParams(max_tokens=24, temperature=0) for _ in prompts]
    assert_same_generation(setup, prompts, params)


@pytest.fixture(scope="module")
def smollm2_int8(smollm2_checkpoint: Path) -> LlamaModel:
    return load_model(
        smollm2_checkpoint, device=torch.device("cpu"), dtype=torch.float32, quantization="int8"
    )


@pytest.mark.slow
def test_int8_perplexity_and_agreement_stay_within_their_thresholds(
    smollm2: LlamaModel, smollm2_int8: LlamaModel, smollm2_kiyal_tokenizer: Tokenizer
) -> None:
    text = load_wikitext2()
    token_ids = smollm2_kiyal_tokenizer.encode(text, add_special_tokens=False)[:WIKITEXT_TOKENS]
    windows = sliding_windows(len(token_ids), WINDOW, STRIDE)
    comparison = compare(smollm2, smollm2_int8, token_ids, windows)
    assert comparison.num_scored == WIKITEXT_TOKENS - 1
    assert comparison.perplexity_increase <= MAX_PERPLEXITY_INCREASE
    assert comparison.agreement >= MIN_TOP1_AGREEMENT
