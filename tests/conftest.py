import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers, processors, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from kiyal.model.config import ModelConfig
from kiyal.tokenizer import Tokenizer

# A tiny Llama in Hugging Face config.json form: 2 layers, 4 query heads sharing 2 KV heads.
TINY_CONFIG: dict[str, object] = {
    "model_type": "llama",
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-5,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    "tie_word_embeddings": True,
    "attention_bias": False,
    "bos_token_id": 1,
    "eos_token_id": 2,
}

TINY_BOS = "<|bos|>"
TINY_EOS = "<|eos|>"
TINY_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|bos|>{{ message['role'] }}\n{{ message['content'] }}<|eos|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|bos|>assistant\n{% endif %}"
)
# Text the tiny tokenizer learns its merges from. Repeated words become single tokens; every
# other character falls back to one token per UTF-8 byte.
TINY_TOKENIZER_CORPUS = [
    "the cat sat on the mat and the dog sat on the log",
    "мама мыла раму, а папа читал газету",
    "кыял деген эмне, кыял деген түш",
    "你好世界，你好朋友",
]


@pytest.fixture
def tiny_config() -> ModelConfig:
    return ModelConfig.from_dict(TINY_CONFIG)


def write_tiny_checkpoint(
    path: Path, config: Mapping[str, object], max_shard_size: str | None, seed: int = 0
) -> Path:
    model = LlamaForCausalLM(LlamaConfig.from_dict(dict(config)))
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            noise = 0.1 * torch.randn(param.shape, generator=generator)
            # Norm weights near 1 and nonzero biases, unlike the transformers defaults of exactly
            # 1 and 0, so a loader that skips either one changes the outputs.
            param.copy_(noise + 1.0 if name.endswith("norm.weight") else noise)
    if max_shard_size is None:
        model.save_pretrained(path)
    else:
        model.save_pretrained(path, max_shard_size=max_shard_size)
    return path


@pytest.fixture(scope="session")
def make_tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., Path]:
    """Returns a function that writes a tiny random checkpoint with transformers.

    Keyword arguments override fields of TINY_CONFIG; `max_shard_size` splits the weights into
    several files. Each distinct checkpoint is written once per session and shared, so tests
    must copy it before modifying it.
    """
    written: dict[str, Path] = {}

    def make(*, max_shard_size: str | None = None, **overrides: object) -> Path:
        key = json.dumps([overrides, max_shard_size], sort_keys=True)
        if key not in written:
            config = {**TINY_CONFIG, **overrides}
            written[key] = write_tiny_checkpoint(
                tmp_path_factory.mktemp("tiny"), config, max_shard_size
            )
        return written[key]

    return make


@pytest.fixture(scope="session")
def tiny_checkpoint(make_tiny_checkpoint: Callable[..., Path]) -> Path:
    """Directory of a tiny random checkpoint written by transformers."""
    return make_tiny_checkpoint()


def write_tiny_tokenizer(path: Path) -> Path:
    # Byte-level BPE like GPT-2's: any text encodes without unknown tokens, and a character
    # outside the learned merges takes one token per UTF-8 byte.
    backend = BackendTokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=320,
        special_tokens=[TINY_BOS, TINY_EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    backend.train_from_iterator(TINY_TOKENIZER_CORPUS, trainer)
    # Plain text gets a leading BOS, as Llama tokenizers add one.
    backend.post_processor = processors.TemplateProcessing(
        single=f"{TINY_BOS} $A", special_tokens=[(TINY_BOS, backend.token_to_id(TINY_BOS))]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token=TINY_BOS, eos_token=TINY_EOS
    )
    tokenizer.chat_template = TINY_CHAT_TEMPLATE
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def tiny_tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory of a tiny byte-level BPE tokenizer with a chat template."""
    return write_tiny_tokenizer(tmp_path_factory.mktemp("tiny-tokenizer"))


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_tokenizer_dir: Path) -> Tokenizer:
    return Tokenizer.from_dir(tiny_tokenizer_dir)


@pytest.fixture(scope="session")
def tiny_model_dir(
    make_tiny_checkpoint: Callable[..., Path],
    tiny_tokenizer_dir: Path,
    tiny_tokenizer: Tokenizer,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Directory holding a tiny random model and the tiny tokenizer it is sized for."""
    checkpoint = make_tiny_checkpoint(
        vocab_size=tiny_tokenizer.vocab_size,
        bos_token_id=tiny_tokenizer.bos_token_id,
        eos_token_id=tiny_tokenizer.eos_token_id,
    )
    model_dir = tmp_path_factory.mktemp("tiny-model")
    shutil.copytree(checkpoint, model_dir, dirs_exist_ok=True)
    shutil.copytree(tiny_tokenizer_dir, model_dir, dirs_exist_ok=True)
    return model_dir
