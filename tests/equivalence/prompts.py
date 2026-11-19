from collections.abc import Sequence

import torch

from kiyal.tokenizer import Tokenizer

# Variants of the tiny model whose attention has other shapes: one KV head per query head,
# one KV head for all of them, and biased projections.
TINY_VARIANTS: dict[str, dict[str, object]] = {
    "gqa": {},
    "mha": {"num_key_value_heads": 4},
    "mqa": {"num_key_value_heads": 1},
    "attention-bias": {"attention_bias": True},
}
TINY_TEXTS = {
    "english": "the cat sat on",
    "russian": "мама мыла раму",
    "kyrgyz": "кыял деген эмне",
    "chinese": "你好世界",
}
SMOLLM2_PROMPTS: dict[str, str | list[dict[str, str]]] = {
    "short": "The capital of France is",
    "code": "def fibonacci(n):\n",
    "paragraph": (
        "A KV cache stores the attention keys and values of the tokens a model has already "
        "processed. Each decoding step then computes keys and values for one new token "
        "only and attends over the stored ones. Without the cache, every step would"
    ),
    "chat": [{"role": "user", "content": "What is the boiling point of water at sea level?"}],
}


def encode_prompt(tokenizer: Tokenizer, prompt: str | list[dict[str, str]]) -> list[int]:
    return tokenizer.encode(prompt) if isinstance(prompt, str) else tokenizer.encode_chat(prompt)


def random_prompts(vocab_size: int, lengths: Sequence[int]) -> list[list[int]]:
    generator = torch.Generator().manual_seed(0)
    return [torch.randint(0, vocab_size, (n,), generator=generator).tolist() for n in lengths]
