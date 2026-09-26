"""Offline doubles for the Voyage client: in-memory tokenizers and a scriptable SDK."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, WordLevel

from linking_engine.embedding.voyage_client import VoyageClient, VoyageSettings
from linking_engine.models import PageText

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

MODEL = "voyage-4-large"
DIMENSION = 16
BASE = "https://example.com"

# Deliberately not unit length, so the client's normalisation is what makes it unit.
RAW_COMPONENT = 3.0
# Every response over-reports by this much, so `api_tokens` is provably the SDK's figure.
API_TOKEN_DRIFT = 7


def word_tokenizer() -> Tokenizer:
    """One whitespace-separated word is one token, with real offsets. No download."""
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tokenizer


BOS, EOS = "<s>", "</s>"
# Whole words and one whole multi-byte char become single tokens; everything else stays bytes.
BPE_WORDS = (" word", " the", "é")


def bpe_tokenizer() -> Tokenizer:
    """Byte-level BPE with NFC and BOS/EOS, no download.

    Emoji stay 4 byte tokens and a CJK char is 2 (its first two bytes merge), so a
    token cut can land inside a character. BOS/EOS have zero-width offsets.
    """
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {token: index for index, token in enumerate([BOS, EOS, *alphabet])}
    chains = [byte_level.pre_tokenize_str(word)[0][0] for word in BPE_WORDS]
    chains.append(byte_level.pre_tokenize_str("日")[0][0][:2])
    merges: list[tuple[str, str]] = []
    for chain in chains:
        left = chain[0]
        for char in chain[1:]:
            merges.append((left, char))
            left += char
            vocab.setdefault(left, len(vocab))
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = byte_level
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{BOS} $A {EOS}", special_tokens=[(BOS, vocab[BOS]), (EOS, vocab[EOS])]
    )
    return tokenizer


def words(text: str) -> int:
    return len(text.split())


def page(index: int, tokens: int) -> PageText:
    """`tokens` words; the first one names the page so the fake can tag its vector."""
    return PageText(url=f"{BASE}/p{index}", text=" ".join([f"p{index}"] + ["w"] * (tokens - 1)))


def page_index(text: str) -> int:
    return int(text.split()[0].removeprefix("p"))


def tagged_vector(text: str, dimension: int) -> list[float]:
    """All RAW_COMPONENT except a spike at the page's index, so argmax identifies the page."""
    vector = [RAW_COMPONENT] * dimension
    vector[page_index(text) % dimension] = 50.0
    return vector


def settings(**overrides: object) -> VoyageSettings:
    values: dict[str, object] = {
        "api_key": "test",
        "context_tokens": 50,
        "max_request_tokens": 130,
        "request_token_budget": 120,
        "max_batch_items": 4,
        "max_attempts": 3,
        "backoff_initial_s": 0.0,
        "backoff_max_s": 0.0,
    }
    values.update(overrides)
    return VoyageSettings(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Call:
    texts: tuple[str, ...]
    model: str
    input_type: str
    output_dimension: int
    truncation: bool


@dataclass(frozen=True)
class FakeResult:
    embeddings: list[list[float]]
    # object, not int, so a test can hand the client a malformed count.
    total_tokens: object


@dataclass
class FakeVoyage:
    """Records every call. Raises `failures` in order, then answers.

    `respond` replaces the vectors for a call, which is how malformed responses are built.
    `hang_s` delays the first `hang_calls` calls (every call when None), for timeout tests;
    `cancelled` counts hangs the client cut short.
    """

    dimension: int = DIMENSION
    failures: list[BaseException] = field(default_factory=list)
    # Raised on that 1-based call number, before `failures`; retries count as calls.
    fail_on: dict[int, BaseException] = field(default_factory=dict)
    respond: Callable[[Sequence[str]], list[list[float]]] | None = None
    # Replaces total_tokens for a call, to build malformed counts.
    usage: Callable[[Sequence[str]], object] | None = None
    hang_s: float = 0.0
    hang_calls: int | None = None
    # False keeps no texts, so the fake cannot grow a memory measurement.
    record_calls: bool = True
    calls: list[Call] = field(default_factory=list)
    call_count: int = 0
    texts_seen: int = 0
    cancelled: int = 0

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
        input_type: str,
        output_dimension: int,
        truncation: bool,
    ) -> FakeResult:
        self.call_count += 1
        number = self.call_count
        self.texts_seen += len(texts)
        if self.record_calls:
            self.calls.append(Call(tuple(texts), model, input_type, output_dimension, truncation))
        if self.hang_s and (self.hang_calls is None or number <= self.hang_calls):
            try:
                await asyncio.sleep(self.hang_s)
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        if number in self.fail_on:
            raise self.fail_on[number]
        if self.failures:
            raise self.failures.pop(0)
        if self.respond is not None:
            vectors = self.respond(texts)
        else:
            vectors = [tagged_vector(text, self.dimension) for text in texts]
        if self.usage is not None:
            return FakeResult(vectors, self.usage(texts))
        return FakeResult(vectors, sum(words(t) for t in texts) + API_TOKEN_DRIFT)


def client(
    fake: FakeVoyage,
    config: VoyageSettings | None = None,
    *,
    tokenizer: Tokenizer | None = None,
) -> VoyageClient:
    return VoyageClient(
        config or settings(),
        model=MODEL,
        dimension=fake.dimension,
        sdk=fake,
        tokenizer=tokenizer or word_tokenizer(),
    )
