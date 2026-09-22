"""NLP processing document and token structures for ABAC Policy Studio."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenInfo:
    """Token representation with linguistic metadata."""

    text: str
    lemma: str
    pos: str = ""
    dependency: str = ""
    start: int = 0
    end: int = 0


@dataclass(frozen=True)
class NLPDocument:
    """Processed NLP document containing original text and tokens."""

    text: str
    tokens: tuple[TokenInfo, ...]

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)


class NLPEngine:
    """Lightweight NLP document processor."""

    def process(self, text: str) -> NLPDocument:
        clean_text = text.strip()
        words = clean_text.split()
        tokens = []
        pos = 0
        for word in words:
            start = clean_text.find(word, pos)
            end = start + len(word)
            pos = end
            tokens.append(
                TokenInfo(
                    text=word,
                    lemma=word.lower(),
                    pos="NOUN",
                    dependency="",
                    start=start,
                    end=end,
                )
            )
        return NLPDocument(text=clean_text, tokens=tuple(tokens))
