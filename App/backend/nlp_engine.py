"""NLP processing with optional spaCy support and a safe fallback."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenInfo:
    """A token and the linguistic information available for it."""

    text: str
    lemma: str
    pos: str = ""
    dependency: str = ""
    start: int = 0
    end: int = 0


@dataclass(frozen=True)
class NLPDocument:
    """Result of processing one ACP sentence."""

    text: str
    tokens: tuple[TokenInfo, ...]
    model_name: str | None = None


class NLPEngine:
    """Run spaCy when a Vietnamese model exists, otherwise tokenize safely."""

    _model_candidates = ("vi_core_news_lg", "vi_core_news_md", "vi_core_news_sm")

    def __init__(self) -> None:
        self._nlp: Any | None = None
        self.model_name: str | None = None
        self._load_optional_model()

    def _load_optional_model(self) -> None:
        try:
            import spacy
        except ImportError:
            logger.info("spaCy is not installed; using rule-based tokenization")
            return

        for model_name in self._model_candidates:
            try:
                self._nlp = spacy.load(model_name)
                self.model_name = model_name
                logger.info("Loaded Vietnamese spaCy model: %s", model_name)
                return
            except OSError:
                continue
        logger.info("No Vietnamese spaCy model found; using rule-based tokenization")

    def process(self, sentence: str) -> NLPDocument:
        """Process an ACP sentence without failing when an optional model is absent."""
        if not sentence or not sentence.strip():
            raise ValueError("ACP sentence must not be empty")

        clean_sentence = sentence.strip()
        logger.info("NLP processing started")
        if self._nlp is not None:
            parsed = self._nlp(clean_sentence)
            tokens = tuple(
                TokenInfo(
                    text=token.text,
                    lemma=token.lemma_,
                    pos=token.pos_,
                    dependency=token.dep_,
                    start=token.idx,
                    end=token.idx + len(token.text),
                )
                for token in parsed
            )
            return NLPDocument(clean_sentence, tokens, self.model_name)

        tokens = tuple(
            TokenInfo(text=match.group(), lemma=match.group().lower(), start=match.start(), end=match.end())
            for match in re.finditer(r"[^\s]+", clean_sentence)
        )
        return NLPDocument(clean_sentence, tokens)