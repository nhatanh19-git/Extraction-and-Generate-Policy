"""Preprocessing component: contraction expansion and text normalization.

[EXTENDED] – Pure string/regex preprocessing; no NLP model dependency.
Does NOT contain any domain wordlists. Outputs normalized text into
``context.metadata["normalized_text"]`` for subsequent components to use.
"""

from __future__ import annotations

import re

from ..core.interfaces import PipelineComponent
from ..core.schemas import PipelineContext

# ---------------------------------------------------------------------------
# Contraction map — structural/syntactic knowledge only (not domain terms)
# ---------------------------------------------------------------------------
_CONTRACTIONS: dict[str, str] = {
    r"\bcan't\b": "cannot",
    r"\bcannot\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\baren't\b": "are not",
    r"\bisn't\b": "is not",
    r"\bwasn't\b": "was not",
    r"\bweren't\b": "were not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwouldn't\b": "would not",
    r"\bshouldn't\b": "should not",
    r"\bcouldn't\b": "could not",
    r"\bmightn't\b": "might not",
    r"\bmustn't\b": "must not",
    r"\bneedn't\b": "need not",
    r"\bI'm\b": "I am",
    r"\bI've\b": "I have",
    r"\bI'll\b": "I will",
    r"\bI'd\b": "I would",
    r"\byou're\b": "you are",
    r"\bthey're\b": "they are",
    r"\bwe're\b": "we are",
    r"\bhe's\b": "he is",
    r"\bshe's\b": "she is",
    r"\bit's\b": "it is",
    r"\bthat's\b": "that is",
    r"\bwho's\b": "who is",
    r"\bwhat's\b": "what is",
    r"\bthere's\b": "there is",
}


def expand_contractions(text: str) -> str:
    """Expand English contractions to their full forms.

    Uses only regex pattern matching — no wordlist lookup via Gazetteer
    because contractions are structural/linguistic facts, not domain knowledge.

    Parameters
    ----------
    text : str
        Raw input sentence.

    Returns
    -------
    str
        Sentence with contractions expanded.
    """
    for pattern, replacement in _CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs/newlines into a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


class PreprocessingComponent(PipelineComponent):
    """Normalize raw NLACP text before dependency parsing.

    Steps (in order):
    1. Expand English contractions (``can't`` → ``cannot``).
    2. Normalize whitespace.

    Writes ``context.metadata["normalized_text"]`` with the cleaned string
    and replaces ``context.doc`` with a new Doc parsed from the normalized
    text *if* the pipeline's NLP model is available via the context. However,
    since components do not own the NLP model, the normalized text is simply
    stored in metadata. The pipeline orchestrator re-runs ``nlp()`` only when
    told to (see ``RuleBasedSAOExtractor`` which passes normalized text to the
    pipeline before parsing).

    Note: No domain wordlists here. All domain knowledge lives in Gazetteers.
    """

    @property
    def name(self) -> str:
        return "preprocessing"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Normalize ``context.original_text`` and store the result in metadata."""
        text = context.original_text
        text = expand_contractions(text)
        text = normalize_whitespace(text)
        context.metadata["normalized_text"] = text
        return context
