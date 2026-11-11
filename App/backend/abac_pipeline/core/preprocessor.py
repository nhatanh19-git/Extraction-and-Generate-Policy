"""Unified NLP Preprocessor for the ABAC Pipeline.

This module provides the single, canonical preprocessing entry point.
It combines text normalization (contraction expansion, whitespace cleanup)
with spaCy NLP parsing to produce a rich `NLPResult` that all downstream
modules consume.

Architecture:
    Raw text → contraction expansion → whitespace normalization → spaCy parse
    → NLPResult(original, normalized, doc, tokens, lemmas, pos, dep, sentences)

Design Notes:
    - Reuses the contraction expansion and whitespace normalization from
      ``sao_extraction.components.preprocessing`` (pure string operations).
    - Does NOT stop-word filter or lowercase — preserving words like "Only",
      "same", "own", "their" is critical for ABAC constraint detection.
    - Sentence segmentation uses spaCy's built-in sentencizer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import spacy
from spacy.tokens import Doc

# Reuse SAO preprocessing utilities (structural/linguistic, not domain)
from backend.sao_extraction.components.preprocessing import (
    expand_contractions,
    normalize_whitespace,
)

logger = logging.getLogger(__name__)


@dataclass
class TokenInfo:
    """Structured info for a single token."""
    text: str
    lemma: str
    pos: str         # Fine-grained POS tag (e.g., NN, VBZ)
    tag: str         # Coarse POS tag (e.g., NOUN, VERB)
    dep: str         # Dependency label (e.g., nsubj, dobj)
    head_idx: int    # Index of the syntactic head
    idx: int         # Character offset in the normalized text
    is_stop: bool
    is_alpha: bool


@dataclass
class SentenceInfo:
    """Info for a single sentence within the document."""
    text: str
    start_token: int   # Index of first token (in document tokens list)
    end_token: int     # Index of last token + 1
    doc: Optional[Doc] = None  # spaCy Doc for this sentence (for dep-parse)


@dataclass
class NLPResult:
    """Complete NLP preprocessing output consumed by all downstream modules.
    
    This is the canonical data structure that flows through the pipeline.
    Every module that needs NLP features should consume this, not raw text.
    """
    original_text: str                       # Exact user input
    normalized_text: str                     # After contraction expansion + whitespace
    doc: Doc                                 # Full spaCy Doc on normalized text
    tokens: List[TokenInfo] = field(default_factory=list)
    sentences: List[SentenceInfo] = field(default_factory=list)
    
    @property
    def token_texts(self) -> List[str]:
        """List of token strings."""
        return [t.text for t in self.tokens]
    
    @property
    def lemmas(self) -> List[str]:
        """List of lemmatized token strings."""
        return [t.lemma for t in self.tokens]
    
    @property
    def pos_tags(self) -> List[str]:
        """List of fine-grained POS tags."""
        return [t.pos for t in self.tokens]
    
    @property
    def dep_labels(self) -> List[str]:
        """List of dependency labels."""
        return [t.dep for t in self.tokens]


class NLPPreprocessor:
    """Unified NLP preprocessing for the ABAC pipeline.
    
    This class is the single entry point for converting raw ACP text into
    structured NLP features. It loads and caches a spaCy model internally.
    
    Usage::
    
        preprocessor = NLPPreprocessor()
        result = preprocessor.process("A nurse can add an item...")
        print(result.tokens)       # List[TokenInfo]
        print(result.sentences)    # List[SentenceInfo]
        print(result.doc)          # spaCy Doc
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize with a spaCy model.
        
        Args:
            spacy_model: Name of the spaCy model to load.
        """
        logger.info(f"NLPPreprocessor: loading spaCy model '{spacy_model}'")
        self.nlp = spacy.load(spacy_model)
        self._model_name = spacy_model
    
    def _fix_spacy_misparses(self, doc: Doc) -> Doc:
        """Fix recurring spaCy parse anomalies in ACP texts."""
        # 1. Action verbs conjoined to aux/verb (e.g. view and download)
        for tok in doc:
            if tok.text.lower() in ("download", "view", "modify", "upload", "create", "delete", "update") and tok.dep_ == "conj" and tok.head.pos_ in ("VERB", "AUX"):
                tok.pos_ = "VERB"
                tok.tag_ = "VB"
                
        # 2. "learning management system(s)" parse anomaly where "learning" is parsed as VERB
        for i, tok in enumerate(doc):
            if tok.text.lower() == "learning" and i + 1 < len(doc) and "management" in doc[i + 1].text.lower():
                systems_tok = None
                for j in range(i + 1, min(i + 4, len(doc))):
                    if "system" in doc[j].text.lower():
                        systems_tok = doc[j]
                        break
                if systems_tok:
                    tok.pos_ = "NOUN"
                    tok.tag_ = "NN"
                    tok.dep_ = "compound"
                    tok.head = systems_tok
                    if i >= 2 and doc[i - 1].text.lower() == "and":
                        info_tok = doc[i - 2]
                        systems_tok.dep_ = "conj"
                        systems_tok.head = info_tok
                    for next_tok in doc[systems_tok.i + 1 : min(systems_tok.i + 4, len(doc))]:
                        if next_tok.dep_ == "prep" and next_tok.head.i == tok.i:
                            next_tok.head = systems_tok
        return doc

    def process(self, text: str) -> NLPResult:
        """Process raw ACP text into a complete NLPResult.
        
        Steps:
            1. Expand contractions (can't → cannot)
            2. Normalize whitespace (collapse multiple spaces)
            3. Parse with spaCy
            4. Extract token-level features
            5. Extract sentence boundaries
        
        Args:
            text: Raw access control policy sentence.
            
        Returns:
            NLPResult with all NLP features populated.
        """
        # Step 1-2: Text normalization (reuses SAO preprocessing)
        normalized = expand_contractions(text)
        normalized = normalize_whitespace(normalized)
        
        # Step 3: spaCy parse
        doc = self.nlp(normalized)
        doc = self._fix_spacy_misparses(doc)
        
        # Step 4: Extract token-level features
        tokens = []
        for token in doc:
            tokens.append(TokenInfo(
                text=token.text,
                lemma=token.lemma_,
                pos=token.tag_,         # Fine-grained (NN, VBZ, etc.)
                tag=token.pos_,         # Coarse (NOUN, VERB, etc.)
                dep=token.dep_,
                head_idx=token.head.i,
                idx=token.idx,
                is_stop=token.is_stop,
                is_alpha=token.is_alpha,
            ))
        
        # Step 5: Extract sentence boundaries
        sentences = []
        for sent in doc.sents:
            sentences.append(SentenceInfo(
                text=sent.text,
                start_token=sent.start,
                end_token=sent.end,
                doc=sent.as_doc(),
            ))
        
        return NLPResult(
            original_text=text,
            normalized_text=normalized,
            doc=doc,
            tokens=tokens,
            sentences=sentences,
        )
    
    def get_spacy_nlp(self) -> spacy.language.Language:
        """Expose the underlying spaCy model for components that need it directly."""
        return self.nlp
