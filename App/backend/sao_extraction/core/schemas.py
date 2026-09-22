"""Data schemas for the SAO extraction pipeline.

[BASELINE: Alohaly et al. 2019] – AttributeConstraint captures attribute-value
pairs used for ABAC policy classification, mirroring the attribute structure
described in Alohaly, Takabi & Blanco (2019).

[EXTENDED] – SAOTriplet, PipelineResult, and PipelineContext are project
extensions: explicit S/A/O triplet extraction is not part of the original paper,
which uses S/O as CNN classifier inputs, not pipeline outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


# ---------------------------------------------------------------------------
# [BASELINE: Alohaly et al. 2019]
# ---------------------------------------------------------------------------

@dataclass
class AttributeConstraint:
    """A single attribute constraint extracted from a policy clause.

    Represents an (attribute-relation, value) pair, e.g.
    relation="during", value="working hours".
    """
    relation: str
    value: str


# ---------------------------------------------------------------------------
# [EXTENDED] – explicit triplet and pipeline result schemas
# ---------------------------------------------------------------------------

@dataclass
class SAOTriplet:
    """A fully resolved Subject/Action/Object triplet extracted from an NLACP sentence.

    Fields
    ------
    subject : str or None
        The acting entity (may be None for impersonal constructions).
    action : str
        The access action verb (or verb phrase).
    objects : list[str]
        One or more resources/objects the action targets.
    effect : "Permit" | "Deny" | "Unspecified"
        The policy effect inferred from modal/permission vocabulary.
    voice : "active" | "passive"
        Grammatical voice of the main clause.
    attribute_constraints : list[AttributeConstraint]
        Additional PP-modifiers or relative clauses encoding constraints.
    confidence : float
        Extraction confidence in [0.0, 1.0]. Default 1.0 for rule-based.
        # [HOOK] – overwritten by CNN/SRL/LLM layers in later versions.
    source : Literal
        Which extraction layer produced this triplet.
        # [HOOK] – pluggable for Version 2 (CNN), 3 (SRL), 4 (LLM).
    """
    subject: Optional[str]
    action: str
    objects: list[str]
    effect: Literal["Permit", "Deny", "Unspecified"]
    voice: Literal["active", "passive"]
    attribute_constraints: list[AttributeConstraint] = field(default_factory=list)
    confidence: float = 1.0
    source: Literal["rule_based", "cnn", "srl", "llm"] = "rule_based"


@dataclass
class PipelineResult:
    """Final output of a complete SAO extraction pipeline run.

    Attributes
    ----------
    original_text : str
        The raw input policy sentence (unmodified).
    triplets : list[SAOTriplet]
        All extracted S/A/O triplets (one per independent clause, typically).
    pipeline_version : str
        Version string read from pipeline_config.yaml, e.g. "1.0".
    warnings : list[str]
        Non-fatal issues logged during extraction (e.g. missing subject).
    """
    original_text: str
    triplets: list[SAOTriplet]
    pipeline_version: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class PipelineContext:
    """Mutable context object passed sequentially through PipelineComponents.

    Each component reads from and writes to this context without side-effects
    (no I/O, no global state). This makes each component independently testable
    by constructing a context with a mock spaCy Doc.

    Attributes
    ----------
    original_text : str
        Raw input text (unchanged throughout pipeline).
    doc : Any
        spaCy ``Doc`` object produced by the orchestrator's NLP model.
        Typed as ``Any`` to avoid a hard import of spacy in schemas.
    clause_docs : list[Any]
        spaCy ``Span`` objects for each independent clause detected by
        ClauseSegmenter. Empty until that component runs.
    triplet_candidates : list[SAOTriplet]
        Accumulates triplets as each extraction component runs.
    metadata : dict[str, Any]
        Freeform bag for inter-component communication (e.g. voice, effect,
        subject_span, action_token, object_spans, attribute_constraints).
    warnings : list[str]
        Non-fatal issues accumulated during processing.
    """
    original_text: str
    doc: Any  # spacy.tokens.Doc
    clause_docs: list[Any] = field(default_factory=list)   # list[spacy.tokens.Span]
    triplet_candidates: list[SAOTriplet] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
