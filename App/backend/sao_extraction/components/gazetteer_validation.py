"""Gazetteer validation and confidence scoring component.

[EXTENDED] – Validates extracted Subject and Object candidates against
domain-knowledge Gazetteers (subject_roles.json, resource_types.json) and
adjusts the confidence score of each triplet candidate accordingly.
All wordlist lookups go through the GazetteerProvider interface — no
hardcoded domain terms in this file.

[BASELINE: Alohaly et al. 2019] – Confidence scoring is a project extension
hook (see ``SAOTriplet.confidence``). In the original paper, a CNN classifier
provides probabilistic output for attribute-value pairs; the ``confidence``
field here serves as a placeholder for that value in future versions.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..core.interfaces import GazetteerProvider, PipelineComponent
from ..core.schemas import AttributeConstraint, PipelineContext, SAOTriplet

logger = logging.getLogger(__name__)

# Confidence deltas — tuned for rule-based baseline (Version 1)
_CONFIDENCE_SUBJECT_MATCH = 0.15    # bonus when subject matches a known role
_CONFIDENCE_OBJECT_MATCH = 0.10     # bonus when object matches a known resource
_CONFIDENCE_SUBJECT_MISSING = -0.30  # penalty when subject is None
_CONFIDENCE_ACTION_MISSING = -0.50   # penalty when action is None (triplet invalid)
_CONFIDENCE_OBJECTS_EMPTY = -0.20    # penalty when no objects found
_CONFIDENCE_BASE = 1.0
_CONFIDENCE_MIN = 0.0
_CONFIDENCE_MAX = 1.0


def _clamp(value: float, lo: float = _CONFIDENCE_MIN, hi: float = _CONFIDENCE_MAX) -> float:
    return max(lo, min(hi, value))


def _subject_head_token(subject_text: Optional[str], doc: Any) -> Optional[Any]:
    """Find the last content token of a subject text string in the doc.

    Used to locate the spaCy token for Gazetteer.match().
    """
    if not subject_text:
        return None
    # Search the doc for a token whose text matches the last word of the subject
    last_word = subject_text.strip().split()[-1].lower()
    for token in doc:
        if token.text.lower() == last_word and token.pos_ in ("NOUN", "PROPN"):
            return token
    return None


def _object_head_token(object_text: str, doc: Any) -> Optional[Any]:
    """Find the head noun token of an object text string in the doc."""
    last_word = object_text.strip().split()[-1].lower()
    for token in doc:
        if token.text.lower() == last_word and token.pos_ in ("NOUN", "PROPN"):
            return token
    return None


def compute_triplet_confidence(
    subject: Optional[str],
    action: Optional[str],
    objects: list[str],
    doc: Any,
    gazetteer: GazetteerProvider,
) -> float:
    """Compute confidence score for a triplet using Gazetteer validation.

    Parameters
    ----------
    subject : str | None
    action : str | None
    objects : list[str]
    doc : spacy.tokens.Doc
        Full document for token lookup.
    gazetteer : GazetteerProvider
        Loaded gazetteer with ``roles`` (subject_roles) and
        ``resources`` (resource_types) categories.

    Returns
    -------
    float
        Confidence in [0.0, 1.0].
    """
    score = _CONFIDENCE_BASE

    # Penalty for missing core elements
    if not action:
        score += _CONFIDENCE_ACTION_MISSING
    if not subject:
        score += _CONFIDENCE_SUBJECT_MISSING
    if not objects:
        score += _CONFIDENCE_OBJECTS_EMPTY

    # Bonus for Gazetteer-validated subject role
    if subject:
        subj_token = _subject_head_token(subject, doc)
        if subj_token and gazetteer.match(subj_token, "roles"):
            score += _CONFIDENCE_SUBJECT_MATCH
            logger.debug("Subject '%s' matched Gazetteer roles.", subject)

    # Bonus for Gazetteer-validated object resource type
    for obj_text in objects:
        obj_token = _object_head_token(obj_text, doc)
        if obj_token and gazetteer.match(obj_token, "resources"):
            score += _CONFIDENCE_OBJECT_MATCH
            logger.debug("Object '%s' matched Gazetteer resources.", obj_text)
            break  # Only bonus once per triplet

    return _clamp(score)


def assemble_triplets(context: PipelineContext, gazetteer: GazetteerProvider) -> list[SAOTriplet]:
    """Assemble final SAOTriplet list from all per-clause metadata.

    Reads ``subjects``, ``actions``, ``objects``, ``clause_meta``, and
    ``attribute_constraints`` from context metadata, then builds one
    ``SAOTriplet`` per clause. Confidence is computed via
    ``compute_triplet_confidence()``.

    Parameters
    ----------
    context : PipelineContext
    gazetteer : GazetteerProvider

    Returns
    -------
    list[SAOTriplet]
    """
    clauses = context.clause_docs or [context.doc[:]]
    n = len(clauses)

    subjects: list[Optional[str]] = context.metadata.get("subjects", [None] * n)
    actions: list[Optional[str]] = context.metadata.get("actions", [None] * n)
    objects_per_clause: list[list[str]] = context.metadata.get("objects", [[] for _ in range(n)])
    clause_meta: list[dict] = context.metadata.get("clause_meta", [{}] * n)
    all_constraints: list[list[AttributeConstraint]] = context.metadata.get(
        "attribute_constraints", [[] for _ in range(n)]
    )

    triplets: list[SAOTriplet] = []
    for i in range(n):
        subject = subjects[i] if i < len(subjects) else None
        action = actions[i] if i < len(actions) else None
        objs = objects_per_clause[i] if i < len(objects_per_clause) else []
        meta = clause_meta[i] if i < len(clause_meta) else {}
        constraints = all_constraints[i] if i < len(all_constraints) else []

        voice = meta.get("voice", "active")
        effect = meta.get("effect", "Unspecified")

        if action is None:
            logger.debug("Clause %d: no action found, skipping triplet assembly.", i)
            context.warnings.append(f"Clause {i}: no action extracted.")
            continue

        confidence = compute_triplet_confidence(
            subject, action, objs, context.doc, gazetteer
        )

        triplet = SAOTriplet(
            subject=subject,
            action=action,
            objects=objs,
            effect=effect,           # type: ignore[arg-type]
            voice=voice,             # type: ignore[arg-type]
            attribute_constraints=constraints,
            confidence=confidence,
            source="rule_based",
        )
        triplets.append(triplet)
        logger.debug("Assembled triplet: %s", triplet)

    return triplets


class GazetteerValidator(PipelineComponent):
    """Validate extracted candidates and assemble final SAOTriplets.

    This component is the last step of the rule-based pipeline. It:
    1. Looks up extracted subjects/objects in domain Gazetteers.
    2. Computes confidence scores.
    3. Assembles ``SAOTriplet`` objects and stores them in
       ``context.triplet_candidates``.

    Parameters
    ----------
    gazetteer : GazetteerProvider
        Must have ``roles`` (subject_roles) and ``resources``
        (resource_types) categories loaded.
    """

    def __init__(self, gazetteer: GazetteerProvider) -> None:
        self._gazetteer = gazetteer

    @property
    def name(self) -> str:
        return "gazetteer_validation"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Assemble triplets with confidence scores and store in context."""
        triplets = assemble_triplets(context, self._gazetteer)
        context.triplet_candidates = triplets
        return context
