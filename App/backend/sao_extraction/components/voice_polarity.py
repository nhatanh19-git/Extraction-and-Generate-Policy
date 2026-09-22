"""Voice and polarity detection component.

[EXTENDED] – Detects grammatical voice (active/passive) and policy effect
(Permit/Deny/Unspecified) for each clause in context.clause_docs.
Effect detection uses Gazetteer via GazetteerProvider.match() — NO domain
wordlists are hardcoded here.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from ..core.interfaces import GazetteerProvider, PipelineComponent
from ..core.schemas import PipelineContext

logger = logging.getLogger(__name__)

# Dependency labels that signal passive voice (structural, not domain knowledge)
_PASSIVE_DEPS = frozenset({"nsubjpass", "auxpass"})

# Dependency label for negation (structural)
_NEG_DEP = "neg"


def detect_voice(clause_span: Any) -> Literal["active", "passive"]:
    """Determine the grammatical voice of a clause span.

    Checks for the presence of ``nsubjpass`` or ``auxpass`` dependency
    relations among the tokens of the clause.

    Parameters
    ----------
    clause_span : spacy.tokens.Span
        A clause-level span (from ClauseSegmenter output).

    Returns
    -------
    "active" | "passive"
    """
    for token in clause_span:
        if token.dep_ in _PASSIVE_DEPS:
            return "passive"
    return "active"


def detect_negation(clause_span: Any) -> bool:
    """Return True if the clause contains a negation dependency (``neg``).

    Parameters
    ----------
    clause_span : spacy.tokens.Span

    Returns
    -------
    bool
        True if any token has dep_ == "neg".
    """
    return any(token.dep_ == _NEG_DEP for token in clause_span)


def detect_effect(
    clause_span: Any,
    gazetteer: GazetteerProvider,
    is_negated: bool,
) -> Literal["Permit", "Deny", "Unspecified"]:
    """Determine the policy effect (Permit/Deny) from modal/permission vocabulary.

    Consults the Gazetteer for ``permit`` and ``deny`` categories instead of
    checking any hardcoded list. Effect is inverted when ``is_negated`` is True.

    Algorithm
    ---------
    1. For each token in the clause, check if it matches the ``permit`` category.
    2. For each token in the clause, check if it matches the ``deny`` category.
    3. Return the appropriate effect, applying negation inversion.

    Parameters
    ----------
    clause_span : spacy.tokens.Span
    gazetteer : GazetteerProvider
        Loaded gazetteer provider (e.g. FileBasedGazetteer).
    is_negated : bool
        Whether the clause contains a negation marker.

    Returns
    -------
    "Permit" | "Deny" | "Unspecified"
    """
    found_permit = False
    found_deny = False

    for token in clause_span:
        if gazetteer.match(token, "permit"):
            found_permit = True
        if gazetteer.match(token, "deny"):
            found_deny = True

    # Determine raw effect before applying negation
    if found_deny and not found_permit:
        raw_effect: Literal["Permit", "Deny", "Unspecified"] = "Deny"
    elif found_permit:
        raw_effect = "Permit"
    else:
        raw_effect = "Unspecified"

    # Invert on negation: "may not" → Deny, "must not" → Deny, "cannot" → Deny
    if is_negated and raw_effect == "Permit":
        return "Deny"
    if is_negated and raw_effect == "Deny":
        return "Permit"

    return raw_effect


class VoicePolarityDetector(PipelineComponent):
    """Detect voice and effect for every clause in the pipeline context.

    Requires
    --------
    context.clause_docs : list[spacy.tokens.Span]
        Populated by ClauseSegmenter.

    Reads
    -----
    Gazetteer categories: ``permit``, ``deny`` (from effect_terms.json via
    the injected GazetteerProvider).

    Writes
    ------
    context.metadata["clause_meta"] : list[dict]
        One dict per clause with keys ``voice``, ``effect``, ``negated``.
        Downstream extractors index into this list by clause position.

    Parameters
    ----------
    gazetteer : GazetteerProvider
        Injected gazetteer provider — must have ``permit`` and ``deny``
        categories loaded.
    """

    def __init__(self, gazetteer: GazetteerProvider) -> None:
        self._gazetteer = gazetteer

    @property
    def name(self) -> str:
        return "voice_polarity"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Compute voice/effect metadata for each clause."""
        clause_meta: list[dict] = []

        clauses = context.clause_docs or [context.doc[:]]
        for span in clauses:
            voice = detect_voice(span)
            negated = detect_negation(span)
            effect = detect_effect(span, self._gazetteer, negated)
            clause_meta.append(
                {"voice": voice, "effect": effect, "negated": negated}
            )
            logger.debug(
                "Clause '%s' → voice=%s, effect=%s, negated=%s",
                span.text[:60],
                voice,
                effect,
                negated,
            )

        context.metadata["clause_meta"] = clause_meta
        return context
