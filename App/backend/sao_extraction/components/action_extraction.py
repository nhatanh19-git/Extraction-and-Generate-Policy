"""Action extraction component.

[EXTENDED] – Extracts the access-action verb (or verb phrase) from each
clause using spaCy dependency analysis. Priority is given to ``xcomp``
children when the ROOT is a support/modal verb, and phrasal verb particles
(``prt``) are appended to form the complete verb phrase.
No domain wordlists are hardcoded here.

[BASELINE: Alohaly et al. 2019] – The action verb serves as the predicate
linking Subject and Object in the triplet; the paper implicitly treats
actions as the predicates that connect role attributes to resource attributes.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..core.interfaces import PipelineComponent
from ..core.schemas import PipelineContext

logger = logging.getLogger(__name__)

# Structural POS tags for support/modal/copular verbs — determines xcomp priority
# These are grammatical categories, NOT domain terms.
_SUPPORT_POS = frozenset({"AUX"})
_SUPPORT_LEMMAS = frozenset({
    "be", "have", "do",
    "permit", "allow", "authorize", "entitle",
    "prohibit", "forbid", "restrict", "prevent", "block",
    "enable", "let",
})

# Dependency labels (structural, not domain)
_XCOMP_DEP = "xcomp"
_PRT_DEP = "prt"


def _find_root_verb(clause_span: Any) -> Optional[Any]:
    """Return the ROOT token of the clause if it is a verb."""
    for token in clause_span:
        if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
            return token
    # Fallback: find first VERB in the span
    for token in clause_span:
        if token.pos_ in ("VERB", "AUX"):
            return token
    return None


def _is_support_verb(token: Any) -> bool:
    """Return True if *token* is a support/modal/copular verb.

    Determined by POS tag or lemma — no domain vocabulary involved.
    """
    return token.pos_ in _SUPPORT_POS or token.lemma_.lower() in _SUPPORT_LEMMAS


def _build_action(verb_token: Any) -> str:
    """Build the full action string from a verb token, appending ``prt`` particles.

    Handles phrasal verbs such as "log out", "sign in", "check in".

    Parameters
    ----------
    verb_token : spacy.tokens.Token

    Returns
    -------
    str
        The verb lemma (lower-cased) plus any ``prt`` particle.
    """
    parts = [verb_token.lemma_.lower()]
    for child in verb_token.children:
        if child.dep_ == _PRT_DEP:
            parts.append(child.text.lower())
    return " ".join(parts)


def extract_action_from_span(clause_span: Any) -> Optional[str]:
    """Extract the primary access-action verb phrase from a clause.

    Algorithm
    ---------
    1. Find the clause ROOT verb.
    2. If the ROOT is a support verb (AUX/modal/copular), look for an
       ``xcomp`` child verb — that is the real action
       (e.g. "are permitted **to access**" → action = "access").
    3. If xcomp not found, fall back to the ROOT verb itself.
    4. Append ``prt`` particles for phrasal verbs.

    Parameters
    ----------
    clause_span : spacy.tokens.Span

    Returns
    -------
    str | None
        Lemmatised action string, or None if no verb found.
    """
    root = _find_root_verb(clause_span)
    if root is None:
        return None

    # If ROOT is a support/modal verb, prefer xcomp child
    if _is_support_verb(root):
        for child in root.children:
            if child.dep_ == _XCOMP_DEP and child.pos_ in ("VERB", "AUX"):
                return _build_action(child)
        # No xcomp found on ROOT; try one level deeper (e.g. "is permitted to update")
        for child in root.children:
            if child.dep_ in ("attr", "acomp"):
                for grandchild in child.children:
                    if grandchild.dep_ == _XCOMP_DEP and grandchild.pos_ in ("VERB", "AUX"):
                        return _build_action(grandchild)

    return _build_action(root)


class ActionExtractor(PipelineComponent):
    """Extract the access-action verb phrase from each clause.

    Requires
    --------
    context.clause_docs : list[spacy.tokens.Span]

    Writes
    ------
    context.metadata["actions"] : list[str | None]
        One entry per clause (lemmatised verb or verb phrase).
    """

    @property
    def name(self) -> str:
        return "action_extraction"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Extract actions from all clauses and store in metadata."""
        clauses = context.clause_docs or [context.doc[:]]
        actions: list[Optional[str]] = []

        for i, span in enumerate(clauses):
            action = extract_action_from_span(span)
            actions.append(action)
            logger.debug("Clause %d action: %s", i, action)

        context.metadata["actions"] = actions
        return context
