"""Attribute constraint extraction component.

[EXTENDED] – Extracts additional contextual constraints from each clause
that are not part of the core S/A/O triplet. These include:
  - PP-modifiers (prepositional phrases) not already consumed as Object.
  - Relative clauses (``relcl``) modifying Subject or Object nouns.

Each constraint becomes an ``AttributeConstraint(relation, value)`` pair.
No domain wordlists are hardcoded here — all dependency checks are structural.

[BASELINE: Alohaly et al. 2019] – Attribute constraints correspond to the
environmental/conditional attribute-value pairs that the paper's CNN classifier
would receive as input features for ABAC policy classification.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..core.interfaces import PipelineComponent
from ..core.schemas import AttributeConstraint, PipelineContext

logger = logging.getLogger(__name__)

# Structural dependency constants
_PREP_DEP = "prep"
_POBJ_DEP = "pobj"
_RELCL_DEP = "relcl"

# Prepositions that are likely to introduce Object (already handled by ObjectExtractor)
# These are excluded here to avoid duplicating the object as an attribute constraint.
_OBJECT_PREPS = frozenset({"to", "of", "on", "for", "from", "into", "about"})

# Prepositions typically encoding environmental/temporal constraints
_CONSTRAINT_PREPS = frozenset({
    "during", "within", "at", "in", "by", "under", "between",
    "after", "before", "until", "with", "without", "through",
    "across", "over", "via", "per",
})


def _expand_span(token: Any) -> str:
    """Return the text of *token*'s subtree as a string."""
    subtree = sorted(token.subtree, key=lambda t: t.i)
    if not subtree:
        return token.text
    return token.doc[subtree[0].i : subtree[-1].i + 1].text


def extract_attribute_constraints(
    clause_span: Any,
    action_text: Optional[str] = None,
) -> list[AttributeConstraint]:
    """Extract PP-modifier and relative-clause constraints from a clause.

    Algorithm
    ---------
    1. For each token in the clause with ``dep_ == "prep"``:
       - Skip prepositions that were consumed by ObjectExtractor (object preps).
       - Prepositions in ``_CONSTRAINT_PREPS`` → ``AttributeConstraint(prep.text, pobj NP)``.
       - Other remaining preps → also captured to maximise recall.
    2. For each token with ``dep_ == "relcl"``:
       - Capture the full relative clause text as a constraint.

    Parameters
    ----------
    clause_span : spacy.tokens.Span
    action_text : str | None
        Not used in constraint extraction but available for future use
        (e.g. constraining the search to the action subtree).

    Returns
    -------
    list[AttributeConstraint]
    """
    constraints: list[AttributeConstraint] = []
    seen_texts: set[str] = set()

    # Scan all tokens in the clause
    for token in clause_span:
        # --- PP-modifier constraints ---
        if token.dep_ == _PREP_DEP:
            prep_lemma = token.lemma_.lower()
            # Skip if this prep is an object prep (handled by ObjectExtractor)
            if prep_lemma in _OBJECT_PREPS:
                continue
            # Collect pobj children
            for child in token.children:
                if child.dep_ == _POBJ_DEP:
                    value_text = _expand_span(child)
                    key = f"{prep_lemma}:{value_text}"
                    if key not in seen_texts:
                        seen_texts.add(key)
                        constraints.append(
                            AttributeConstraint(
                                relation=token.text.lower(),
                                value=value_text,
                            )
                        )

        # --- Relative clause constraints ---
        if token.dep_ == _RELCL_DEP:
            relcl_text = _expand_span(token)
            key = f"relcl:{relcl_text}"
            if key not in seen_texts:
                seen_texts.add(key)
                # Relation is "that" / "who" / "which" (the relativizer)
                relativizer = "that"
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass") and child.pos_ in ("PRON",):
                        relativizer = child.text.lower()
                        break
                constraints.append(
                    AttributeConstraint(relation=relativizer, value=relcl_text)
                )

    return constraints


class AttributeConstraintExtractor(PipelineComponent):
    """Extract PP-modifier and relcl attribute constraints from each clause.

    Requires
    --------
    context.clause_docs : list[spacy.tokens.Span]
    context.metadata["actions"] : list[str | None]  (optional, for future use)

    Writes
    ------
    context.metadata["attribute_constraints"] : list[list[AttributeConstraint]]
        One list of constraints per clause.
    """

    @property
    def name(self) -> str:
        return "attribute_attachment"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Extract attribute constraints from all clauses."""
        clauses = context.clause_docs or [context.doc[:]]
        actions = context.metadata.get("actions", [None] * len(clauses))

        all_constraints: list[list[AttributeConstraint]] = []
        for i, span in enumerate(clauses):
            action = actions[i] if i < len(actions) else None
            constraints = extract_attribute_constraints(span, action)
            all_constraints.append(constraints)
            logger.debug(
                "Clause %d constraints: %s",
                i,
                [(c.relation, c.value) for c in constraints],
            )

        context.metadata["attribute_constraints"] = all_constraints
        return context
