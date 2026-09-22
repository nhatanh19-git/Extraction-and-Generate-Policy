"""Subject extraction component.

[EXTENDED] – Extracts the Subject noun phrase from each clause using
spaCy dependency relations: ``nsubj`` (active voice) or the ``agent``
prepositional phrase (passive voice). No domain wordlists hardcoded here.

[BASELINE: Alohaly et al. 2019] – Subject identification mirrors the role
of "subject" as described in the paper's attribute extraction pipeline,
where subject entities are candidates for subject-role attribute classification.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..core.interfaces import PipelineComponent
from ..core.schemas import PipelineContext

logger = logging.getLogger(__name__)

# Structural dependency constants — not domain knowledge
_SUBJ_DEPS = frozenset({"nsubj", "nsubjpass"})
_RELCL_DEP = "relcl"
_AGENT_DEP = "agent"
_POBJ_DEP = "pobj"


def expand_noun_phrase(token: Any) -> str:
    """Return the full noun-phrase text rooted at *token*.

    Collects the token's subtree, excluding relative clause (``relcl``)
    modifiers, which are handled separately by AttributeConstraintExtractor.

    Parameters
    ----------
    token : spacy.tokens.Token
        Head of the noun phrase (typically the nsubj token).

    Returns
    -------
    str
        The NP text with original casing and spacing.
    """
    # Gather subtree indices, excluding relcl subtrees
    relcl_roots: set[int] = {
        child.i
        for child in token.children
        if child.dep_ == _RELCL_DEP
    }
    # Expand each relcl to exclude its whole subtree
    excluded: set[int] = set()
    for relcl_root_i in relcl_roots:
        # Retrieve the token object from the doc
        relcl_token = token.doc[relcl_root_i]
        for t in relcl_token.subtree:
            excluded.add(t.i)

    kept = sorted(
        [t for t in token.subtree if t.i not in excluded],
        key=lambda t: t.i,
    )
    if not kept:
        return token.text

    return token.doc[kept[0].i : kept[-1].i + 1].text


def extract_subject_from_span(
    clause_span: Any, voice: str
) -> Optional[str]:
    """Extract the Subject from a single clause span.

    For **active** voice:
        Finds the token with ``dep_ == "nsubj"`` among the clause tokens
        and expands its noun phrase.

    For **passive** voice:
        First attempts ``nsubjpass`` (the grammatical subject of the passive,
        e.g. "records" in "Records are accessed by doctors" — this is the
        *object* semantically, so we skip it for Subject and look for the
        ``agent`` instead).
        Then looks for ``prep → agent → pobj`` pattern to find the by-agent.

    Parameters
    ----------
    clause_span : spacy.tokens.Span
    voice : "active" | "passive"

    Returns
    -------
    str | None
        Expanded NP text, or None if no subject found.
    """
    # Collect clause token indices for boundary checking
    clause_indices = {t.i for t in clause_span}

    if voice == "active":
        for token in clause_span:
            if token.dep_ == "nsubj" and token.i in clause_indices:
                return expand_noun_phrase(token)
        return None

    # Passive: look for agent prep phrase ("by <NP>")
    for token in clause_span:
        if token.dep_ == _AGENT_DEP:
            # agent's children include pobj
            for child in token.children:
                if child.dep_ == _POBJ_DEP:
                    return expand_noun_phrase(child)
        # Also handle direct nsubjpass for cases where the by-phrase is absent
        # but we still want to know the grammatical subject (for downstream)
        # We keep this separate from the main subject field.

    # Last resort: check nsubj even in passive (e.g. modal constructions)
    for token in clause_span:
        if token.dep_ == "nsubj" and token.i in clause_indices:
            return expand_noun_phrase(token)

    return None


class SubjectExtractor(PipelineComponent):
    """Extract Subject noun phrases from each clause.

    Requires
    --------
    context.clause_docs : list[spacy.tokens.Span]
    context.metadata["clause_meta"] : list[dict] with ``voice`` keys.

    Writes
    ------
    context.metadata["subjects"] : list[str | None]
        One entry per clause.
    """

    @property
    def name(self) -> str:
        return "subject_extraction"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Extract subjects from all clauses and store in metadata."""
        clauses = context.clause_docs or [context.doc[:]]
        clause_meta = context.metadata.get("clause_meta", [{}] * len(clauses))

        subjects: list[Optional[str]] = []
        for i, span in enumerate(clauses):
            voice = clause_meta[i].get("voice", "active") if i < len(clause_meta) else "active"
            subj = extract_subject_from_span(span, voice)
            subjects.append(subj)
            logger.debug("Clause %d subject: %s", i, subj)

        context.metadata["subjects"] = subjects
        return context
