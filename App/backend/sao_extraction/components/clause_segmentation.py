"""Clause segmentation component.

[EXTENDED] – Splits a spaCy Doc into independent clauses so that each
clause can be processed independently by downstream extraction components.
Handles coordinating conjunctions (``and``, ``or``, ``but``) at the ROOT
verb level, which is the most common clause boundary in NLACP sentences.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.interfaces import PipelineComponent
from ..core.schemas import PipelineContext

logger = logging.getLogger(__name__)


def _get_clause_root_tokens(doc: Any) -> list[Any]:
    """Return the root token(s) of each independent clause in the doc.

    Strategy
    --------
    1. Find the sentence ROOT (the main finite verb).
    2. Walk the ROOT's ``conj`` children — each ``conj`` child that is a VERB
       constitutes the head of a coordinated independent clause.
    3. Return [ROOT] + [conj verbs], in token order.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        Parsed document.

    Returns
    -------
    list[Token]
        Root token for each identified clause, sorted by position.
    """
    roots = []
    for token in doc:
        if token.dep_ == "ROOT":
            roots.append(token)
            # Collect coordinated verb conjuncts at the same level
            for child in token.children:
                if child.dep_ == "conj" and child.pos_ in ("VERB", "AUX"):
                    roots.append(child)
    # Sort by position to preserve reading order
    return sorted(roots, key=lambda t: t.i)


def _span_for_clause_root(doc: Any, root_token: Any, all_roots: list[Any]) -> Any:
    """Return a spaCy Span covering the subtree of *root_token*.

    Boundaries are trimmed so that tokens already owned by a sibling root's
    subtree are not double-counted. Uses the document's sentence span as the
    outer boundary.

    Parameters
    ----------
    doc : spacy.tokens.Doc
    root_token : spacy.tokens.Token
        The clause root token.
    all_roots : list[Token]
        All clause root tokens (used to compute non-overlapping boundaries).

    Returns
    -------
    spacy.tokens.Span
        A Span object that can be treated as a mini-Doc by downstream
        components.
    """
    # Collect the full subtree token indices for this root
    subtree_indices = sorted(t.i for t in root_token.subtree)
    if not subtree_indices:
        return doc[root_token.i : root_token.i + 1]

    start = subtree_indices[0]
    end = subtree_indices[-1] + 1

    # Restrict to avoid including tokens in a sibling's subtree
    other_root_indices = {r.i for r in all_roots if r.i != root_token.i}
    other_subtrees: set[int] = set()
    for other_root in all_roots:
        if other_root.i != root_token.i:
            other_subtrees.update(t.i for t in other_root.subtree)

    # Keep only indices up to the first conflicting boundary
    # (tokens left of our root that are owned by another root are excluded)
    filtered = [
        i for i in subtree_indices
        if i not in other_subtrees or i == root_token.i
    ]
    if not filtered:
        return doc[root_token.i : root_token.i + 1]

    start = filtered[0]
    end = filtered[-1] + 1
    return doc[start:end]


class ClauseSegmenter(PipelineComponent):
    """Split a parsed Doc into independent-clause Spans.

    Writes
    ------
    context.clause_docs : list[spacy.tokens.Span]
        One Span per detected independent clause. If the sentence has no
        coordinating conjuncts at the ROOT level, ``clause_docs`` contains
        exactly one element: the full sentence span.

    The subsequent components (VoicePolarityDetector, SubjectExtractor, etc.)
    iterate over ``context.clause_docs`` rather than the full Doc, enabling
    correct extraction from sentences like:
        "Doctors may view records and nurses may update them."
    """

    @property
    def name(self) -> str:
        return "clause_segmentation"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Detect clause boundaries and populate ``context.clause_docs``."""
        doc = context.doc
        roots = _get_clause_root_tokens(doc)

        if not roots:
            # Fallback: treat the entire document as one clause
            logger.debug("ClauseSegmenter: no ROOT found, using full doc span.")
            context.clause_docs = [doc[:]]
            return context

        if len(roots) == 1:
            # Single-clause sentence — no segmentation needed
            context.clause_docs = [doc[:]]
            return context

        # Multi-clause: compute non-overlapping spans per root
        spans = [_span_for_clause_root(doc, r, roots) for r in roots]
        # Filter out empty spans
        spans = [s for s in spans if len(s) > 0]

        logger.debug(
            "ClauseSegmenter: found %d clause(s): %s",
            len(spans),
            [s.text for s in spans],
        )
        context.clause_docs = spans
        return context
