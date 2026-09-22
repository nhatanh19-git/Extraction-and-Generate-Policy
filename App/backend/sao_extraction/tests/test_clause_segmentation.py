"""Unit tests for ClauseSegmenter component.

Tests are independent of I/O: a real spaCy model is loaded once per session
and injected into PipelineContext via a module-scoped fixture.
"""

from __future__ import annotations

import pytest
import spacy

from backend.sao_extraction.components.clause_segmentation import ClauseSegmenter
from backend.sao_extraction.core.schemas import PipelineContext


@pytest.fixture(scope="module")
def nlp():
    """Load spaCy model once for the entire test module."""
    return spacy.load("en_core_web_sm")


def make_context(nlp, text: str) -> PipelineContext:
    """Helper: create a minimal PipelineContext from raw text."""
    doc = nlp(text)
    return PipelineContext(original_text=text, doc=doc)


class TestClauseSegmenter:
    """Unit tests for ClauseSegmenter.process()."""

    def test_single_clause_returns_one_span(self, nlp):
        """Simple sentence with no coordination → exactly one clause span."""
        ctx = make_context(nlp, "Doctors may view patient records.")
        result = ClauseSegmenter().process(ctx)
        assert len(result.clause_docs) == 1
        assert "view" in result.clause_docs[0].text.lower() or "records" in result.clause_docs[0].text.lower()

    def test_coordinated_clauses_split_into_two(self, nlp):
        """Sentence with two coordinated main clauses → two spans."""
        ctx = make_context(nlp, "Doctors may view records and nurses may update them.")
        result = ClauseSegmenter().process(ctx)
        # We expect 2 clause spans
        assert len(result.clause_docs) >= 2, (
            f"Expected >=2 clauses, got {len(result.clause_docs)}: "
            f"{[s.text for s in result.clause_docs]}"
        )

    def test_passive_single_clause(self, nlp):
        """Passive single clause → one span."""
        ctx = make_context(nlp, "Patient records can be accessed by nurses.")
        result = ClauseSegmenter().process(ctx)
        assert len(result.clause_docs) == 1

    def test_negation_single_clause(self, nlp):
        """Negated single clause → one span."""
        ctx = make_context(nlp, "Guests cannot access restricted files.")
        result = ClauseSegmenter().process(ctx)
        assert len(result.clause_docs) == 1

    def test_relcl_clause_not_treated_as_independent(self, nlp):
        """A relative clause should NOT create an extra independent-clause span."""
        ctx = make_context(nlp, "Users who are authenticated can access the database.")
        result = ClauseSegmenter().process(ctx)
        # The relative clause is subordinate — only one independent clause expected
        assert len(result.clause_docs) == 1

    def test_empty_clause_docs_fallback(self, nlp):
        """Malformed or very short input → at least one span (full doc fallback)."""
        ctx = make_context(nlp, "OK.")
        result = ClauseSegmenter().process(ctx)
        assert len(result.clause_docs) >= 1
