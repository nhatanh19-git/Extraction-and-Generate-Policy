"""Unit tests for ActionExtractor component."""

from __future__ import annotations

import pytest
import spacy

from backend.sao_extraction.components.clause_segmentation import ClauseSegmenter
from backend.sao_extraction.components.action_extraction import ActionExtractor
from backend.sao_extraction.core.schemas import PipelineContext


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


def run_action(nlp, text: str):
    """Run segmentation → action extraction and return actions list."""
    doc = nlp(text)
    ctx = PipelineContext(original_text=text, doc=doc)
    ctx = ClauseSegmenter().process(ctx)
    ctx = ActionExtractor().process(ctx)
    return ctx.metadata.get("actions", [])


class TestActionExtractor:
    def test_simple_modal_active(self, nlp):
        """'may view' → action should be 'view'."""
        actions = run_action(nlp, "Doctors may view patient records.")
        assert actions[0] is not None
        assert "view" in actions[0]

    def test_xcomp_after_permitted(self, nlp):
        """'are permitted to update' → xcomp should give 'update'."""
        actions = run_action(nlp, "Authorized staff are permitted to update test results.")
        assert actions[0] is not None
        assert "update" in actions[0]

    def test_xcomp_after_allowed(self, nlp):
        """'is allowed to access' → action should be 'access'."""
        actions = run_action(nlp, "The nurse is allowed to access patient files.")
        assert actions[0] is not None
        assert "access" in actions[0]

    def test_negated_action(self, nlp):
        """Negation should not affect the extracted action lemma."""
        actions = run_action(nlp, "Guests cannot access restricted files.")
        assert actions[0] is not None
        assert "access" in actions[0]

    def test_can_review(self, nlp):
        """'can review' → action should be 'review'."""
        actions = run_action(nlp, "The physician can review patient files.")
        assert actions[0] is not None
        assert "review" in actions[0]
