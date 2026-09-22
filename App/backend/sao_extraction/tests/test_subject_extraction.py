"""Unit tests for SubjectExtractor component."""

from __future__ import annotations

import pytest
import spacy

from backend.sao_extraction.components.clause_segmentation import ClauseSegmenter
from backend.sao_extraction.components.voice_polarity import VoicePolarityDetector
from backend.sao_extraction.components.subject_extraction import SubjectExtractor
from backend.sao_extraction.core.schemas import PipelineContext
from backend.sao_extraction.gazetteers.provider import FileBasedGazetteer
from pathlib import Path

GAZETTEER_DIR = Path(__file__).parent.parent / "gazetteers" / "data"


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture(scope="module")
def gazetteer():
    g = FileBasedGazetteer()
    for f in GAZETTEER_DIR.glob("*.json"):
        g.load(str(f))
    return g


def run_subject(nlp, gazetteer, text: str):
    """Run segmentation → voice → subject extraction and return subjects list."""
    doc = nlp(text)
    ctx = PipelineContext(original_text=text, doc=doc)
    ctx = ClauseSegmenter().process(ctx)
    ctx = VoicePolarityDetector(gazetteer).process(ctx)
    ctx = SubjectExtractor().process(ctx)
    return ctx.metadata.get("subjects", [])


class TestSubjectExtractor:
    def test_simple_active_nsubj(self, nlp, gazetteer):
        """Active sentence: nsubj should be extracted."""
        subjects = run_subject(nlp, gazetteer, "Doctors may view patient records.")
        assert subjects[0] is not None
        assert "doctor" in subjects[0].lower() or "Doctors" in subjects[0]

    def test_multi_word_subject(self, nlp, gazetteer):
        """Multi-word NP subject: 'attending physician'."""
        subjects = run_subject(nlp, gazetteer, "The attending physician can review patient files.")
        assert subjects[0] is not None
        assert "physician" in subjects[0].lower()

    def test_passive_with_agent(self, nlp, gazetteer):
        """Passive voice: subject should come from by-agent phrase."""
        subjects = run_subject(nlp, gazetteer, "Records can be accessed by authorized nurses.")
        assert subjects[0] is not None
        assert "nurse" in subjects[0].lower() or "nurses" in subjects[0].lower()

    def test_negated_active_subject(self, nlp, gazetteer):
        """Negation does not affect subject extraction."""
        subjects = run_subject(nlp, gazetteer, "Guests cannot access restricted files.")
        assert subjects[0] is not None
        assert "guest" in subjects[0].lower() or "Guests" in subjects[0]

    def test_relcl_subject_head_only(self, nlp, gazetteer):
        """relcl modifier should not pollute the subject noun phrase."""
        subjects = run_subject(
            nlp, gazetteer, "Users who are authenticated can access the database."
        )
        assert subjects[0] is not None
        # "Users" is the subject; "who are authenticated" is the relcl constraint
        assert "users" in subjects[0].lower()
        # The relative clause text should NOT be part of the subject string
        assert "authenticated" not in subjects[0].lower()
