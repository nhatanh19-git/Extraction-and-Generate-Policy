"""Unit tests for GazetteerValidator component and FileBasedGazetteer."""

from __future__ import annotations

import pytest
import spacy
from pathlib import Path

from backend.sao_extraction.gazetteers.provider import FileBasedGazetteer
from backend.sao_extraction.components.gazetteer_validation import compute_triplet_confidence
from backend.sao_extraction.extractors.rule_based_extractor import RuleBasedSAOExtractor

GAZETTEER_DIR = Path(__file__).parent.parent / "gazetteers" / "data"


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture(scope="module")
def gazetteer():
    g = FileBasedGazetteer()
    for f in sorted(GAZETTEER_DIR.glob("*.json")):
        g.load(str(f))
    return g


@pytest.fixture(scope="module")
def extractor():
    return RuleBasedSAOExtractor()


class TestFileBasedGazetteer:
    def test_permit_category_loaded(self, gazetteer):
        """'permit' category must be present after loading effect_terms.json."""
        assert "permit" in gazetteer.categories()

    def test_deny_category_loaded(self, gazetteer):
        """'deny' category must be present after loading effect_terms.json."""
        assert "deny" in gazetteer.categories()

    def test_roles_category_loaded(self, gazetteer):
        """'roles' category must be present after loading subject_roles.json."""
        assert "roles" in gazetteer.categories()

    def test_resources_category_loaded(self, gazetteer):
        """'resources' category must be present after loading resource_types.json."""
        assert "resources" in gazetteer.categories()

    def test_match_lemma(self, nlp, gazetteer):
        """Token with lemma 'allow' should match 'permit' category."""
        doc = nlp("Users are allowed to access the system.")
        # 'allowed' → lemma 'allow' should match permit
        for token in doc:
            if token.text.lower() == "allowed":
                result = gazetteer.match(token, "permit")
                assert len(result) > 0, f"Expected 'allowed' to match permit, got {result}"
                break

    def test_no_match_for_unknown_word(self, nlp, gazetteer):
        """A random word should not match any domain category."""
        doc = nlp("The quick brown fox jumps.")
        for token in doc:
            if token.text.lower() == "fox":
                result = gazetteer.match(token, "roles")
                assert result == []
                break


class TestConfidenceScoring:
    def test_full_triplet_high_confidence(self, nlp, gazetteer):
        """A complete triplet with known role subject → confidence ≥ 0.8."""
        doc = nlp("Doctors may view patient records.")
        score = compute_triplet_confidence(
            subject="Doctors",
            action="view",
            objects=["patient records"],
            doc=doc,
            gazetteer=gazetteer,
        )
        assert score >= 0.8, f"Expected high confidence, got {score}"

    def test_missing_action_lowers_confidence(self, nlp, gazetteer):
        """Missing action should lower confidence significantly."""
        doc = nlp("Doctors view records.")
        score = compute_triplet_confidence(
            subject="Doctors",
            action=None,
            objects=["records"],
            doc=doc,
            gazetteer=gazetteer,
        )
        assert score < 0.6, f"Expected lower confidence with no action, got {score}"

    def test_missing_subject_lowers_confidence(self, nlp, gazetteer):
        """Missing subject should lower confidence."""
        doc = nlp("Access the database.")
        score = compute_triplet_confidence(
            subject=None,
            action="access",
            objects=["database"],
            doc=doc,
            gazetteer=gazetteer,
        )
        assert score < 0.85, f"Expected confidence < 0.85 for no subject, got {score}"


class TestEndToEndExtractor:
    """Integration tests running the full RuleBasedSAOExtractor pipeline."""

    def test_simple_active_sentence(self, extractor):
        result = extractor.extract("The attending physician can review patient files.")
        assert len(result.triplets) > 0
        t = result.triplets[0]
        assert t.action is not None
        assert "review" in t.action
        assert t.effect == "Permit"
        assert t.voice == "active"

    def test_passive_sentence(self, extractor):
        result = extractor.extract(
            "Authorized laboratory staff are permitted to update test results."
        )
        assert len(result.triplets) > 0
        t = result.triplets[0]
        assert "update" in t.action
        assert t.effect == "Permit"

    def test_negation_deny_effect(self, extractor):
        result = extractor.extract("Guests cannot access restricted files.")
        assert len(result.triplets) > 0
        t = result.triplets[0]
        assert t.effect == "Deny"
        assert "access" in t.action

    def test_empty_input(self, extractor):
        result = extractor.extract("")
        assert result.triplets == []
        assert len(result.warnings) > 0

    def test_pipeline_version_present(self, extractor):
        result = extractor.extract("Doctors may view records.")
        assert result.pipeline_version is not None
        assert len(result.pipeline_version) > 0
