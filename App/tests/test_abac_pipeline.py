"""Tests for ABAC Pipeline End-to-End."""

import pytest
from backend.abac_pipeline.pipeline import ABACPipeline

def test_pipeline_e2e():
    """Test full pipeline extraction."""
    pipeline = ABACPipeline()
    
    text = "A user can read a document if his department is the same as the document's department."
    result = pipeline.process_policy(text, rule_id="test_01")
    
    assert result is not None
    assert result.original_sentence == text
    assert result.rule is not None
    
    rule = result.rule
    assert rule.effect == "permit"
    assert "read" in rule.actions
    
    # Check subjects mapped to "role"
    assert len(rule.subjects) > 0
    assert rule.subjects[0].attribute == "role"
    
    # Check constraints
    assert len(rule.constraints) > 0
    constraint = rule.constraints[0]
    
    assert constraint.attribute == "department"
    assert constraint.operator == "EQUALS"
    
    print("Rule String:", rule.to_string())

if __name__ == "__main__":
    test_pipeline_e2e()
    print("All E2E tests passed.")
