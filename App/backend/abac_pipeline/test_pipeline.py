"""End-to-end pipeline test.
Runs the full pipeline on sample ABAC policy sentences and prints structured outputs.
"""

import sys
import json
import logging
from pathlib import Path

# Ensure backend module can be imported when running standalone
base_dir = Path(__file__).resolve().parent.parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from backend.abac_pipeline.pipeline import ABACPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Test cases: (sentence, expected_effect, expected_subjects, expected_actions)
TEST_CASES = [
    {
        "sentence": "A nurse can add an item in a HR for a patient in the ward in which he/she works.",
        "expected_effect": "Permit",
        "expected_subject_contains": "nurse",
        "expected_action_contains": "add",
    },
    {
        "sentence": "Only members of the sales department can send invoices.",
        "expected_effect": "Permit",
        "expected_subject_contains": "member",
        "expected_action_contains": "send",
    },
    {
        "sentence": "Application admins can view documents that are not confidential.",
        "expected_effect": "Permit",
        "expected_subject_contains": "admin",
        "expected_action_contains": "view",
    },
    {
        "sentence": "A supervisor can read documents sent by their supervisees.",
        "expected_effect": "Permit",
        "expected_subject_contains": "supervisor",
        "expected_action_contains": "read",
    },
    {
        "sentence": "Technicians can only view and complete tasks that are assigned to them.",
        "expected_effect": "Permit",
        "expected_subject_contains": "technician",
        "expected_action_contains": "view",
    },
    {
        "sentence": "Users cannot access restricted files.",
        "expected_effect": "Deny",
        "expected_subject_contains": "user",
        "expected_action_contains": "access",
    },
]


def run_tests():
    logger.info("Initializing ABACPipeline...")
    pipeline = ABACPipeline()
    
    total = len(TEST_CASES)
    passed = 0
    
    print("\n" + "=" * 80)
    print("ABAC Pipeline End-to-End Test")
    print("=" * 80)
    
    for i, tc in enumerate(TEST_CASES, 1):
        sentence = tc["sentence"]
        print(f"\n--- Test {i}/{total} ---")
        print(f"Input: {sentence}")
        
        result = pipeline.process_policy(sentence, rule_id=f"test_{i}")
        
        # Print entities
        print(f"\nEntities:")
        for ent_type, ent_list in result.entities.items():
            for ent in ent_list:
                print(f"  [{ent_type}] '{ent.text}' (span: {ent.span_start}-{ent.span_end})")
        
        # Print constraints
        print(f"\nConstraints:")
        if result.constraint_spans:
            for cs in result.constraint_spans:
                print(f"  '{cs.text}' (span: {cs.start}-{cs.end})")
        else:
            print("  (none detected)")
        
        # Print rule
        print(f"\nGenerated Rule:")
        rule = result.rule
        print(f"  Effect: {rule.effect}")
        print(f"  Subjects: {[{'attr': s.attribute, 'op': s.operator, 'val': s.values} for s in rule.subjects]}")
        print(f"  Resources: {[{'attr': r.attribute, 'op': r.operator, 'val': r.values} for r in rule.resources]}")
        print(f"  Actions: {rule.actions}")
        print(f"  Constraints: {[{'attr': c.attribute, 'op': c.operator, 'val': c.values, 'ref': c.attribute_ref} for c in rule.constraints]}")
        
        # Validate
        errors = []
        
        if rule.effect != tc["expected_effect"]:
            errors.append(f"Effect: expected '{tc['expected_effect']}', got '{rule.effect}'")
            
        # Check subjects contain expected substring
        all_subj_text = " ".join([s.values[0] if s.values else "" for s in rule.subjects]).lower()
        all_entity_text = " ".join([ent.text for ents in result.entities.values() for ent in ents]).lower()
        if tc["expected_subject_contains"].lower() not in all_entity_text:
            errors.append(f"Entities missing expected subject containing '{tc['expected_subject_contains']}'")
            
        # Check actions contain expected substring
        all_action_text = " ".join(rule.actions).lower()
        if tc["expected_action_contains"].lower() not in all_action_text:
            # Also check entity actions
            action_entities = " ".join([ent.text for ent in result.entities.get("ACTION", [])]).lower()
            if tc["expected_action_contains"].lower() not in action_entities:
                errors.append(f"Actions missing expected action containing '{tc['expected_action_contains']}'")
        
        if errors:
            print(f"\n  [FAIL]: {'; '.join(errors)}")
        else:
            print(f"\n  [PASS]")
            passed += 1
    
    print("\n" + "=" * 80)
    print(f"Results: {passed}/{total} passed ({passed/total*100:.0f}%)")
    print("=" * 80)
    
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
