"""Structured JSON Builder and Validator."""

import json
from typing import Dict, Any, List
from ..core.abac_schemas import NLPResult
from ..core.abac_rule import ABACRule

class JSONBuilder:
    """Builds and validates structured JSON output from NLP results."""
    
    def __init__(self, schema_dict: Dict[str, List[str]] = None):
        """Optionally accepts a target schema to validate against."""
        self.schema_dict = schema_dict
        
    def build(self, result: NLPResult, rule_id: str = "rule_1") -> Dict[str, Any]:
        """Convert NLPResult into a structured dictionary."""
        # We can construct an ABACRule from NLPResult
        rule = ABACRule(rule_id=rule_id)
        
        # Populate fields
        rule.effect = result.effect
        
        # Subjects
        if result.subjects:
            for s in result.subjects:
                if s.inferred_attribute:
                    rule.add_subject_attribute(s.inferred_attribute, s.lemma)
                else:
                    rule.add_subject_attribute("role", s.lemma) # Default fallback
                    
        # Resources
        if result.objects:
            for o in result.objects:
                if o.inferred_attribute:
                    rule.add_resource_attribute(o.inferred_attribute, o.lemma)
                else:
                    rule.add_resource_attribute("type", o.lemma) # Default fallback
                    
        # Actions
        if result.actions:
            for a in result.actions:
                rule.add_action(a.lemma)
                
        # Constraints
        if result.constraints:
            for c in result.constraints:
                # Add to both subject and resource constraints for now, 
                # or intelligently route based on attribute_ref if needed.
                rule.add_constraint(c)
                
        json_data = rule.to_dict()
        
        # Validate against schema if provided
        if self.schema_dict:
            self.validate(json_data)
            
        return json_data
        
    def validate(self, json_data: Dict[str, Any]) -> bool:
        """Validates that the keys in the JSON exist in the schema vocabulary."""
        if not self.schema_dict:
            return True
            
        is_valid = True
        
        # Validate Subject Attributes
        subject_vocab = self.schema_dict.get("subject_attributes", [])
        if subject_vocab:
            for k in json_data.get("subject", {}).keys():
                if k not in subject_vocab:
                    is_valid = False
                    
        # Validate Resource Attributes
        resource_vocab = self.schema_dict.get("resource_attributes", [])
        if resource_vocab:
            for k in json_data.get("resource", {}).keys():
                if k not in resource_vocab:
                    is_valid = False
                    
        # Validate Actions
        action_vocab = self.schema_dict.get("action_attributes", [])
        if action_vocab:
            for action in json_data.get("action", []):
                if action not in action_vocab:
                    is_valid = False
                    
        return is_valid
