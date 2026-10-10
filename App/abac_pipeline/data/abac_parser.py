"""Parser for ABAC-Lab .abac dataset files."""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path


@dataclass
class ABACPolicyRule:
    """Represents a rule parsed from a .abac file."""
    rule_id: str
    description: str
    subject_condition: str
    resource_condition: str
    actions: List[str]
    constraint_condition: str
    raw_rule_text: str


class ABACDatasetParser:
    """Parses ABAC-Lab dataset files (.abac)."""

    def __init__(self):
        # rule(subj_cond ; res_cond ; {actions} ; constraint)
        # Note: sometimes constraint is missing, so it's optional, and conditions might be empty
        self.rule_pattern = re.compile(r'rule\((.*?);(.*?);(.*?);?(.*?)\)')
        
    def parse_file(self, file_path: str | Path) -> List[ABACPolicyRule]:
        """Parse an .abac file and return the list of rules with their descriptions."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        content = path.read_text(encoding='utf-8')
        lines = content.splitlines()
        
        rules = []
        current_description = []
        rule_counter = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('#'):
                # Heuristic: if a comment starts with a number (e.g., "# 1. A user..."),
                # it's likely a rule description.
                comment_text = line[1:].strip()
                if re.match(r'^\d+\.', comment_text):
                    # Start of a new rule description
                    current_description = [comment_text]
                elif current_description:
                    # Continuation of description
                    current_description.append(comment_text)
            
            elif line.startswith('rule('):
                # We found a rule definition
                match = self.rule_pattern.match(line)
                if match:
                    subj = match.group(1).strip()
                    res = match.group(2).strip()
                    actions_raw = match.group(3).strip()
                    constraint = match.group(4).strip() if match.group(4) else ""
                    
                    # Clean up actions: "{read write}" -> ["read", "write"]
                    actions = [a.strip() for a in actions_raw.strip('{}').split()]
                    
                    desc_text = " ".join(current_description) if current_description else f"Rule {rule_counter}"
                    # Strip the leading number from description if present
                    desc_text = re.sub(r'^\d+\.\s*', '', desc_text)
                    
                    rule = ABACPolicyRule(
                        rule_id=f"{path.stem}_{rule_counter}",
                        description=desc_text,
                        subject_condition=subj,
                        resource_condition=res,
                        actions=actions,
                        constraint_condition=constraint,
                        raw_rule_text=line
                    )
                    rules.append(rule)
                    rule_counter += 1
                    current_description = []  # Reset for next rule
                    
        return rules

