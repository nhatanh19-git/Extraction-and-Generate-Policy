"""Canonical ABAC Rule generator models."""

from dataclasses import dataclass, field
from typing import List, Optional
from .abac_schemas import CanonicalCondition


@dataclass(frozen=True)
class CanonicalABACRule:
    """Deterministic, canonical representation of an ABAC Rule."""
    rule_id: str
    effect: str
    subjects: List[CanonicalCondition] = field(default_factory=list)
    objects: List[CanonicalCondition] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    environment: List[CanonicalCondition] = field(default_factory=list)
    constraints: List[CanonicalCondition] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert the rule to a dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def to_json(self) -> str:
        """Convert the rule to a JSON string."""
        import json
        return json.dumps(self.to_dict())

    def _fmt_val(self, c: CanonicalCondition) -> str:
        if c.values:
            return ",".join(c.values)
        return str(c.attribute_ref)

    def to_string(self) -> str:
        """Format the rule into a readable canonical syntax."""
        subj_str = ", ".join([f"S({c.attribute}, {c.operator}, {self._fmt_val(c)})" for c in self.subjects])
        obj_str = ", ".join([f"O({c.attribute}, {c.operator}, {self._fmt_val(c)})" for c in self.objects])
        act_str = f"A({', '.join(self.actions)})"
        
        parts = []
        if subj_str: parts.append(subj_str)
        if obj_str: parts.append(obj_str)
        if act_str: parts.append(act_str)
        
        if self.constraints:
            const_str = ", ".join([f"C({c.attribute}, {c.operator}, {self._fmt_val(c)})" for c in self.constraints])
            parts.append(const_str)
            
        return f"RULE({self.effect.upper()}: {'; '.join(parts)})"

    def to_abac_lab_syntax(self) -> str:
        """Format the rule into ABAC-Lab syntax: rule(subj_cond ; obj_cond ; {actions} ; constraint)."""
        def format_conds(conds: List[CanonicalCondition]) -> str:
            parts = []
            for c in conds:
                parts.append(f"{c.attribute}={self._fmt_val(c)}")
            return " ^ ".join(parts) if parts else ""

        subj_str = format_conds(self.subjects)
        obj_str = format_conds(self.objects)
        act_str = "{" + " ".join(self.actions) + "}"
        const_str = format_conds(self.constraints)
        
        return f"rule({subj_str} ; {obj_str} ; {act_str} ; {const_str})"
