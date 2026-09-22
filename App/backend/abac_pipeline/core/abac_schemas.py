"""Structured ABAC dataclasses for the NLP Pipeline."""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass(frozen=True)
class ABACEntity:
    """Represents a basic extracted entity (Subject/Action/Object/Environment)."""
    text: str
    lemma: str
    span_start: int
    span_end: int
    type: str  # e.g., "SUBJECT", "ACTION", "OBJECT", "ENVIRONMENT"
    confidence: float = 1.0


@dataclass(frozen=True)
class ConstraintAnalysis:
    """Semantic analysis of a constraint span."""
    relation: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstraintSpan:
    """Represents an extracted constraint clause."""
    text: str
    start: int
    end: int
    semantic_relation: Optional[str] = None
    semantic_analysis: Optional[ConstraintAnalysis] = None
    canonical: Optional[Any] = None  # Will hold CanonicalCondition later


@dataclass(frozen=True)
class CanonicalCondition:
    """Canonical form of a condition: Attribute Operator Value(s)."""
    attribute: str
    operator: str  # e.g., "IN", "EQUALS", "CONTAINS"
    values: List[str] = field(default_factory=list)
    attribute_ref: Optional[str] = None  # For cross-entity checks, e.g., 'resource.department'


@dataclass(frozen=True)
class ABACCanonical:
    """The canonical semantic representation of a policy."""
    subject_conditions: List[CanonicalCondition] = field(default_factory=list)
    resource_conditions: List[CanonicalCondition] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    constraints: List[CanonicalCondition] = field(default_factory=list)
    effect: str = "Permit"


@dataclass(frozen=True)
class StructuredExtractionResult:
    """The final composite output of the NLP pipeline."""
    id: str
    original_sentence: str
    entities: Dict[str, List[ABACEntity]] = field(default_factory=dict)
    constraint_spans: List[ConstraintSpan] = field(default_factory=list)
    canonical: Optional[ABACCanonical] = None
    rule: Optional[Any] = None  # Will hold CanonicalABACRule

    def to_dict(self) -> Dict[str, Any]:
        """Convert to standard JSON dictionary."""
        result = asdict(self)
        if self.rule and hasattr(self.rule, 'to_dict'):
            result['rule'] = self.rule.to_dict()
        return result
