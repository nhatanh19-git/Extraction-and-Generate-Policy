"""Canonical operator normalization for ABAC policy conditions.

Maps natural language expressions and ABAC-Lab syntax operators to a small
canonical set: IN, EQUALS, CONTAINS, SUPERSET_EQ, NOT_IN, NOT_EQUALS.

This module is the single source of truth for operator semantics in the pipeline.
"""

from enum import Enum
from typing import Optional


class CanonicalOperator(str, Enum):
    """The canonical set of ABAC condition operators."""
    IN = "IN"                       # subject value is IN the allowed set
    EQUALS = "EQUALS"               # subject value equals resource value
    CONTAINS = "CONTAINS"           # subject set contains (is superset of) resource value
    SUPERSET_EQ = "SUPERSET_EQ"     # subject set is superset-or-equal of resource set
    NOT_IN = "NOT_IN"               # negation of IN
    NOT_EQUALS = "NOT_EQUALS"       # negation of EQUALS

    def __str__(self) -> str:
        return self.value


# ---- ABAC-Lab syntax operator mapping ----
# In ABAC-Lab .abac files, operators are single characters:
#   [ means "is a member of" / "is a subset of"  -> IN
#   ] means "contains" / "is a superset of"      -> CONTAINS
#   > means "is a superset-or-equal of"           -> SUPERSET_EQ
#   = means "equals"                               -> EQUALS
_ABAC_LAB_OPERATOR_MAP = {
    "[": CanonicalOperator.IN,
    "]": CanonicalOperator.CONTAINS,
    ">": CanonicalOperator.SUPERSET_EQ,
    "=": CanonicalOperator.EQUALS,
}


# ---- Natural language patterns ----
# Maps common NLP-extracted phrases to canonical operators.
# Ordered by specificity (longest match first when applicable).
_NL_OPERATOR_PATTERNS = {
    # Membership / Subset
    "member of": CanonicalOperator.IN,
    "belongs to": CanonicalOperator.IN,
    "belong to": CanonicalOperator.IN,
    "is a": CanonicalOperator.IN,
    "is an": CanonicalOperator.IN,
    "is in": CanonicalOperator.IN,
    "are in": CanonicalOperator.IN,
    "in the": CanonicalOperator.IN,
    "one of": CanonicalOperator.IN,
    "among": CanonicalOperator.IN,
    "within": CanonicalOperator.IN,
    "assigned to": CanonicalOperator.IN,
    "working on": CanonicalOperator.IN,
    "works in": CanonicalOperator.IN,
    "works on": CanonicalOperator.IN,

    # Equality
    "same as": CanonicalOperator.EQUALS,
    "equal to": CanonicalOperator.EQUALS,
    "equals": CanonicalOperator.EQUALS,
    "matches": CanonicalOperator.EQUALS,
    "is the same": CanonicalOperator.EQUALS,
    "identical to": CanonicalOperator.EQUALS,

    # Superset / Contains
    "includes": CanonicalOperator.CONTAINS,
    "include": CanonicalOperator.CONTAINS,
    "contains": CanonicalOperator.CONTAINS,
    "contain": CanonicalOperator.CONTAINS,
    "has": CanonicalOperator.CONTAINS,
    "have": CanonicalOperator.CONTAINS,
    "covers": CanonicalOperator.CONTAINS,

    # Negation
    "not in": CanonicalOperator.NOT_IN,
    "not a member of": CanonicalOperator.NOT_IN,
    "does not belong": CanonicalOperator.NOT_IN,
    "not equal to": CanonicalOperator.NOT_EQUALS,
    "different from": CanonicalOperator.NOT_EQUALS,
    "not the same": CanonicalOperator.NOT_EQUALS,
}


def normalize_operator_from_abac_lab(symbol: str) -> CanonicalOperator:
    """Convert an ABAC-Lab syntax operator symbol to canonical form.
    
    Args:
        symbol: One of '[', ']', '>', '='
        
    Returns:
        The corresponding CanonicalOperator.
        
    Raises:
        ValueError: If the symbol is not a recognized ABAC-Lab operator.
    """
    op = _ABAC_LAB_OPERATOR_MAP.get(symbol.strip())
    if op is None:
        raise ValueError(
            f"Unknown ABAC-Lab operator: '{symbol}'. "
            f"Expected one of: {list(_ABAC_LAB_OPERATOR_MAP.keys())}"
        )
    return op


def normalize_operator_from_nl(text: str) -> Optional[CanonicalOperator]:
    """Infer a canonical operator from a natural language phrase.
    
    Performs longest-match-first search over known NL patterns.
    
    Args:
        text: A natural language string, e.g., "is a member of".
        
    Returns:
        The matched CanonicalOperator, or None if no pattern matches.
    """
    text_lower = text.lower().strip()
    
    # Sort by length descending to prefer longer (more specific) matches
    for pattern in sorted(_NL_OPERATOR_PATTERNS, key=len, reverse=True):
        if pattern in text_lower:
            return _NL_OPERATOR_PATTERNS[pattern]
    
    return None


def normalize_operator(raw: str) -> CanonicalOperator:
    """Universal operator normalizer.
    
    Tries ABAC-Lab syntax first, then NL patterns, then defaults to IN.
    
    Args:
        raw: A raw operator string from any source.
        
    Returns:
        The best-matching CanonicalOperator.
    """
    stripped = raw.strip()
    
    # 1. Try exact ABAC-Lab symbol match
    if stripped in _ABAC_LAB_OPERATOR_MAP:
        return _ABAC_LAB_OPERATOR_MAP[stripped]
    
    # 2. Try canonical name directly (e.g., "IN", "EQUALS")
    try:
        return CanonicalOperator(stripped.upper())
    except ValueError:
        pass
    
    # 3. Try NL pattern matching
    nl_result = normalize_operator_from_nl(stripped)
    if nl_result is not None:
        return nl_result
    
    # 4. Default fallback
    return CanonicalOperator.IN
