"""ABAC Pipeline Core module."""

from .abac_schemas import (
    ABACEntity,
    ConstraintSpan,
    ConstraintAnalysis,
    CanonicalCondition,
    ABACCanonical,
    StructuredExtractionResult
)
from .abac_rule import CanonicalABACRule
from .normalizer import CanonicalOperator, normalize_operator
from .preprocessor import NLPPreprocessor, NLPResult, TokenInfo, SentenceInfo

__all__ = [
    "ABACEntity",
    "ConstraintSpan",
    "ConstraintAnalysis",
    "CanonicalCondition",
    "ABACCanonical",
    "StructuredExtractionResult",
    "CanonicalABACRule",
    "CanonicalOperator",
    "normalize_operator",
    "NLPPreprocessor",
    "NLPResult",
    "TokenInfo",
    "SentenceInfo",
]
