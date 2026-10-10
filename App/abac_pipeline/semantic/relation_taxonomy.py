"""Taxonomy of Semantic Relations for Constraints.
DEPRECATED: Use core.normalizer.CanonicalOperator instead.
This file is kept for backward compatibility during the refactoring.
"""

from ..core.normalizer import CanonicalOperator as SemanticOperator

# Alias it for backward compatibility
__all__ = ["SemanticOperator"]
