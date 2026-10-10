"""Master Attribute Extractor — adapter bridging the new SAO pipeline to main.py.

This module maintains the **exact same public interface** as the original
``attribute_extractor.py`` so that ``main.py``, ``batch_processor.py``, and
existing tests continue to work without any modification.

Architecture
------------
Internally, ``AttributeExtractor.extract()`` now delegates to
``RuleBasedSAOExtractor`` (dependency-parsing pipeline) instead of the
former regex-based approach. The result is mapped from ``PipelineResult``
back to ``ExtractionResult`` via a thin adapter layer.

Adapter contract
----------------
- ``ExtractionResult.subject.text``  ← first ``SAOTriplet.subject``
- ``ExtractionResult.action.text``   ← first ``SAOTriplet.action``
- ``ExtractionResult.object.text``   ← first object in ``SAOTriplet.objects``
- ``ExtractionResult.environment``   ← first ``AttributeConstraint`` value
  (if any), otherwise empty string
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..sao_extraction.extractors.rule_based_extractor import RuleBasedSAOExtractor
from ..sao_extraction.core.schemas import PipelineResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses — unchanged from original interface
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractedAttribute:
    """A single extracted attribute with its raw text and metadata."""
    text: str
    confidence: str
    source: str


@dataclass(frozen=True)
class ExtractionResult:
    """Aggregated S/A/O extraction result (adapter-facing schema).

    Intentionally kept identical to the original ``ExtractionResult`` so
    that ``main.py`` and ``batch_processor.py`` require zero changes.
    """
    subject: ExtractedAttribute | None
    action: ExtractedAttribute | None
    object: ExtractedAttribute | None
    environment: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject.text if self.subject else None,
            "action": self.action.text if self.action else None,
            "object": self.object.text if self.object else None,
            "environment": self.environment,
            "raw_display": {
                "subject": self.subject.text if self.subject else "",
                "action": self.action.text if self.action else "",
                "object": self.object.text if self.object else "",
            },
        }


# ---------------------------------------------------------------------------
# Adapter mapping
# ---------------------------------------------------------------------------

def _pipeline_result_to_extraction_result(result: PipelineResult) -> ExtractionResult:
    """Map a ``PipelineResult`` to the legacy ``ExtractionResult`` interface.

    Uses the first triplet in the result (highest-confidence clause).
    If no triplets were extracted, returns an ExtractionResult with all None.

    Parameters
    ----------
    result : PipelineResult

    Returns
    -------
    ExtractionResult
    """
    if not result.triplets:
        return ExtractionResult(subject=None, action=None, object=None, environment="")

    # Use the triplet with the highest confidence
    best = max(result.triplets, key=lambda t: t.confidence)

    confidence_label = (
        "high" if best.confidence >= 0.8
        else "medium" if best.confidence >= 0.5
        else "low"
    )

    subject = (
        ExtractedAttribute(
            text=best.subject,
            confidence=confidence_label,
            source="rule_based",
        )
        if best.subject
        else None
    )

    action = ExtractedAttribute(
        text=best.action,
        confidence=confidence_label,
        source="rule_based",
    )

    # First object as the primary object text
    obj_text = best.objects[0] if best.objects else ""
    object_attr = (
        ExtractedAttribute(
            text=obj_text,
            confidence=confidence_label,
            source="rule_based",
        )
        if obj_text
        else None
    )

    # Environment: first attribute constraint value (if any)
    environment = ""
    if best.attribute_constraints:
        first = best.attribute_constraints[0]
        environment = f"{first.relation} {first.value}".strip()

    return ExtractionResult(
        subject=subject,
        action=action,
        object=object_attr,
        environment=environment,
    )


# ---------------------------------------------------------------------------
# Public API — unchanged from original
# ---------------------------------------------------------------------------

class AttributeExtractor:
    """Extract access-control roles, actions, and resource phrases.

    **Adapter class**: the public interface is identical to the original
    ``AttributeExtractor``. Internally delegates to ``RuleBasedSAOExtractor``
    which uses spaCy dependency parsing instead of regex/hardcoded wordlists.

    Usage
    -----
    extractor = AttributeExtractor()
    result = extractor.extract("The attending physician can review patient files.")
    print(result.subject.text)  # "attending physician"
    print(result.action.text)   # "review"
    print(result.object.text)   # "patient files"
    """

    def __init__(self) -> None:
        self._engine = RuleBasedSAOExtractor()

    def extract(self, document: Any) -> ExtractionResult:
        """Extract S/A/O from a policy sentence.

        Accepts either a plain ``str`` or a legacy ``NLPDocument`` (duck-typed
        via ``.text`` attribute) for backward compatibility with ``batch_processor.py``.

        Parameters
        ----------
        document : str | NLPDocument
            The input policy sentence.

        Returns
        -------
        ExtractionResult
        """
        # Duck-type: accept both str and legacy NLPDocument
        sentence: str = document.text if hasattr(document, "text") else str(document)

        if not sentence.strip():
            logger.warning("AttributeExtractor: empty input received.")
            return ExtractionResult(subject=None, action=None, object=None)

        try:
            pipeline_result: PipelineResult = self._engine.extract(sentence)
            return _pipeline_result_to_extraction_result(pipeline_result)
        except Exception as exc:  # noqa: BLE001
            logger.exception("AttributeExtractor: pipeline failed: %s", exc)
            return ExtractionResult(subject=None, action=None, object=None)

    def export_to_json(self, result: ExtractionResult, output_path: str | Path) -> None:
        """Save extraction result to a JSON file.

        Parameters
        ----------
        result : ExtractionResult
        output_path : str | Path
        """
        data = result.as_dict()
        Path(output_path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Saved extraction result to %s", output_path)
