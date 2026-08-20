"""Build ABAC policies from extraction results."""

from __future__ import annotations

import logging

from .extractor import ExtractionResult
from .policy_model import Policy

logger = logging.getLogger(__name__)


def generate_policy(attributes: ExtractionResult, effect: str = "Permit") -> Policy:
    """Create a policy or raise a clear validation error for incomplete extraction."""
    missing = [
        name
        for name, value in (("Subject", attributes.subject), ("Action", attributes.action), ("Object", attributes.object))
        if value is None or not value.text.strip()
    ]
    if missing:
        raise ValueError(f"Không thể xác định {', '.join(missing)}.")

    policy = Policy(attributes.subject.text, attributes.action.text, attributes.object.text, effect)
    logger.info("Policy generated")
    return policy