"""Policy Generator from extracted attribute results."""

from typing import Any
from .policy_model import Policy


def generate_policy(extraction_result: Any, effect: str = "Permit") -> Policy:
    """Generate ABAC Policy object using extracted raw attributes."""
    subject = extraction_result.subject.text if extraction_result.subject else ""
    action = extraction_result.action.text if extraction_result.action else ""
    object_attr = extraction_result.object.text if extraction_result.object else ""
    environment = getattr(extraction_result, "environment", "")

    if not subject or not action or not object_attr:
        missing = []
        if not subject: missing.append("Subject")
        if not action: missing.append("Action")
        if not object_attr: missing.append("Object")
        raise ValueError(f"Không thể sinh ABAC Policy vì thiếu: {', '.join(missing)}.")

    return Policy(
        subject=subject,
        action=action,
        object=object_attr,
        effect=effect,
        environment=environment,
    )
