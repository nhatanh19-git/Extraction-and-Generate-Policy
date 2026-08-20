"""ABAC policy domain model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    """A minimal ABAC policy rule."""

    subject: str
    action: str
    object: str
    effect: str = "Permit"

    def __post_init__(self) -> None:
        if self.effect not in {"Permit", "Deny"}:
            raise ValueError("Effect must be Permit or Deny")
        if not all((self.subject.strip(), self.action.strip(), self.object.strip())):
            raise ValueError("Subject, action, and object must not be empty")