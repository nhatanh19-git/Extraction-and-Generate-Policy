"""ABAC Policy Data Model for policy generation."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Policy:
    subject: str
    action: str
    object: str
    effect: str = "Permit"
    environment: Optional[str] = None

    def __post_init__(self):
        if self.effect not in ("Permit", "Deny"):
            raise ValueError(f"Effect must be 'Permit' or 'Deny', got '{self.effect}'")
