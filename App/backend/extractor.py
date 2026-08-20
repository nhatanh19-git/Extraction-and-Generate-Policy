"""Subject, action, and object extraction for Vietnamese ACP sentences."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .nlp_engine import NLPDocument

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedAttribute:
    text: str
    confidence: str
    source: str


@dataclass(frozen=True)
class ExtractionResult:
    subject: ExtractedAttribute | None
    action: ExtractedAttribute | None
    object: ExtractedAttribute | None

    def as_dict(self) -> dict[str, dict[str, str] | None]:
        return {
            name: ({"text": value.text, "confidence": value.confidence} if value else None)
            for name, value in (("subject", self.subject), ("action", self.action), ("object", self.object))
        }


class AttributeExtractor:
    """Extract access-control roles, actions, and resource phrases."""

    _permission_markers = re.compile(
        r"\b(?:có thể|được phép|được quyền|có quyền|được cho phép)\b",
        re.IGNORECASE,
    )
    _actions = (
        "tải xuống", "đăng nhập", "truy cập", "cập nhật", "xem", "đọc", "sửa", "xóa", "xoá", "tạo", "tải",
    )
    _leading_noise = re.compile(r"^(?:và|thì)\s+", re.IGNORECASE)

    def extract(self, document: NLPDocument) -> ExtractionResult:
        """Extract attributes using structured rules and return partial results safely."""
        sentence = self._normalize(document.text)
        action_match = self._find_action(sentence)
        if not action_match:
            logger.warning("Action could not be extracted")
            return ExtractionResult(None, None, None)

        action_text, action_start, action_end = action_match
        marker_matches = list(self._permission_markers.finditer(sentence[:action_start]))
        subject_end = marker_matches[-1].start() if marker_matches else action_start
        subject_text = self._clean_phrase(sentence[:subject_end])
        object_text = self._clean_phrase(sentence[action_end:])

        subject = ExtractedAttribute(subject_text, "high", "rule-based phrase extraction") if subject_text else None
        action = ExtractedAttribute(action_text, "high", "access-action lexicon")
        object_attribute = ExtractedAttribute(object_text, "high", "post-action noun phrase") if object_text else None
        if subject:
            logger.info("Subject extracted")
        if object_attribute:
            logger.info("Object extracted")
        return ExtractionResult(subject, action, object_attribute)

    def _find_action(self, sentence: str) -> tuple[str, int, int] | None:
        for action in sorted(self._actions, key=len, reverse=True):
            match = re.search(rf"(?<!\w){re.escape(action)}(?!\w)", sentence, re.IGNORECASE)
            if match:
                return match.group(), match.start(), match.end()
        return None

    @staticmethod
    def _normalize(sentence: str) -> str:
        return re.sub(r"\s+", " ", sentence.strip())

    def _clean_phrase(self, phrase: str) -> str:
        phrase = phrase.strip(" \t.,;:!?()[]{}\"")
        phrase = self._permission_markers.sub("", phrase)
        phrase = self._leading_noise.sub("", phrase.strip())
        return phrase.strip(" \t.,;:!?()[]{}\"")