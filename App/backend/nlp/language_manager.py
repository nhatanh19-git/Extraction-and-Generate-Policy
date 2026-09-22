"""Language configuration and selection manager for ABAC Policy Studio."""

from typing import List, Dict, Any


class LanguageManager:
    """Manages language options for policy extraction."""

    SUPPORTED_LANGUAGES = [
        {"code": "en", "name": "English (US) - Active Engine", "active": True},
        {"code": "vi", "name": "Tiếng Việt (Sắp cập nhật / Future Beta)", "active": False},
    ]

    def __init__(self, default_lang: str = "en"):
        self.current_lang = default_lang

    def get_language_options(self) -> List[str]:
        return [lang["name"] for lang in self.SUPPORTED_LANGUAGES]

    def set_language_by_index(self, index: int) -> Dict[str, Any]:
        if 0 <= index < len(self.SUPPORTED_LANGUAGES):
            selected = self.SUPPORTED_LANGUAGES[index]
            self.current_lang = selected["code"]
            return selected
        return self.SUPPORTED_LANGUAGES[0]

    def is_current_language_active(self) -> bool:
        return self.current_lang == "en"
