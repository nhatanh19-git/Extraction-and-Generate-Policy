"""Core interfaces for the SAO extraction pipeline."""

from abc import ABC, abstractmethod
from typing import Any
from .schemas import PipelineResult, PipelineContext

class PipelineComponent(ABC):
    """Interface chung cho mọi bước xử lý trong pipeline (clause split, subject
    extraction, v.v.). Mỗi component nhận context hiện tại, trả về context đã
    được bổ sung — không side-effect ra ngoài, không I/O trực tiếp."""

    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """Process the context and return the updated context."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the pipeline component."""
        ...


class GazetteerProvider(ABC):
    """Interface cho nguồn tri thức từ vựng miền (effect terms, roles,
    resource types...). Cho phép thay nguồn (file/DB/ontology/online) không ảnh
    hưởng logic gọi."""

    @abstractmethod
    def match(self, span: Any, category: str) -> list[str]:
        """Check if a spaCy Span matches a category in the gazetteer."""
        ...

    @abstractmethod
    def load(self, source: str) -> None:
        """Load gazetteer data from a source (file path, DB connection string, URL)."""
        ...


class SAOExtractorStrategy(ABC):
    """Interface cho 1 chiến lược trích xuất SAO hoàn chỉnh (rule-based,
    sau này: hybrid CNN, SRL, LLM). Pipeline chính có thể chạy nhiều strategy
    song song và merge kết quả qua confidence score."""

    @abstractmethod
    def extract(self, text: str) -> PipelineResult:
        """Extract SAO triplets from a given policy text."""
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Name of the extraction strategy."""
        ...
