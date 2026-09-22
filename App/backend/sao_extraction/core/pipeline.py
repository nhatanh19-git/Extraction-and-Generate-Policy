"""Orchestrator for the SAO extraction pipeline.

[EXTENDED] – The SAOPipeline class is a project extension that sequences
NLP components into a dependency-parsing based extraction pipeline.
It is not part of Alohaly, Takabi & Blanco (2019).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import spacy
import yaml

from .interfaces import PipelineComponent
from .schemas import PipelineContext

logger = logging.getLogger(__name__)


class SAOPipeline:
    """Orchestrates the sequential execution of PipelineComponents.

    The pipeline owns the spaCy NLP model and passes a ``PipelineContext``
    through each registered component in insertion order. Components are
    enabled/disabled via the ``components`` section of ``pipeline_config.yaml``,
    supporting ablation studies without code changes.

    Usage
    -----
    # Build from config (recommended):
    pipeline = SAOPipeline.from_config("backend/sao_extraction/config/pipeline_config.yaml")

    # Or build manually:
    pipeline = SAOPipeline(model_name="en_core_web_sm")
    pipeline.add_component(PreprocessingComponent())
    context = pipeline.process("Users may access the database.")
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self.components: List[PipelineComponent] = []
        self._disabled_components: set[str] = set()
        self._pipeline_version: str = "1.0"
        self.nlp = self._load_spacy_model(model_name)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str | Path) -> "SAOPipeline":
        """Create a pipeline instance from a YAML configuration file.

        The config must contain:
        - ``version`` (str): pipeline version label.
        - ``model_name`` (str): spaCy model to load.
        - ``components`` (dict[str, bool]): map of component name → enabled.

        Components that are ``false`` in the config are registered as disabled
        and skipped during ``process()``.

        Parameters
        ----------
        config_path : str | Path
            Path to ``pipeline_config.yaml``.

        Returns
        -------
        SAOPipeline
            Configured pipeline instance (no components registered yet —
            the extractor strategy is responsible for adding components
            in the correct order).
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning("Pipeline config not found at %s; using defaults.", path)
            return cls()

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        model_name = cfg.get("model_name", "en_core_web_sm")
        instance = cls(model_name=model_name)
        instance._pipeline_version = str(cfg.get("version", "1.0"))

        # Record which components are disabled per config
        for name, enabled in cfg.get("components", {}).items():
            if not enabled:
                instance._disabled_components.add(name)
                logger.debug("Component '%s' is disabled in config.", name)

        return instance

    # ------------------------------------------------------------------
    # Component management
    # ------------------------------------------------------------------

    def add_component(self, component: PipelineComponent) -> None:
        """Append a component to the pipeline.

        Disabled components (per config) are silently skipped at runtime.
        """
        self.components.append(component)

    def is_enabled(self, component_name: str) -> bool:
        """Return True if the component is enabled in the current config."""
        return component_name not in self._disabled_components

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def process(self, text: str) -> PipelineContext:
        """Run *text* through all enabled pipeline components.

        Parameters
        ----------
        text : str
            Raw NLACP policy sentence.

        Returns
        -------
        PipelineContext
            Fully populated context after all components have run.
        """
        doc = self.nlp(text)
        context = PipelineContext(
            original_text=text,
            doc=doc,
            metadata={"pipeline_version": self._pipeline_version},
        )

        for component in self.components:
            # Honor the enable/disable config at runtime
            if not self.is_enabled(component.name):
                logger.debug("Skipping disabled component: %s", component.name)
                continue

            logger.debug("Running pipeline component: %s", component.name)
            try:
                context = component.process(context)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error in component '%s': %s", component.name, exc)
                context.warnings.append(f"Component '{component.name}' failed: {exc}")

        return context

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_spacy_model(model_name: str):
        """Load a spaCy model, downloading it automatically if not found."""
        try:
            return spacy.load(model_name)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found. Attempting to download...", model_name
            )
            spacy.cli.download(model_name)
            return spacy.load(model_name)

    @property
    def pipeline_version(self) -> str:
        """Pipeline version string from config."""
        return self._pipeline_version
