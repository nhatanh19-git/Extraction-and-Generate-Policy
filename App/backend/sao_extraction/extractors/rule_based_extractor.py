"""Rule-based SAO extractor strategy (Version 1 — Pure NLP).

[EXTENDED] – ``RuleBasedSAOExtractor`` implements ``SAOExtractorStrategy``
by assembling the full dependency-parsing pipeline from components in
``sao_extraction/components/``. It is the sole extraction strategy in
Version 1; future strategies (CNN, SRL, LLM) will be added as sibling
classes in this package and registered in the pipeline orchestrator.

Architecture notes
------------------
* Follows Strategy Pattern: ``SAOExtractorStrategy`` interface is stable;
  adding a new extractor never requires modifying this file.
* Gazetteer is injected (``FileBasedGazetteer``) — can be swapped for
  ``DatabaseGazetteer`` or ``OntologyGazetteer`` without touching this class.
* Pipeline config path is injected — defaults to the standard YAML location
  relative to this file, but can be overridden for testing.
* All text normalization (contraction expansion) happens inside the pipeline
  via ``PreprocessingComponent`` before spaCy parsing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..core.interfaces import SAOExtractorStrategy
from ..core.pipeline import SAOPipeline
from ..core.schemas import PipelineResult
from ..components.preprocessing import PreprocessingComponent
from ..components.clause_segmentation import ClauseSegmenter
from ..components.voice_polarity import VoicePolarityDetector
from ..components.subject_extraction import SubjectExtractor
from ..components.action_extraction import ActionExtractor
from ..components.object_extraction import ObjectExtractor
from ..components.attribute_attachment import AttributeConstraintExtractor
from ..components.gazetteer_validation import GazetteerValidator
from ..gazetteers.provider import FileBasedGazetteer

logger = logging.getLogger(__name__)

# Default config path, relative to this file's location in the package tree
_DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "config" / "pipeline_config.yaml"
)

# Default gazetteer data directory
_DEFAULT_GAZETTEER_DIR = (
    Path(__file__).resolve().parent.parent / "gazetteers" / "data"
)


def _build_gazetteer(data_dir: Path) -> FileBasedGazetteer:
    """Load all JSON Gazetteer files from *data_dir* into a single provider.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``effect_terms.json``, ``subject_roles.json``,
        and ``resource_types.json``.

    Returns
    -------
    FileBasedGazetteer
        Provider with all three namespaces loaded.
    """
    gazetteer = FileBasedGazetteer()
    for json_file in sorted(data_dir.glob("*.json")):
        gazetteer.load(str(json_file))
        logger.debug("Loaded gazetteer: %s", json_file.name)
    return gazetteer


class RuleBasedSAOExtractor(SAOExtractorStrategy):
    """Pure dependency-parsing SAO extractor (Version 1 — Rule-Based/NLP-Only).

    Assembles a ``SAOPipeline`` with the following component order:
    1. PreprocessingComponent      (contraction expansion, normalization)
    2. ClauseSegmenter             (independent clause splitting)
    3. VoicePolarityDetector       (active/passive, Permit/Deny, negation)
    4. SubjectExtractor            (nsubj / agent-pobj + NP expansion)
    5. ActionExtractor             (xcomp priority, phrasal verb prt)
    6. ObjectExtractor             (dobj/attr/oprd, prep>pobj, conj)
    7. AttributeConstraintExtractor(PP-modifier + relcl constraints)
    8. GazetteerValidator          (confidence scoring + triplet assembly)

    Components can be individually disabled via ``pipeline_config.yaml`` for
    ablation studies (e.g. disabling ``attribute_attachment`` for benchmarking).

    Parameters
    ----------
    config_path : str | Path | None
        Path to ``pipeline_config.yaml``. Defaults to the standard location
        inside the ``sao_extraction/config/`` directory.
    gazetteer_dir : str | Path | None
        Directory containing Gazetteer JSON files. Defaults to
        ``sao_extraction/gazetteers/data/``.

    # [HOOK] Ensemble/fallback: the SAOPipeline can run multiple strategies
    # on the same text. Add a second strategy via pipeline.add_strategy() in
    # a future version without modifying this class.
    """

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        gazetteer_dir: Optional[str | Path] = None,
    ) -> None:
        cfg = Path(config_path) if config_path else _DEFAULT_CONFIG
        data_dir = Path(gazetteer_dir) if gazetteer_dir else _DEFAULT_GAZETTEER_DIR

        # Build and load Gazetteer
        self._gazetteer = _build_gazetteer(data_dir)

        # Build pipeline from config
        self._pipeline = SAOPipeline.from_config(cfg)
        self._register_components()

        logger.info(
            "RuleBasedSAOExtractor initialized (version=%s, model=%s).",
            self._pipeline.pipeline_version,
            self._pipeline.model_name,
        )

    def _register_components(self) -> None:
        """Register pipeline components in the correct processing order."""
        self._pipeline.add_component(PreprocessingComponent())
        self._pipeline.add_component(ClauseSegmenter())
        self._pipeline.add_component(VoicePolarityDetector(self._gazetteer))
        self._pipeline.add_component(SubjectExtractor())
        self._pipeline.add_component(ActionExtractor())
        self._pipeline.add_component(ObjectExtractor())
        self._pipeline.add_component(AttributeConstraintExtractor())
        self._pipeline.add_component(GazetteerValidator(self._gazetteer))

    @property
    def strategy_name(self) -> str:
        return "rule_based"

    def extract(self, text: str) -> PipelineResult:
        """Extract SAO triplets from an NLACP sentence.

        The input text is first normalized by ``PreprocessingComponent``
        (contraction expansion). The normalized text is then re-parsed by
        spaCy before the remaining components run. This two-pass approach
        ensures the dependency tree is built on clean text.

        Parameters
        ----------
        text : str
            Raw natural language access control policy sentence.

        Returns
        -------
        PipelineResult
            Contains all extracted triplets, pipeline version, and warnings.
        """
        if not text or not text.strip():
            return PipelineResult(
                original_text=text,
                triplets=[],
                pipeline_version=self._pipeline.pipeline_version,
                warnings=["Empty input text."],
            )

        # Pass 1: preprocessing only (no NLP) to expand contractions
        from ..components.preprocessing import expand_contractions, normalize_whitespace
        normalized = normalize_whitespace(expand_contractions(text))

        # Pass 2: full pipeline on the normalized text
        # Re-parse with the normalized text so the spaCy Doc is correct
        context = self._pipeline.process(normalized)
        # Preserve the original (pre-normalization) text in the result
        context.original_text = text

        return PipelineResult(
            original_text=text,
            triplets=context.triplet_candidates,
            pipeline_version=self._pipeline.pipeline_version,
            warnings=context.warnings,
        )
