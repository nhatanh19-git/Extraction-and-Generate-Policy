"""Batch processor for loading .txt input files and exporting structured extraction results to JSON."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..extractor.attribute_extractor import AttributeExtractor
from ..nlp.nlp_engine import NLPEngine

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processes multiple ACP policy sentences from a text file."""

    def __init__(self):
        self.nlp_engine = NLPEngine()
        self.extractor = AttributeExtractor()

    def process_txt_file(self, txt_file_path: str | Path, output_json_path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
        """
        Read sentences from a .txt file (one policy per line), run attribute extraction,
        and export complete structured results to JSON file.
        """
        txt_path = Path(txt_file_path)
        if not txt_path.exists():
            raise FileNotFoundError(f"Input file not found: {txt_file_path}")

        lines = txt_path.read_text(encoding="utf-8").splitlines()
        sentences = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

        results = []
        for idx, sentence in enumerate(sentences, start=1):
            doc = self.nlp_engine.process(sentence)
            extraction = self.extractor.extract(doc)
            item = {
                "id": idx,
                "sentence": sentence,
                "subject": extraction.subject.text if extraction.subject else "",
                "action": extraction.action.text if extraction.action else "",
                "object": extraction.object.text if extraction.object else "",
                "environment": extraction.environment,
            }
            results.append(item)

        if output_json_path:
            out_p = Path(output_json_path)
            out_p.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"Batch results saved to {out_p}")

        return results
